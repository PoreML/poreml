"""Re-encode the solver's gzip HDF5 runs to zstd-3 under poreml/data, bit-identical and verified.

the solver writes one gzip-4 HDF5 per run. gzip decodes at ~134 ms per 128^3 frame on one core,
and a training window needs two frames, so with the loader workers a node can spare the
voxel models trained 3-6x slower than their GPU rate. zstd-3 through the `hdf5plugin`
filter gives the same file size and a 3.6x faster decode (37 ms; measured on a 128^3
frame, 2026-09-02). Nothing else about the file changes: every group, dataset, attribute,
dtype, shape and chunk layout is copied as is, NaN solid fill included, and the copy is
read back and compared against the source before it is renamed into place.

The source tree (`<src>/<campaign>/runs/<family>/<size>/<run_id>/`) is never modified;
the destination mirrors it so `data.discover` works on either root. Beside the copy go
the run's small sidecars (`run_meta*.json`, geometry, xdmf, metrics, report) and a
`<run_id>.conversion.json` record (source path, size and mtime, sha256 of the copy,
filter, versions, SLURM job). Checkpoints, frames and GIFs stay behind.

Scope: finished runs of the 128-class sizes (`128`, `128x128x64`), the 256-class sizes (`256`,
`256x256x128`; the Scale-Up test domains, added 2026-09-07) and every underfill run.
`scan` classifies every source run; `write_ledger` turns that into the solver's
`<case root>/h5_todo.md` — what is not converted and why.
"""

import getpass
import hashlib
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np

CONVERSION_SUFFIX = ".conversion.json"
ZSTD_LEVEL = 3
ZSTD_FILTER_ID = "32015"
SCOPE_SIZES = frozenset({"128", "128x128x64", "256", "256x256x128"})  # 256 classes added 2026-09-07 for case/scale
ALL_SIZES_CAMPAIGNS = frozenset({"underfill"})
SIDECAR_GLOBS = ("run_meta*.json", "*_geometry.json", "*.npy", "*.xdmf", "metrics*.csv", "*report*.md")


def in_scope(campaign: str, size: str) -> bool:
    """The 128- and 256-class domains of every campaign, and underfill whatever its size."""
    return campaign in ALL_SIZES_CAMPAIGNS or size in SCOPE_SIZES


def run_status(run_dir: Path) -> str:
    """'finished' only when every run_meta*.json in the directory says so (trapping runs
    carry a flood and a stab meta); otherwise the first other status, or 'missing'."""
    metas = sorted(run_dir.glob("run_meta*.json"))
    if not metas:
        return "missing"
    statuses = [str(json.loads(m.read_text()).get("run", {}).get("status", "unknown")) for m in metas]
    return next((s for s in statuses if s != "finished"), "finished")


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def convert_h5(src: Path, dst: Path, clevel: int = ZSTD_LEVEL) -> int:
    """Copy `src` to `dst` object by object, every chunked dataset re-filtered to zstd.
    Returns the number of objects copied (groups + datasets, root excluded)."""
    count = 0
    with h5py.File(src, "r") as s, h5py.File(dst, "w") as d:
        _copy_attrs(s, d)

        def visit(name: str, obj) -> None:
            nonlocal count
            count += 1
            if isinstance(obj, h5py.Group):
                _copy_attrs(obj, d.create_group(name))
                return
            kwargs: dict[str, Any] = {}
            if obj.chunks is not None:
                kwargs.update(chunks=obj.chunks, fillvalue=obj.fillvalue, **hdf5plugin.Zstd(clevel=clevel))
                if obj.shuffle:
                    kwargs["shuffle"] = True
                if obj.fletcher32:
                    kwargs["fletcher32"] = True
            ds = d.create_dataset(name, shape=obj.shape, dtype=obj.dtype, data=obj[()], **kwargs)
            _copy_attrs(obj, ds)

        s.visititems(visit)
    return count


def _values_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _attr_problems(name: str, a, b) -> list[str]:
    problems = []
    for key in a.attrs.keys() - b.attrs.keys():
        problems.append(f"{name or '/'}: attr {key!r} missing in copy")
    for key in b.attrs.keys() - a.attrs.keys():
        problems.append(f"{name or '/'}: attr {key!r} extra in copy")
    for key in a.attrs.keys() & b.attrs.keys():
        if not _values_equal(a.attrs[key], b.attrs[key]):
            problems.append(f"{name or '/'}: attr {key!r} differs")
    return problems


def verify_h5(original: Path, copy: Path) -> list[str]:
    """Every difference between the two files: objects, attributes, dtype, shape, chunks,
    values (NaN equals NaN). An empty list means the copy is the original, bit for bit."""
    problems: list[str] = []
    with h5py.File(original, "r") as a, h5py.File(copy, "r") as b:
        names_a: list[str] = []
        a.visit(names_a.append)
        names_b: list[str] = []
        b.visit(names_b.append)
        for name in sorted(set(names_a) - set(names_b)):
            problems.append(f"{name}: missing in copy")
        for name in sorted(set(names_b) - set(names_a)):
            problems.append(f"{name}: extra in copy")
        problems += _attr_problems("", a, b)
        for name in sorted(set(names_a) & set(names_b)):
            oa, ob = a[name], b[name]
            problems += _attr_problems(name, oa, ob)
            if isinstance(oa, h5py.Dataset) != isinstance(ob, h5py.Dataset):
                problems.append(f"{name}: group/dataset mismatch")
                continue
            if isinstance(oa, h5py.Dataset):
                if oa.shape != ob.shape or oa.dtype != ob.dtype:
                    problems.append(f"{name}: shape/dtype {oa.shape} {oa.dtype} != {ob.shape} {ob.dtype}")
                    continue
                if oa.chunks != ob.chunks:
                    problems.append(f"{name}: chunks {oa.chunks} != {ob.chunks}")
                if not _values_equal(oa[()], ob[()]):
                    problems.append(f"{name}: values differ")
    return problems


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _record_path(dst_dir: Path, run_id: str) -> Path:
    return dst_dir / f"{run_id}{CONVERSION_SUFFIX}"


def _source_h5(run_dir: Path) -> Path:
    h5s = sorted(run_dir.glob("*.h5"))
    if len(h5s) != 1:
        raise RuntimeError(f"{run_dir}: expected exactly one .h5, found {[p.name for p in h5s]}")
    return h5s[0]


def _up_to_date(record_path: Path, dst_h5: Path, src_h5: Path) -> dict[str, Any] | None:
    """The existing record when the copy still matches the source's size and mtime."""
    if not (record_path.is_file() and dst_h5.is_file()):
        return None
    record = json.loads(record_path.read_text())
    stat = src_h5.stat()
    if record.get("source_size") == stat.st_size and record.get("source_mtime_ns") == stat.st_mtime_ns:
        return record
    return None


def convert_run(src_dir: Path, dst_dir: Path, clevel: int = ZSTD_LEVEL) -> dict[str, Any]:
    """Convert one run directory into `dst_dir`. Refuses a run the solver has not finished, skips a
    copy that is already up to date, and never leaves a torn or unverified file behind."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    status = run_status(src_dir)
    if status != "finished":
        raise RuntimeError(f"{src_dir.name} is {status}, not finished; refusing to convert a run the solver may still change")
    src_h5 = _source_h5(src_dir)
    dst_h5 = dst_dir / src_h5.name
    record_path = _record_path(dst_dir, src_dir.name)
    existing = _up_to_date(record_path, dst_h5, src_h5)
    if existing is not None:
        return {**existing, "skipped": True}

    started = time.perf_counter()
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir / f"{src_h5.name}.tmp"
    tmp.unlink(missing_ok=True)
    try:
        n_objects = convert_h5(src_h5, tmp, clevel)
        problems = verify_h5(src_h5, tmp)
        if problems:
            raise RuntimeError(f"{src_dir.name}: verification failed, copy discarded: " + "; ".join(problems[:10]))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dst_h5)
    for pattern in SIDECAR_GLOBS:
        for path in sorted(src_dir.glob(pattern)):
            if path.is_file():
                shutil.copy2(path, dst_dir / path.name)
    stat = src_h5.stat()
    record = {
        "run_id": src_dir.name,
        "source": str(src_h5),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "destination": str(dst_h5),
        "destination_size": dst_h5.stat().st_size,
        "sha256": _sha256(dst_h5),
        "filter": {"zstd": clevel},
        "verified": True,
        "n_objects": n_objects,
        "converted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - started, 1),
        "h5py": h5py.__version__,
        "hdf5plugin": hdf5plugin.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    record_path.write_text(json.dumps(record, indent=2))
    return {**record, "skipped": False}


def scan(src_root: Path, dst_root: Path) -> list[dict[str, Any]]:
    """One entry per source run: where it is, what the solver says about it, and whether its
    zstd copy exists and is current. States: converted | pending | not finished |
    out of scope | no h5 | unexpected layout."""
    src_root, dst_root = Path(src_root), Path(dst_root)
    entries: list[dict[str, Any]] = []
    metas = sorted({p.parent for p in src_root.glob("*/runs/**/run_meta*.json")})
    for run_dir in metas:
        rel = run_dir.relative_to(src_root)
        parts = rel.parts
        entry: dict[str, Any] = {"run_id": run_dir.name, "src": str(run_dir), "rel": str(rel)}
        if len(parts) != 5 or parts[1] != "runs":
            entries.append(
                {**entry, "campaign": parts[0], "family": "?", "size": "?", "state": "unexpected layout", "reason": str(rel)}
            )
            continue
        campaign, _, family, size, _ = parts
        entry.update(campaign=campaign, family=family, size=size)
        status = run_status(run_dir)
        entry["status"] = status
        if not in_scope(campaign, size):
            entry.update(
                state="out of scope", reason=f"{size}-class domain; only {sorted(SCOPE_SIZES)} and underfill are converted"
            )
        elif status != "finished":
            entry.update(state="not finished", reason=f"the solver reports status {status!r}; the file can still change")
        else:
            h5s = sorted(run_dir.glob("*.h5"))
            if len(h5s) != 1:
                entry.update(state="no h5", reason=f"{len(h5s)} .h5 files in the run directory")
            else:
                dst_dir = dst_root / rel
                current = _up_to_date(_record_path(dst_dir, run_dir.name), dst_dir / h5s[0].name, h5s[0])
                if current is not None:
                    entry.update(state="converted", reason="", dst=str(dst_dir / h5s[0].name))
                elif _record_path(dst_dir, run_dir.name).is_file():
                    entry.update(
                        state="pending", reason="copy exists but the source changed since (the run was resumed); will be redone"
                    )
                else:
                    entry.update(state="pending", reason="finished and in scope; not converted yet")
        entries.append(entry)
    return entries


def write_ledger(entries: list[dict[str, Any]], path: Path, dst_root: Path) -> None:
    """the solver's `<case root>/h5_todo.md`: why the conversion exists, what it covers, and every
    run that is not converted with the reason. Converted runs are counted, not listed."""
    campaigns = sorted({e["campaign"] for e in entries})
    states = ("converted", "pending", "not finished", "out of scope", "no h5", "unexpected layout")
    lines = [
        "# h5_todo.md — zstd conversion ledger for the case root",
        "",
        f"Generated by `poreml convert ledger` on {datetime.now(UTC).isoformat(timespec='seconds')}. Do not edit by hand;",
        "rerun the command after every conversion pass.",
        "",
        "## Why the conversion",
        "",
        "the solver writes each run as one gzip-4 HDF5. gzip decodes at ~134 ms per 128^3 frame on one core and a training",
        "window needs two frames, so with the CPU cores a node can spare per GPU the voxel models train 3-6x slower",
        "than their GPU rate (measured 2026-09-02, H200). Re-encoding the same datasets with the zstd-3 HDF5 filter",
        "(hdf5plugin, filter id 32015) keeps the file size (11.8 vs 11.2 MB per frame) and decodes 3.6x faster (37 ms).",
        "Every group, dataset, attribute, dtype, shape and chunk layout is copied unchanged, NaN solid fill included,",
        "and every copy is read back and compared with the original before it is renamed into place.",
        "",
        "## Where the copies are",
        "",
        f"`{dst_root}` mirrors this tree (`<campaign>/runs/<family>/<size>/<run_id>/`): the zstd `.h5`, the run's",
        "`run_meta*.json`, geometry, xdmf, metrics and report files, and a `<run_id>.conversion.json` record with the",
        "source path, size, mtime, the copy's sha256 and the versions used. Checkpoints, frames and GIFs are not copied.",
        "**The originals here are never modified.** the solver still writes gzip, so a run that finishes after a conversion",
        "pass shows up below as pending until the next `poreml convert submit`.",
        "",
        "## Scope",
        "",
        f"Finished runs whose size directory is one of {sorted(SCOPE_SIZES)}, plus every {sorted(ALL_SIZES_CAMPAIGNS)} run.",
        "Trapping runs count as finished only when both `run_meta_flood.json` and `run_meta_stab.json` say so.",
        "",
        "## Summary",
        "",
        "| campaign | " + " | ".join(states) + " |",
        "|---|" + "---:|" * len(states),
    ]
    for campaign in campaigns:
        counts = [sum(1 for e in entries if e["campaign"] == campaign and e["state"] == s) for s in states]
        lines.append(f"| {campaign} | " + " | ".join(str(c) for c in counts) + " |")
    lines += ["", "## Not converted", "", "| campaign | family/size | run_id | state | reason |", "|---|---|---|---|---|"]
    for e in entries:
        if e["state"] != "converted":
            lines.append(
                f"| {e['campaign']} | {e.get('family', '?')}/{e.get('size', '?')} | {e['run_id']} | "
                f"{e['state']} | {e['reason']} |"
            )
    if all(e["state"] == "converted" for e in entries):
        lines.append("| — | — | — | — | everything in scope is converted |")
    Path(path).write_text("\n".join(lines) + "\n")


# --- anonymising the mirror for release --------------------------------------------------
#
# the solver records where and by whom a run was produced: the working directory and argument list
# (`run.cwd`, `run.argv`), the geometry's source path and the node name in every
# `run_meta*.json`, the same block again as the HDF5 root attribute `run_meta`, and the
# conversion record's absolute `source` / `destination` paths and SLURM job id. Releasing the
# mirror as an anonymous dataset means rewriting exactly those strings in place (2026-09-13:
# the cluster username -> `poreml_author`, the job id -> null). Everything else is untouched:
# the JSON files are edited as text so the solver's formatting survives, the HDF5 edit is one root
# attribute (a variable-length string, so it may grow), and every dataset keeps its bytes.
# The copy's `sha256` in the conversion record is recomputed afterwards so the field keeps
# meaning "this file"; the pre-edit hash — the one the conversion verified — is kept under
# `anonymised.sha256_before`. Never run the HDF5 pass while anything reads the tree: HDF5 has
# no writer-beside-readers mode outside SWMR.

# The default scrubs whoever runs it: the login name of the current user -> `poreml_author`.
DEFAULT_REPLACEMENTS = {getpass.getuser(): "poreml_author"}
NULLED_RECORD_KEYS = ("slurm_job_id",)


def anonymise_text(text: str, replacements: dict[str, str]) -> tuple[str, int]:
    """`text` with every replacement applied and the number of occurrences replaced."""
    n = 0
    for old, new in replacements.items():
        n += text.count(old)
        text = text.replace(old, new)
    return text, n


def anonymise_json(path: Path, replacements: dict[str, str], *, dry_run: bool = False) -> int:
    """Rewrite one JSON sidecar in place (atomically); returns the number of changes made.

    Text replacement keeps the solver's formatting byte for byte elsewhere. A conversion record is
    additionally parsed so its `NULLED_RECORD_KEYS` can be set to null (it is poreml's own file,
    written with `indent=2`, so re-dumping it changes nothing else).
    """
    original = path.read_text()
    text, n = anonymise_text(original, replacements)
    if path.name.endswith(CONVERSION_SUFFIX):
        record = json.loads(text)
        for key in NULLED_RECORD_KEYS:
            if record.get(key) is not None:
                record[key] = None
                n += 1
        if n:
            text = json.dumps(record, indent=2)
    if n and not dry_run:
        _write_text_atomic(path, text)
    return n


def anonymise_h5(path: Path, replacements: dict[str, str], *, dry_run: bool = False, deep: bool = False) -> int:
    """Rewrite the string attributes of the root group (every object with `deep`); returns how many changed.

    Opened without HDF5 file locking: the tree lives on NFS, where the lock protocol is not reliable,
    and the caller guarantees there is no other reader or writer.
    """

    def _changes(obj) -> dict[str, str]:
        out = {}
        for key, value in obj.attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if isinstance(value, str):
                text, n = anonymise_text(value, replacements)
                if n:
                    out[key] = text
        return out

    changed = 0
    with h5py.File(path, "r" if dry_run else "r+", locking=False) as f:
        targets = [f]
        if deep:
            f.visititems(lambda _name, obj: targets.append(obj))
        for obj in targets:
            for key, text in _changes(obj).items():
                if not dry_run:
                    obj.attrs[key] = text  # variable-length UTF-8, so a longer string fits
                changed += 1
    return changed


def anonymise_run(
    run_dir: Path,
    replacements: dict[str, str] = DEFAULT_REPLACEMENTS,
    *,
    json_files: bool = True,
    h5: bool = True,
    rehash: bool = True,
    dry_run: bool = False,
    deep: bool = False,
) -> dict[str, Any]:
    """Anonymise one mirrored run directory; returns what changed.

    With `h5`, a changed HDF5 file gets its conversion record updated: `sha256` recomputed
    (`rehash`, else set to null — a stale hash is worse than none) and an `anonymised` block
    with the pre-edit hash, the replacements and the time.
    """
    out: dict[str, Any] = {"run_dir": str(run_dir), "json_replacements": 0, "h5_attrs_changed": 0, "rehashed": 0}
    if json_files:
        for path in sorted(run_dir.glob("*.json")):
            out["json_replacements"] += anonymise_json(path, replacements, dry_run=dry_run)
    if h5:
        h5s = sorted(run_dir.glob("*.h5"))
        records = sorted(run_dir.glob(f"*{CONVERSION_SUFFIX}"))
        for h5_path in h5s:
            record_path = next((r for r in records if r.name == h5_path.stem + CONVERSION_SUFFIX or len(records) == 1), None)
            before = json.loads(record_path.read_text()).get("sha256") if record_path else None
            changed = anonymise_h5(h5_path, replacements, dry_run=dry_run, deep=deep)
            out["h5_attrs_changed"] += changed
            if changed and record_path and not dry_run:
                record = json.loads(record_path.read_text())
                record["sha256"] = _sha256(h5_path) if rehash else None
                record["anonymised"] = {  # never the old strings: this record ships with the data
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "replaced_with": sorted(set(replacements.values())),
                    "sha256_before": before,
                    "h5_attrs_changed": changed,
                }
                _write_text_atomic(record_path, json.dumps(record, indent=2))
                out["rehashed"] += int(rehash)
    return out


def anonymise_tree(
    root: Path,
    replacements: dict[str, str] = DEFAULT_REPLACEMENTS,
    *,
    json_files: bool = True,
    h5: bool = True,
    rehash: bool = True,
    dry_run: bool = False,
    deep: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """`anonymise_run` over every run directory under `root` (those holding a conversion record).

    `workers` > 1 runs the directories in parallel processes — the HDF5 pass is bound by the
    hash's read of every file, not by CPU. Returns the totals and the per-run records.
    """
    root = Path(root)
    run_dirs = sorted({p.parent for p in root.rglob(f"*{CONVERSION_SUFFIX}")})
    kw = dict(json_files=json_files, h5=h5, rehash=rehash, dry_run=dry_run, deep=deep)
    if workers > 1 and len(run_dirs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as pool:
            runs = list(pool.map(_anonymise_run_star, [(d, replacements, kw) for d in run_dirs]))
    else:
        runs = [anonymise_run(d, replacements, **kw) for d in run_dirs]
    totals = {k: sum(r[k] for r in runs) for k in ("json_replacements", "h5_attrs_changed", "rehashed")}
    return {
        "root": str(root),
        "runs": len(runs),
        "dry_run": dry_run,
        "replacements": dict(replacements),
        **totals,
        "per_run": runs,
    }


def _anonymise_run_star(args: tuple) -> dict[str, Any]:
    run_dir, replacements, kw = args
    return anonymise_run(run_dir, replacements, **kw)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    shutil.copymode(path, tmp)
    tmp.replace(path)


# --- repacking without the derivable fields ---------------------------------------------
#
# the solver writes five fields per step; two are exact functions of the others (2026-09-13 audit):
# `rho` = `p` / `cs2` (the solver computes p from rho with the constant lattice cs2, so the
# relation holds to float32 rounding, same NaN mask) and `umag` = |`u`| (to 2e-8). Nothing in
# poreml reads either, and together they are ~32 % of every file. `repack_run` rewrites a
# run's HDF5 without them: every kept dataset is copied by H5Ocopy, raw compressed chunks
# and all — no decode, no re-encode, same chunking and filters, verifiable by storage size —
# the root attributes are anonymised on the way (`anonymise_text`), `fields` loses the two
# names, the XDMF index loses their blocks, and the conversion record gets the new size and
# hash plus a `repacked` block with the old ones. The two identities are asserted on a few
# steps of the very file first and the run is refused if they fail. Deleting a dataset in
# place would free nothing — HDF5 does not reclaim space — hence the copy, written beside the
# original and renamed over it only after verification. No reader may have the file open.

DROPPED_FIELDS = ("rho", "umag")
REPACK_TMP_SUFFIX = ".repack.tmp"  # must not end in .h5: discovery wants exactly one *.h5 per run


def _sample_steps(f: h5py.File, n: int = 3) -> list[str]:
    steps = sorted(f["steps"].keys()) if "steps" in f else []
    if len(steps) <= n:
        return steps
    picks = {steps[0], steps[len(steps) // 2], steps[-1]}
    return sorted(picks)


def derivable_problems(f: h5py.File, drop: tuple[str, ...] = DROPPED_FIELDS) -> list[str]:
    """Why the `drop` fields could *not* be recomputed from what stays, on sampled steps; empty when they can."""
    problems: list[str] = []
    cs2 = float(f.attrs["cs2"]) if "cs2" in f.attrs else None
    for step in _sample_steps(f):
        g = f["steps"][step]
        if "rho" in drop and "rho" in g:
            if cs2 is None:
                problems.append("no cs2 root attribute: rho cannot be derived from p")
            else:
                p, rho = g["p"][()], g["rho"][()]
                finite = np.isfinite(p) & np.isfinite(rho)
                if not np.array_equal(np.isnan(p), np.isnan(rho)) or not np.allclose(
                    p[finite], rho[finite] * cs2, rtol=1e-5, atol=1e-6
                ):
                    problems.append(f"steps/{step}: rho is not p / cs2")
        if "umag" in drop and "umag" in g:
            umag, u = g["umag"][()], g["u"][()]
            norm = np.sqrt((u.astype(np.float64) ** 2).sum(axis=-1))
            finite = np.isfinite(umag)
            if np.isnan(umag).sum() != np.isnan(norm).sum() or not np.allclose(
                umag[finite], norm[finite], rtol=1e-5, atol=1e-6
            ):
                problems.append(f"steps/{step}: umag is not |u|")
    return problems


def _copy_without(src_group, dst_group, drop: tuple[str, ...], counter: dict[str, int]) -> None:
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Dataset):
            if name in drop:
                counter["dropped"] += 1
                continue
            src_group.copy(obj, dst_group, name=name)  # H5Ocopy: raw chunks, attributes, layout as they are
        else:
            _copy_attrs(obj, dst_group.create_group(name))
            _copy_without(obj, dst_group[name], drop, counter)
        counter["objects"] += 1


def _fields_without(value, drop: tuple[str, ...]) -> str:
    text = value.decode() if isinstance(value, bytes) else str(value)
    return ",".join(x for x in text.split(",") if x not in drop)


def repack_h5(
    src: Path, dst: Path, *, drop: tuple[str, ...] = DROPPED_FIELDS, replacements: dict[str, str] | None = None
) -> dict[str, int]:
    """Write `dst` as `src` without the `drop` datasets, root attributes anonymised and `fields` trimmed."""
    replacements = replacements or {}
    counter = {"objects": 0, "dropped": 0, "attrs_changed": 0}
    with h5py.File(src, "r", locking=False) as s, h5py.File(dst, "w") as d:
        _copy_attrs(s, d)
        for key, value in list(d.attrs.items()):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if isinstance(value, str):
                text, n = anonymise_text(value, replacements)
                if n:
                    d.attrs[key] = text
                    counter["attrs_changed"] += 1
        if "fields" in d.attrs:
            d.attrs["fields"] = _fields_without(s.attrs["fields"], drop)
        _copy_without(s, d, drop, counter)
    return counter


def verify_repack(
    original: Path, copy: Path, *, drop: tuple[str, ...] = DROPPED_FIELDS, replacements: dict[str, str] | None = None
) -> list[str]:
    """Every difference between the two files that the repack does not explain."""
    replacements = replacements or {}
    problems: list[str] = []
    with h5py.File(original, "r", locking=False) as a, h5py.File(copy, "r", locking=False) as b:
        names_a: list[str] = []
        a.visit(names_a.append)
        names_b: list[str] = []
        b.visit(names_b.append)
        expected = [n for n in names_a if not (isinstance(a[n], h5py.Dataset) and n.rsplit("/", 1)[-1] in drop)]
        for name in sorted(set(expected) - set(names_b)):
            problems.append(f"{name}: missing in copy")
        for name in sorted(set(names_b) - set(expected)):
            problems.append(f"{name}: unexpected in copy")
        for key in a.attrs.keys() ^ b.attrs.keys():
            problems.append(f"/: attr {key!r} only on one side")
        for key in a.attrs.keys() & b.attrs.keys():
            va, vb = a.attrs[key], b.attrs[key]
            if key == "fields":
                ok = _fields_without(va, drop) == (vb.decode() if isinstance(vb, bytes) else str(vb))
            elif isinstance(va, (str, bytes)):
                va = va.decode("utf-8", errors="replace") if isinstance(va, bytes) else va
                ok = anonymise_text(va, replacements)[0] == (vb.decode() if isinstance(vb, bytes) else str(vb))
            else:
                ok = _values_equal(va, vb)
            if not ok:
                problems.append(f"/: attr {key!r} differs")
        sampled = {f"steps/{s}" for s in _sample_steps(a)}
        for name in sorted(set(expected) & set(names_b)):
            oa, ob = a[name], b[name]
            problems += _attr_problems(name, oa, ob)
            if isinstance(oa, h5py.Dataset) != isinstance(ob, h5py.Dataset):
                problems.append(f"{name}: group/dataset mismatch")
                continue
            if isinstance(oa, h5py.Dataset):
                if (oa.shape, oa.dtype, oa.chunks, oa._filters) != (ob.shape, ob.dtype, ob.chunks, ob._filters):
                    problems.append(f"{name}: layout differs")
                    continue
                if oa.id.get_storage_size() != ob.id.get_storage_size():
                    problems.append(f"{name}: storage size differs — the raw chunks were not copied as they were")
                if (name.rsplit("/", 1)[0] in sampled or "/" not in name) and not _values_equal(oa[()], ob[()]):
                    problems.append(f"{name}: values differ")
    return problems


def _rewrite_xdmf(path: Path, drop: tuple[str, ...]) -> int:
    """Remove the `<Attribute Name="<field>">...</Attribute>` blocks of the dropped fields; returns how many."""
    import re

    text = path.read_text()
    removed = 0
    for name in drop:
        text, n = re.subn(rf'\s*<Attribute Name="{name}"[^>]*>.*?</Attribute>', "", text, flags=re.S)
        removed += n
    if removed:
        _write_text_atomic(path, text)
    return removed


def repack_run(
    run_dir: Path,
    *,
    drop: tuple[str, ...] = DROPPED_FIELDS,
    replacements: dict[str, str] = DEFAULT_REPLACEMENTS,
    drop_npy: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repack one mirrored run without the `drop` fields (see the section comment). Idempotent:
    a run whose conversion record already says `repacked` is skipped. Refuses a run on which
    the dropped fields are not derivable. Never leaves a torn file: the copy is verified and then
    renamed over the original."""
    run_dir = Path(run_dir)
    started = time.perf_counter()
    h5 = _source_h5(run_dir)
    record_path = _record_path(run_dir, run_dir.name)
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    out: dict[str, Any] = {"run_dir": str(run_dir), "h5": h5.name, "dry_run": dry_run, "skipped": False, "dropped": 0}
    if record.get("repacked"):
        return {**out, "skipped": True, "reason": "already repacked"}
    tmp = h5.with_name(h5.name + REPACK_TMP_SUFFIX)
    tmp.unlink(missing_ok=True)
    with h5py.File(h5, "r", locking=False) as f:
        present: list[str] = []
        f.visit(lambda n: present.append(n) if n.rsplit("/", 1)[-1] in drop and isinstance(f[n], h5py.Dataset) else None)
        if present:
            problems = derivable_problems(f, drop)
            if problems:
                raise RuntimeError(f"{run_dir.name}: refusing to drop fields that are not derivable: " + "; ".join(problems))
        out["dropped"] = len(present)
    if dry_run:
        return out if present else {**out, "reason": "fields already gone; the record and XDMF would be completed"}
    if present:
        size_before, sha_before = h5.stat().st_size, record.get("sha256")
        try:
            counter = repack_h5(h5, tmp, drop=drop, replacements=replacements)
            problems = verify_repack(h5, tmp, drop=drop, replacements=replacements)
            if problems:
                raise RuntimeError(f"{run_dir.name}: repack verification failed, copy discarded: " + "; ".join(problems[:10]))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        shutil.copymode(h5, tmp)
        tmp.replace(h5)
    else:
        # Repacked by a task that died between the rename and the record: finish the bookkeeping.
        size_before, sha_before = record.get("destination_size"), record.get("sha256")
        counter = {"objects": 0, "dropped": 0, "attrs_changed": anonymise_h5(h5, replacements)}
        with h5py.File(h5, "r", locking=False) as f:
            f.visit(lambda _n: counter.__setitem__("objects", counter["objects"] + 1))
    out["xdmf_blocks_removed"] = sum(_rewrite_xdmf(p, drop) for p in sorted(run_dir.glob("*.xdmf")))
    if drop_npy:
        with h5py.File(h5, "r", locking=False) as f:
            rock = f["rock"][()].astype(bool)
        for npy in sorted(run_dir.glob("*.npy")):
            if np.array_equal(np.load(npy).astype(bool), rock):
                npy.unlink()
                out["npy_removed"] = out.get("npy_removed", 0) + 1
    for path in sorted(run_dir.glob("*.json")):
        if path != record_path:
            anonymise_json(path, replacements)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with h5py.File(h5, "r", locking=False) as f:
        fields_after = _fields_without(f.attrs["fields"], drop) if "fields" in f.attrs else None
    record.update(
        {
            "destination_size": h5.stat().st_size,
            "sha256": _sha256(h5),
            "n_objects": counter["objects"],
            "repacked": {
                "at": now,
                "dropped": list(drop),
                "fields": fields_after,
                "size_before": size_before,
                "sha256_before": sha_before,
                "n_objects_before": record.get("n_objects"),
            },
        }
    )
    if counter["attrs_changed"]:
        record["anonymised"] = {
            "at": now,
            "replaced_with": sorted(set(replacements.values())),
            "sha256_before": sha_before,
            "h5_attrs_changed": counter["attrs_changed"],
        }
    for key in NULLED_RECORD_KEYS:
        if record.get(key) is not None:
            record[key] = None
    _write_text_atomic(record_path, json.dumps(record, indent=2)) if record_path.is_file() else record_path.write_text(
        json.dumps(record, indent=2)
    )
    return {
        **out,
        "size_before": size_before,
        "size_after": record["destination_size"],
        "sha256": record["sha256"],
        "attrs_changed": counter["attrs_changed"],
        "seconds": round(time.perf_counter() - started, 1),
    }
