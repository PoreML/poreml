"""Stage the trained checkpoints of `case/train` and `case/train_push` for a double-blind release.

    uv run python util/release/stage_checkpoints.py [--out data/release/checkpoints] [--dry-run]
    uv run python util/release/upload_hf.py --repo PoreML/PoreML_checkpoint --repo-type model \
        --folder data/release/checkpoints

The staged tree mirrors `case/` — `<phase>/<campaign>/<model>_<kind>/ckpts/<run>/...` — so that
`poreml checkpoints` downloads into `case/` and everything that looks a run up (the push
configs' `train.init_from`, the studies, `util/inference`) finds it with nothing to move.

Per **finished** run it takes `config.yaml`, `run_meta.json`, `metrics.csv`, `train_log.csv` and
the weights `ckpts/{best,last,best_rollout}.pt`. Left behind on purpose: failed attempts,
`resume.pt` (optimiser and RNG state of a run that is over), `epoch_<k>.pt` (the rollout
candidates; the winner is already `best_rollout.pt`), and the stored frames, evals and renders,
which `poreml inference|metric|render` reproduce from the weights.

Weights are hard-linked (copied across filesystems), byte-identical: a `.pt` holds the state
dict, the model config, the epoch and its metrics, nothing else, and `run_meta.json` of a push
run records the sha256 of the `best.pt` it started from. The text files are where a run
identifies its authors — absolute paths, the node's hostname — so they are rewritten: the
repository root becomes relative, `provenance.hostname` is dropped. `check_anonymous.py` then
sweeps the staged tree, and staging fails when anything survives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHASES = ("train", "train_push")
TEXT = ("config.yaml", "run_meta.json", "metrics.csv", "train_log.csv")
WEIGHTS = ("best.pt", "last.pt", "best_rollout.pt")
CARD = HERE / "checkpoint_card.md"
INDEX = "checkpoints.csv"

sys.path.insert(0, str(HERE))
from check_anonymous import load_patterns, sweep  # noqa: E402


def finished_runs(case: Path) -> list[Path]:
    """Run directories whose `run_meta.json` says finished, as paths relative to `case`."""
    runs = []
    for phase in PHASES:
        for meta in sorted((case / phase).glob("*/*/ckpts/*/run_meta.json")):
            if json.loads(meta.read_text()).get("status") == "finished":
                runs.append(meta.parent.relative_to(case))
    return runs


def scrub(data: bytes, roots: list[str]) -> bytes:
    """`data` with every absolute root made relative. Longest root first: the repository, then what holds it.

    Bytes, not text: the CSVs end their lines with CRLF and must come out as they went in.
    """
    for root in roots:
        data = data.replace(root.encode() + b"/", b"").replace(root.encode(), b".")
    return data


def scrub_meta(data: bytes, roots: list[str]) -> bytes:
    meta = json.loads(data)
    meta.get("provenance", {}).pop("hostname", None)
    return scrub(json.dumps(meta, indent=2).encode(), roots) + b"\n"


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def index_rows(case: Path, run: Path) -> list[dict]:
    """One row per published weight file: where it is, which epoch it holds, its checksum."""
    phase, campaign, cell = run.parts[:3]
    model, kind = cell.rsplit("_", 1)
    progress = json.loads((case / run / "run_meta.json").read_text()).get("progress", {})
    epochs = {"best.pt": progress.get("best", {}).get("epoch"), "last.pt": progress.get("last", {}).get("epoch")}
    epochs["best_rollout.pt"] = progress.get("best_rollout", {}).get("epoch")
    rows = []
    for name in WEIGHTS:
        path = case / run / "ckpts" / name
        if path.is_file():
            rows.append(
                {
                    "phase": phase,
                    "campaign": campaign,
                    "model": model,
                    "kind": kind,
                    "path": f"{run.as_posix()}/ckpts/{name}",
                    "epoch": epochs[name],
                    "bytes": path.stat().st_size,
                    "sha256": sha256_of(path),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=REPO / "case", help="the folder holding train/ and train_push/")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "release" / "checkpoints")
    ap.add_argument("--deny", type=Path, default=HERE / "deny.txt", help="the sweep's patterns, one regex per line")
    ap.add_argument("--dry-run", action="store_true", help="list what would be staged; write nothing")
    args = ap.parse_args(argv)

    case, out = args.case.resolve(), args.out.resolve()
    runs = finished_runs(case)
    n_weights = sum((case / r / "ckpts" / w).is_file() for r in runs for w in WEIGHTS)
    size = sum((case / r / "ckpts" / w).stat().st_size for r in runs for w in WEIGHTS if (case / r / "ckpts" / w).is_file())
    print(f"{len(runs)} finished run(s), {n_weights} weight file(s), {size / 1e9:.1f} GB  ->  {out}")
    if args.dry_run:
        for r in runs:
            print(f"  {r}")
        return 0
    if out.exists():
        raise SystemExit(f"{out} exists; remove it or name another directory")
    if not args.deny.is_file():
        raise SystemExit(f"{args.deny} is missing: nothing would check the staged tree")

    # The roots a run may have recorded, longest first; anything else absolute is caught below.
    roots = [str(REPO), str(REPO.parent), str(Path.home())]
    rows = []
    for run in runs:
        (out / run / "ckpts").mkdir(parents=True)
        for name in TEXT:
            src = case / run / name
            if src.is_file():
                data = src.read_bytes()
                (out / run / name).write_bytes(scrub_meta(data, roots) if name == "run_meta.json" else scrub(data, roots))
        for name in WEIGHTS:
            if (case / run / "ckpts" / name).is_file():
                link(case / run / "ckpts" / name, out / run / "ckpts" / name)
        rows += index_rows(case, run)
    with (out / INDEX).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(CARD, out / "README.md")

    found = sweep(out, load_patterns(args.deny))
    # The deny list knows names; this knows shapes — a home or scratch path nobody listed.
    stray = re.compile(r"/(?:home|users|mnt|scratch|lustre|gpfs|nfs)/[\w.-]+", re.IGNORECASE)
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.suffix != ".pt"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            found += [(path.relative_to(out), n, m.group(0)) for m in stray.finditer(line)]
    for path, n, hit in found:
        print(f"{path}:{n}: {hit}")
    if found:
        print(f"{len(found)} identifying string(s) survive in {out}: NOT ready to publish")
        return 1
    print(f"staged {len(runs)} runs, {len(rows)} weight files; the sweep found nothing identifying")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
