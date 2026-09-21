"""Re-encoding the solver's gzip HDF5 runs to zstd under poreml/data: bit-identical, verified, ledgered.

The source tree is the solver's `<case root>/<campaign>/runs/<family>/<size>/<run_id>/`; the
destination mirrors it. Only finished runs of the 128-class sizes (and every underfill run)
are in scope; nothing in the source is ever modified.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from poreml.convert import (
    CONVERSION_SUFFIX,
    convert_run,
    in_scope,
    run_status,
    scan,
    verify_h5,
    write_ledger,
)


def write_source_run(
    root: Path,
    campaign: str = "drainage",
    family: str = "blob",
    size: str = "128",
    run_id: str = "drain_blob128_0000",
    status: str = "finished",
    metas: tuple[str, ...] = ("run_meta.json",),
) -> Path:
    """A miniature the solver run: gzip HDF5 with NaN-filled solid, nested groups, attrs on every
    level, plus the sidecars the solver leaves beside it (some of which must not be copied)."""
    run = root / campaign / "runs" / family / size / run_id
    run.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rock = (rng.random((4, 5, 6)) < 0.3).astype(np.uint8)
    with h5py.File(run / f"{run_id}.h5", "w") as f:
        f.attrs["shape"] = np.array([4, 5, 6], dtype="int64")
        f.attrs["fields"] = "phi,p,u"
        f.attrs["run_meta"] = json.dumps({"solver": {"theta": 140.0}})
        f.create_dataset("rock", data=rock, chunks=(1, 5, 6), compression="gzip", compression_opts=4)
        steps = f.create_group("steps")
        steps.attrs["note"] = "two frames"
        for i in range(2):
            g = steps.create_group(f"{i * 100:09d}")
            phi = rng.random((4, 5, 6), dtype=np.float32) * 2 - 1
            phi[rock == 1] = np.nan
            g.create_dataset("phi", data=phi, chunks=(1, 5, 6), compression="gzip", compression_opts=4)
            g["phi"].attrs["units"] = "-"
            g.create_dataset("p", data=np.full((4, 5, 6), 0.33, np.float32), chunks=(1, 5, 6), compression="gzip")
            g.create_dataset("u", data=rng.random((4, 5, 6, 3), dtype=np.float32), chunks=(1, 5, 6, 3), compression="gzip")
        f.create_dataset("scalar_no_filter", data=np.float64(1.5))  # contiguous, no chunks, no filter
    for meta in metas:
        (run / meta).write_text(json.dumps({"run": {"status": status}, "geometry": {"shape": [4, 5, 6]}}))
    (run / f"{run_id}.xdmf").write_text("<Xdmf/>")
    (run / f"{run_id}_geometry.json").write_text("{}")
    (run / f"{run_id}.npy").write_bytes(b"NPY")
    (run / "metrics.csv").write_text("step,sat\n0,0.1\n")
    (run / "report.md").write_text("# run\n")
    (run / "ckpt_step00001000_sat0.5.npz").write_bytes(b"\0" * 1000)  # never copied
    (run / "frames").mkdir()
    (run / f"{run_id}.gif").write_bytes(b"GIF")
    return run


def read_all(path: Path) -> dict:
    out = {}
    with h5py.File(path, "r") as f:

        def visit(name, obj):
            out[name] = (dict(obj.attrs), obj[()] if isinstance(obj, h5py.Dataset) else None)

        out[""] = (dict(f.attrs), None)
        f.visititems(visit)
    return out


@pytest.fixture
def roots(tmp_path):
    src, dst = tmp_path / "solver_case", tmp_path / "data" / "case"
    src.mkdir()
    return src, dst


def test_convert_run_writes_a_zstd_copy_identical_down_to_nans_attrs_and_chunks(roots):
    src, dst = roots
    run = write_source_run(src)

    record = convert_run(run, dst / run.relative_to(src))

    out = dst / run.relative_to(src) / f"{run.name}.h5"
    assert out.is_file() and record["verified"] is True
    a, b = read_all(run / f"{run.name}.h5"), read_all(out)
    assert a.keys() == b.keys()
    for name in a:
        (attrs_a, data_a), (attrs_b, data_b) = a[name], b[name]
        assert attrs_a.keys() == attrs_b.keys()
        for k in attrs_a:
            np.testing.assert_array_equal(attrs_a[k], attrs_b[k])
        if data_a is not None:
            np.testing.assert_array_equal(data_a, data_b)  # NaN == NaN under array_equal semantics here
    with h5py.File(out) as f, h5py.File(run / f"{run.name}.h5") as g:
        for name in ("rock", "steps/000000000/phi", "steps/000000100/u"):
            assert f[name]._filters == {"32015": (3,)}, name
            assert f[name].chunks == g[name].chunks and f[name].dtype == g[name].dtype
        assert f["scalar_no_filter"].chunks is None and f["scalar_no_filter"]._filters == {}
    assert np.isnan(b["steps/000000000/phi"][1]).sum() > 0  # the NaN solid fill survived


def test_convert_run_copies_the_small_sidecars_and_skips_checkpoints_frames_and_gifs(roots):
    src, dst = roots
    run = write_source_run(src)

    convert_run(run, dst / run.relative_to(src))

    names = sorted(p.name for p in (dst / run.relative_to(src)).iterdir())
    assert names == sorted(
        [
            f"{run.name}.h5",
            f"{run.name}{CONVERSION_SUFFIX}",
            "run_meta.json",
            f"{run.name}.xdmf",
            f"{run.name}_geometry.json",
            f"{run.name}.npy",
            "metrics.csv",
            "report.md",
        ]
    )
    record = json.loads((dst / run.relative_to(src) / f"{run.name}{CONVERSION_SUFFIX}").read_text())
    assert record["source"] == str(run / f"{run.name}.h5")
    assert record["filter"] == {"zstd": 3} and record["verified"] is True and len(record["sha256"]) == 64


def test_convert_run_never_touches_the_source(roots):
    src, dst = roots
    run = write_source_run(src)
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in run.iterdir()}

    convert_run(run, dst / run.relative_to(src))

    assert {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in run.iterdir()} == before


def test_convert_run_refuses_a_run_the_solver_has_not_finished(roots):
    src, dst = roots
    run = write_source_run(src, status="running")

    with pytest.raises(RuntimeError, match="running"):
        convert_run(run, dst / run.relative_to(src))
    assert not (dst / run.relative_to(src)).exists()


def test_trapping_needs_both_stage_metas_finished(roots):
    src, dst = roots
    run = write_source_run(src, campaign="trapping", metas=("run_meta_flood.json", "run_meta_stab.json"))
    assert run_status(run) == "finished"
    (run / "run_meta_stab.json").write_text(json.dumps({"run": {"status": "running"}}))
    assert run_status(run) == "running"
    with pytest.raises(RuntimeError):
        convert_run(run, dst / run.relative_to(src))


def test_convert_run_is_idempotent_and_redoes_a_changed_source(roots):
    src, dst = roots
    run = write_source_run(src)
    out = dst / run.relative_to(src) / f"{run.name}.h5"

    first = convert_run(run, dst / run.relative_to(src))
    stamp = out.stat().st_mtime_ns
    second = convert_run(run, dst / run.relative_to(src))

    assert first["skipped"] is False and second["skipped"] is True
    assert out.stat().st_mtime_ns == stamp
    with h5py.File(run / f"{run.name}.h5", "a") as f:  # the solver appended a frame: the copy is stale
        f["steps"].create_group("000000200").create_dataset("phi", data=np.zeros((4, 5, 6), np.float32))
    third = convert_run(run, dst / run.relative_to(src))
    assert third["skipped"] is False and "steps/000000200/phi" in read_all(out)


def test_verify_h5_reports_every_kind_of_difference(roots, tmp_path):
    src, _ = roots
    run = write_source_run(src)
    a = run / f"{run.name}.h5"
    b = tmp_path / "b.h5"
    b.write_bytes(a.read_bytes())
    assert verify_h5(a, b) == []
    with h5py.File(b, "a") as f:
        f["steps/000000000/p"][0, 0, 0] = 0.0
        del f["steps"].attrs["note"]
        del f["steps/000000100/u"]
    problems = verify_h5(a, b)
    assert any("steps/000000000/p" in p and "values" in p for p in problems)
    assert any("note" in p for p in problems)
    assert any("steps/000000100/u" in p and "missing" in p for p in problems)


def test_a_torn_output_is_rejected_not_renamed_into_place(roots, monkeypatch):
    src, dst = roots
    run = write_source_run(src)
    import poreml.convert as convert

    monkeypatch.setattr(convert, "verify_h5", lambda a, b: ["simulated mismatch"])
    with pytest.raises(RuntimeError, match="verification"):
        convert_run(run, dst / run.relative_to(src))
    assert not (dst / run.relative_to(src) / f"{run.name}.h5").exists()
    assert not list((dst / run.relative_to(src)).glob("*.tmp"))


@pytest.mark.parametrize(
    "campaign, size, expected",
    [
        ("drainage", "128", True),
        ("trapping", "128", True),
        ("GDL", "128x128x64", True),
        ("underfill", "26x482x476", True),
        ("drainage", "256", True),
        ("GDL", "256x256x128", True),
        ("drainage", "512", False),
        ("GDL", "64x64x32", False),
    ],
)
def test_in_scope_is_the_128_and_256_classes_plus_every_underfill_run(campaign, size, expected):
    assert in_scope(campaign, size) is expected


def test_scan_classifies_every_source_run_and_the_ledger_explains_each_omission(roots):
    src, dst = roots
    done = write_source_run(src, run_id="drain_blob128_0000")
    convert_run(done, dst / done.relative_to(src))
    write_source_run(src, run_id="drain_blob128_0001")  # finished, not yet converted
    write_source_run(src, run_id="drain_blob128_0002", status="running")
    write_source_run(src, family="blob", size="512", run_id="drain_blob512_0000")
    write_source_run(src, campaign="underfill", family="flipchip", size="26x482x476", run_id="fill_0000")

    entries = scan(src, dst)

    by_id = {e["run_id"]: e for e in entries}
    assert by_id["drain_blob128_0000"]["state"] == "converted"
    assert by_id["drain_blob128_0001"]["state"] == "pending"
    assert by_id["drain_blob128_0002"]["state"] == "not finished"
    assert by_id["drain_blob512_0000"]["state"] == "out of scope"
    assert by_id["fill_0000"]["state"] == "pending"

    ledger = src / "h5_todo.md"
    write_ledger(entries, ledger, dst)
    text = ledger.read_text()
    assert "zstd" in text and "drain_blob128_0002" in text and "drain_blob512_0000" in text
    assert "drain_blob128_0000" not in text.split("## Not converted")[1]  # converted runs are counted, not listed
    assert "512" in text and "running" in text.lower()


def test_poreml_reader_opens_a_converted_run_and_sees_the_same_frames(roots):
    """`data.discover` and `Trajectory` on the mirrored tree: the plugin import in data.py is
    what lets a plain `h5py.File` decode the zstd filter."""
    from poreml.data import Trajectory, discover

    src, dst = roots
    run = write_source_run(src)
    convert_run(run, dst / run.relative_to(src))

    ref = discover(dst, "drainage")[run.name]
    original = Trajectory(run / f"{run.name}.h5")
    copy = Trajectory(ref.h5_path)
    assert copy.steps == original.steps
    for step in copy.steps:
        np.testing.assert_array_equal(copy.frame(step, "phi"), original.frame(step, "phi"))


# ---------------------------------------------------------------------------
# Anonymising the mirror for release: the username in every path, the cluster job id


def _converted_copy_with_identity(roots) -> tuple[Path, Path, Path]:
    """A converted run whose sidecars and HDF5 root attribute carry the cluster identity the solver
    records: home-directory paths in the metadata and the job id in the conversion record."""
    from poreml.convert import anonymise_tree  # noqa: F401  (the module under test must import)

    src, dst = roots
    run = write_source_run(src)
    dst_run = dst / run.relative_to(src)
    convert_run(run, dst_run)
    meta = {
        "run": {
            "status": "finished",
            "cwd": "/home/alice/work/solver",
            "argv": ["/home/alice/work/geometry/data/blob/128/blob128_0000.npy"],
        },
        "environment": {"hostname": "node07"},
    }
    (dst_run / "run_meta.json").write_text(json.dumps(meta))
    h5 = dst_run / f"{run.name}.h5"
    with h5py.File(h5, "r+") as f:
        f.attrs["run_meta"] = json.dumps(meta)
    from poreml.convert import _sha256

    record_path = dst_run / f"{run.name}{CONVERSION_SUFFIX}"
    record = json.loads(record_path.read_text())
    record["slurm_job_id"] = "32795"
    record["sha256"] = _sha256(h5)  # the record describes the file as it is now, as on the real mirror
    record_path.write_text(json.dumps(record, indent=2))
    return dst, dst_run, h5


def _json_texts(run_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(run_dir.glob("*.json"))}


def test_anonymise_replaces_the_username_everywhere_and_nulls_the_job_id(roots):
    from poreml.convert import _sha256, anonymise_tree

    dst, dst_run, h5 = _converted_copy_with_identity(roots)
    before = read_all(h5)
    sha_before = _sha256(h5)

    summary = anonymise_tree(dst, {"alice": "poreml_author"})

    assert summary["runs"] == 1 and summary["json_replacements"] > 0 and summary["h5_attrs_changed"] == 1
    for name, text in _json_texts(dst_run).items():
        assert "alice" not in text, name
        json.loads(text)  # still valid JSON
    meta = json.loads((dst_run / "run_meta.json").read_text())
    assert meta["run"]["cwd"] == "/home/poreml_author/work/solver"
    after = read_all(h5)
    assert "alice" not in after[""][0]["run_meta"] and "/home/poreml_author/work/solver" in after[""][0]["run_meta"]
    for name, (attrs, value) in before.items():  # every dataset and every other attribute is untouched
        if name:
            assert attrs.keys() == after[name][0].keys()
            if value is not None:
                np.testing.assert_array_equal(value, after[name][1])
    record = json.loads((dst_run / f"{dst_run.name}{CONVERSION_SUFFIX}").read_text())
    assert record["slurm_job_id"] is None
    assert record["sha256"] == _sha256(h5) != sha_before
    assert record["anonymised"]["sha256_before"] == sha_before
    assert record["anonymised"]["replaced_with"] == ["poreml_author"]  # the old string never enters the record
    assert record["anonymised"]["at"]
    assert record["destination"] == str(h5).replace("alice", "poreml_author")  # the record's own paths too


def test_anonymise_dry_run_writes_nothing_and_a_second_pass_changes_nothing(roots):
    from poreml.convert import anonymise_tree

    dst, dst_run, h5 = _converted_copy_with_identity(roots)
    texts, h5_bytes = _json_texts(dst_run), h5.read_bytes()

    dry = anonymise_tree(dst, {"alice": "poreml_author"}, dry_run=True)
    assert dry["json_replacements"] > 0 and dry["h5_attrs_changed"] == 1
    assert _json_texts(dst_run) == texts and h5.read_bytes() == h5_bytes

    anonymise_tree(dst, {"alice": "poreml_author"})
    texts, h5_bytes = _json_texts(dst_run), h5.read_bytes()
    again = anonymise_tree(dst, {"alice": "poreml_author"})
    assert again["json_replacements"] == 0 and again["h5_attrs_changed"] == 0 and again["rehashed"] == 0
    assert _json_texts(dst_run) == texts and h5.read_bytes() == h5_bytes


def test_anonymise_json_only_leaves_the_h5_and_its_recorded_hash_alone(roots):
    from poreml.convert import _sha256, anonymise_tree

    dst, dst_run, h5 = _converted_copy_with_identity(roots)
    sha_before = _sha256(h5)

    summary = anonymise_tree(dst, {"alice": "poreml_author"}, h5=False)

    assert summary["json_replacements"] > 0 and summary["h5_attrs_changed"] == 0
    assert "alice" not in (dst_run / "run_meta.json").read_text()
    assert _sha256(h5) == sha_before
    record = json.loads((dst_run / f"{dst_run.name}{CONVERSION_SUFFIX}").read_text())
    assert record["sha256"] == sha_before and record["slurm_job_id"] is None and "anonymised" not in record
    with h5py.File(h5) as f:
        assert "alice" in f.attrs["run_meta"]


# ---------------------------------------------------------------------------
# Repacking the mirror without its derivable fields (rho = p / cs2, umag = |u|)


def _converted_copy_with_derivable_fields(roots, *, break_rho: bool = False) -> tuple[Path, Path, Path]:
    """A converted run whose steps also carry `rho` and `umag`, as the solver writes them, plus an XDMF
    that indexes every field and a conversion record — the shape of the real mirror."""
    dst, dst_run, h5 = _converted_copy_with_identity(roots)
    with h5py.File(h5, "r+") as f:
        f.attrs["cs2"] = 1.0 / 3.0
        f.attrs["fields"] = "phi,rho,p,umag,u"
        for step in f["steps"].values():
            p = step["p"][()]
            rho = (p * 3.0).astype(np.float32) if not break_rho else (p * 2.0).astype(np.float32)
            step.create_dataset("rho", data=rho, chunks=(1, 5, 6), compression="gzip")
            u = step["u"][()]
            step.create_dataset("umag", data=np.sqrt((u**2).sum(-1)).astype(np.float32), chunks=(1, 5, 6), compression="gzip")
    xdmf = dst_run / f"{dst_run.name}.xdmf"
    blocks = []
    for name in ("phi", "rho", "p", "umag", "u"):
        kind = "Vector" if name == "u" else "Scalar"
        item = f'<DataItem Format="HDF">{h5.name}:/steps/000000000/{name}</DataItem>'
        blocks.append(f'<Attribute Name="{name}" AttributeType="{kind}" Center="Node">\n{item}\n</Attribute>')
    xdmf.write_text("<Xdmf>\n<Grid>\n" + "\n".join(blocks) + "\n</Grid>\n</Xdmf>\n")
    from poreml.convert import _sha256

    record_path = dst_run / f"{dst_run.name}{CONVERSION_SUFFIX}"
    record = json.loads(record_path.read_text())
    record["sha256"], record["destination_size"] = _sha256(h5), h5.stat().st_size
    record_path.write_text(json.dumps(record, indent=2))
    return dst, dst_run, h5


def test_repack_drops_the_derivable_fields_and_keeps_every_other_byte(roots):
    from poreml.convert import _sha256, repack_run

    dst, dst_run, h5 = _converted_copy_with_derivable_fields(roots)
    before = read_all(h5)
    size_before, sha_before = h5.stat().st_size, _sha256(h5)
    with h5py.File(h5) as f:
        storage_before = {
            n: (f[n].id.get_storage_size(), f[n].chunks) for n in ("rock", "steps/000000000/phi", "steps/000000100/u")
        }

    record = repack_run(dst_run, replacements={"alice": "poreml_author"})

    after = read_all(h5)
    assert not [n for n in after if n.endswith(("/rho", "/umag"))]
    kept = [n for n in before if n and not n.endswith(("/rho", "/umag"))]
    assert sorted(after) == sorted([""] + kept)
    for name in kept:
        attrs_a, data_a = before[name]
        attrs_b, data_b = after[name]
        assert attrs_a.keys() == attrs_b.keys()
        if data_a is not None:
            np.testing.assert_array_equal(data_a, data_b)
    with h5py.File(h5) as f:
        for n, (size, chunks) in storage_before.items():  # raw chunks copied as they were: same bytes, same layout
            assert f[n].id.get_storage_size() == size and f[n].chunks == chunks, n
        assert f["steps/000000000/phi"]._filters == {"32015": (3,)}
        assert f.attrs["fields"] == "phi,p,u"
        assert "alice" not in f.attrs["run_meta"] and "/home/poreml_author/work/solver" in f.attrs["run_meta"]
        assert f.attrs["cs2"] == pytest.approx(1 / 3)
    assert h5.stat().st_size < size_before
    xdmf = (dst_run / f"{dst_run.name}.xdmf").read_text()
    assert "rho" not in xdmf and "umag" not in xdmf and xdmf.count("<Attribute") == 3
    saved = json.loads((dst_run / f"{dst_run.name}{CONVERSION_SUFFIX}").read_text())
    assert saved["sha256"] == _sha256(h5) != sha_before and saved["destination_size"] == h5.stat().st_size
    assert saved["repacked"]["dropped"] == ["rho", "umag"] and saved["repacked"]["sha256_before"] == sha_before
    assert saved["repacked"]["size_before"] == size_before and saved["repacked"]["at"]
    assert saved["anonymised"]["replaced_with"] == ["poreml_author"]
    assert record["skipped"] is False and record["dropped"] == 4  # two fields x two steps
    assert not list(dst_run.glob("*.tmp"))


def test_repack_is_idempotent_and_refuses_when_the_field_is_not_derivable(roots):
    from poreml.convert import repack_run

    dst, dst_run, h5 = _converted_copy_with_derivable_fields(roots)
    repack_run(dst_run)
    stamp = h5.stat().st_mtime_ns
    again = repack_run(dst_run)
    assert again["skipped"] is True and h5.stat().st_mtime_ns == stamp

    dst2, dst_run2, h5_2 = _converted_copy_with_derivable_fields(
        (dst.parent.parent / "other", dst.parent.parent / "other_data" / "case"), break_rho=True
    )
    before = h5_2.read_bytes()
    with pytest.raises(RuntimeError, match="rho"):
        repack_run(dst_run2)
    assert h5_2.read_bytes() == before and not list(dst_run2.glob("*.tmp"))


def test_repack_dry_run_reports_without_writing(roots):
    from poreml.convert import repack_run

    dst, dst_run, h5 = _converted_copy_with_derivable_fields(roots)
    before = h5.read_bytes()
    record = repack_run(dst_run, dry_run=True)
    assert record["dry_run"] is True and record["dropped"] == 4
    assert h5.read_bytes() == before and "repacked" not in json.loads(
        (dst_run / f"{dst_run.name}{CONVERSION_SUFFIX}").read_text()
    )


def test_repack_finishes_the_bookkeeping_of_a_file_repacked_but_left_unrecorded(roots):
    """A task killed between the rename and the record update leaves a repacked file with a
    stale record and an XDMF still naming the fields: the rerun completes both."""
    from poreml.convert import _sha256, repack_run

    dst, dst_run, h5 = _converted_copy_with_derivable_fields(roots)
    record_path = dst_run / f"{dst_run.name}{CONVERSION_SUFFIX}"
    xdmf_path = dst_run / f"{dst_run.name}.xdmf"
    stale_record, stale_xdmf = record_path.read_text(), xdmf_path.read_text()
    repack_run(dst_run)
    record_path.write_text(stale_record)  # as if the task died right after the rename
    xdmf_path.write_text(stale_xdmf)

    out = repack_run(dst_run)

    assert out["skipped"] is False and out["dropped"] == 0
    saved = json.loads(record_path.read_text())
    assert saved["repacked"]["dropped"] == ["rho", "umag"] and saved["sha256"] == _sha256(h5)
    assert saved["destination_size"] == h5.stat().st_size
    assert "rho" not in xdmf_path.read_text()
    assert repack_run(dst_run)["skipped"] is True
