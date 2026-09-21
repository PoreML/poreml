import json

import pytest
import yaml

from poreml.data import discover, load_split, resolve


def test_discover_indexes_runs_by_directory_name(fake_root):
    runs = discover(fake_root, "drainage")

    assert set(runs) == {"tiny_0000_M1_theta120", "tiny_0001_M1_theta130", "tiny_0002_M10_theta140"}


def test_discover_populates_run_ref_paths(fake_root):
    ref = discover(fake_root, "drainage")["tiny_0000_M1_theta120"]

    assert ref.run_id == "tiny_0000_M1_theta120"
    assert ref.campaign == "drainage"
    assert ref.h5_path.exists() and ref.h5_path.suffix == ".h5"
    assert ref.meta_path.name == "run_meta.json"
    assert ref.geometry_path is not None and ref.geometry_path.exists()


def test_run_ref_exposes_physical_parameters(fake_root):
    ref = discover(fake_root, "drainage")["tiny_0002_M10_theta140"]

    assert ref.params["M"] == pytest.approx(10.0)
    assert ref.params["theta"] == pytest.approx(140.0)
    assert ref.shape == (8, 8, 10)
    assert ref.regions["rock"] == [1, 10]


def test_discover_on_missing_campaign_raises(fake_root):
    with pytest.raises(FileNotFoundError, match="GDL"):
        discover(fake_root, "GDL")


def test_discover_ignores_directories_without_run_meta(fake_root):
    stray = fake_root / "drainage" / "runs" / "not_a_run"
    stray.mkdir()
    (stray / "notes.txt").write_text("hello")

    assert "not_a_run" not in discover(fake_root, "drainage")


def test_load_split_reads_the_three_sections(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"train": ["a"], "val": ["b"], "test": ["c"]}))

    assert load_split(path) == {"train": ["a"], "val": ["b"], "test": ["c"]}


def test_load_split_rejects_an_unknown_section(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"train": ["a"], "holdout": ["b"]}))

    with pytest.raises(ValueError, match="holdout"):
        load_split(path)


def test_load_split_rejects_a_run_in_two_sections(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"train": ["a"], "test": ["a"]}))

    with pytest.raises(ValueError, match="run 'a' appears in both 'train' and 'test'"):
        load_split(path)


def test_resolve_joins_split_ids_to_discovered_runs(fake_root):
    available = discover(fake_root, "drainage")
    split = {"train": ["tiny_0000_M1_theta120"], "val": ["tiny_0001_M1_theta130"], "test": ["tiny_0002_M10_theta140"]}

    resolved = resolve(split, available)

    assert [r.run_id for r in resolved["train"]] == ["tiny_0000_M1_theta120"]
    assert [r.run_id for r in resolved["test"]] == ["tiny_0002_M10_theta140"]


def test_resolve_raises_on_a_missing_run_id(fake_root):
    available = discover(fake_root, "drainage")
    split = {"train": ["tiny_0000_M1_theta120", "ghost_run"], "val": [], "test": []}

    with pytest.raises(KeyError, match="ghost_run"):
        resolve(split, available)


def test_fixture_writes_nan_inside_solid(fake_root):
    import h5py
    import numpy as np

    path = next((fake_root / "drainage" / "runs" / "tiny_0000_M1_theta120").glob("*.h5"))
    with h5py.File(path, "r") as f:
        rock = f["rock"][:]
        phi = f["steps"]["000000000"]["phi"][:]

    assert np.isnan(phi[rock == 1]).all()
    assert not np.isnan(phi[rock == 0]).any()


def test_run_ref_exposes_status_and_finish_type(fake_root):
    ref = discover(fake_root, "drainage")["tiny_0000_M1_theta120"]

    assert ref.status == "finished"
    assert ref.finish_type == "pv_cap"


def test_run_ref_geometry_is_parsed_from_bobs_source_path(fake_root):
    ref = discover(fake_root, "drainage")["tiny_0000_M1_theta120"]

    assert ref.geometry["family"] == "tiny"
    assert ref.geometry["id"] == "tiny_0000"
    assert ref.geometry["sha256"]
    assert 0.0 < ref.geometry["porosity"] < 1.0


def test_run_ref_geometry_is_none_where_the_solver_recorded_nothing(tmp_path):
    from tests.fixtures import write_fake_run

    run_dir = write_fake_run(tmp_path, "drainage", "bare_0000_M1_theta120")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    meta["geometry"] = {"shape": [8, 8, 10]}
    (run_dir / "run_meta.json").write_text(json.dumps(meta))

    ref = discover(tmp_path, "drainage")["bare_0000_M1_theta120"]

    assert ref.geometry == {"id": None, "family": None, "sha256": None, "porosity": None}
    assert ref.status == "finished"


def test_discover_reads_a_two_stage_trapping_run_from_its_flood_meta(tmp_path):
    # the solver's trapping runs are two-stage: the dir holds run_meta_flood.json and
    # run_meta_stab.json (no plain run_meta.json) and the single h5 is the flood.
    from tests.fixtures import write_fake_run

    run_dir = write_fake_run(tmp_path, "trapping", "trap_0000_M1_theta120")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    (run_dir / "run_meta.json").unlink()
    (run_dir / "run_meta_flood.json").write_text(json.dumps(meta))
    stab = dict(meta, run={"status": "finished"}, extra=dict(meta["extra"], finish_type="equilibrated"))
    (run_dir / "run_meta_stab.json").write_text(json.dumps(stab))

    ref = discover(tmp_path, "trapping")["trap_0000_M1_theta120"]

    assert ref.meta_path.name == "run_meta_flood.json"
    assert ref.status == "finished"
    assert ref.finish_type == "pv_cap"  # the flood stage's, not the stab stage's


def test_run_ref_params_reads_underfills_m_ratio_as_m(tmp_path):
    # the solver's underfill driver predates the extra.M convention and records the
    # viscosity ratio as extra.m_ratio; params must map it to "M", not NaN.
    from tests.fixtures import write_fake_run

    run_dir = write_fake_run(tmp_path, "underfill", "fill_0000_M5_th30")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    del meta["extra"]["M"]
    meta["extra"]["m_ratio"] = 5.0
    (run_dir / "run_meta.json").write_text(json.dumps(meta))

    ref = discover(tmp_path, "underfill")["fill_0000_M5_th30"]

    assert ref.params["M"] == pytest.approx(5.0)


def test_run_ref_summary_is_the_flat_record_artifacts_embed(fake_root):
    ref = discover(fake_root, "drainage")["tiny_0002_M10_theta140"]

    record = ref.summary()

    assert record["run_id"] == "tiny_0002_M10_theta140"
    assert record["status"] == "finished"
    assert record["params"]["M"] == pytest.approx(10.0)
    assert record["geometry"]["family"] == "tiny"


def test_resolve_rejects_an_unfinished_run_by_default(tmp_path):
    from tests.fixtures import write_fake_run

    write_fake_run(tmp_path, "drainage", "done_0000_M1_theta120")
    write_fake_run(tmp_path, "drainage", "live_0001_M1_theta120", status="running")
    available = discover(tmp_path, "drainage")
    split = {"train": ["done_0000_M1_theta120"], "val": [], "test": ["live_0001_M1_theta120"]}

    with pytest.raises(ValueError, match="live_0001_M1_theta120.*running"):
        resolve(split, available)


def test_resolve_can_be_told_to_accept_unfinished_runs(tmp_path):
    from tests.fixtures import write_fake_run

    write_fake_run(tmp_path, "drainage", "live_0001_M1_theta120", status="running")
    available = discover(tmp_path, "drainage")
    split = {"train": ["live_0001_M1_theta120"], "val": [], "test": []}

    resolved = resolve(split, available, require_finished=False)

    assert resolved["train"][0].status == "running"


def _fake_refs(n_per_family: dict[str, int], status: str = "finished"):
    """RunRefs with just enough metadata for `draw_split`: family, status, run_id."""
    from pathlib import Path

    from poreml.data import RunRef

    refs = []
    for family, n in n_per_family.items():
        for i in range(n):
            run_id = f"drain_{family}128_{i:04d}_M1_th120"
            meta = {
                "run": {"status": status},
                "geometry": {"source": f"/data/{family}/128/{family}128_{i:04d}.npy"},
                "extra": {"M": 1.0, "ca": 1e-5},
                "solver": {"theta": 120.0},
            }
            refs.append(RunRef(run_id, "drainage", Path(f"{run_id}.h5"), Path("run_meta.json"), None, meta))
    return refs


def test_draw_split_is_family_stratified_disjoint_and_complete():
    from poreml.data import draw_split

    refs = _fake_refs({"bentheimer": 20, "blob": 10, "poly": 10})

    split = draw_split(refs, n_train=16, n_val=4, n_test=4, seed=0)

    ids = split["train"] + split["val"] + split["test"]
    assert len(ids) == len(set(ids)) == 24
    assert len(split["train"]) == 16 and len(split["val"]) == 4 and len(split["test"]) == 4
    # Every family is present in every section, in proportion (bentheimer is half the pool).
    for section, n in (("train", 16), ("val", 4), ("test", 4)):
        families = [next(r for r in refs if r.run_id == i).geometry["family"] for i in split[section]]
        assert set(families) == {"bentheimer", "blob", "poly"}
        assert families.count("bentheimer") == n // 2


def test_draw_split_is_deterministic_under_a_seed_and_moves_with_it():
    from poreml.data import draw_split

    refs = _fake_refs({"bentheimer": 8, "blob": 8})

    assert draw_split(refs, 4, 2, 2, seed=1) == draw_split(refs, 4, 2, 2, seed=1)
    assert draw_split(refs, 4, 2, 2, seed=1) != draw_split(refs, 4, 2, 2, seed=2)


def test_draw_split_only_uses_finished_runs_and_refuses_to_overdraw():
    from poreml.data import draw_split

    refs = _fake_refs({"blob": 4}) + _fake_refs({"poly": 4}, status="running")

    split = draw_split(refs, 2, 1, 1, seed=0)
    assert all("blob" in i for i in split["train"] + split["val"] + split["test"])

    with pytest.raises(ValueError, match="only 4 finished"):
        draw_split(refs, 4, 1, 1, seed=0)


def test_draw_split_refuses_a_run_pool_with_unknown_family():
    from poreml.data import draw_split

    refs = _fake_refs({"blob": 4})
    refs[0].meta["geometry"] = {}

    with pytest.raises(ValueError, match="family"):
        draw_split(refs, 2, 1, 1, seed=0)
