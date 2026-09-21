"""The evaluation-only transfer studies (Geo-Shift, Scale-Up): a cell config names a
checkpoint source and two shared files — a per-campaign data file and a per-study scheme —
and `study.derive` composes them onto the checkpoint's own saved config."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pydantic
import pytest
import yaml

from poreml.config import Config
from poreml.metrics import METRICS_VERSION
from poreml.metrics.compose import resolve
from poreml.study import (
    Cell,
    CellConfig,
    cell_paths,
    derive,
    pool_test_runs,
    resolve_checkpoint,
    rock_max_axis,
)

REPO = Path(__file__).resolve().parents[1]
CHANNELS = ("phi", "ux", "uy", "uz", "p")
STUDIES = {"scale": "_all", "shift": "_gen"}  # study -> the kind of checkpoint its main cells score
CAMPAIGN_FOLDERS = {"drainage": "drainage", "gdl": "GDL", "trapping": "trapping"}


def write_run(train_case: Path, stamp: str, status: str, cfg: Config, mtime: float) -> Path:
    """One training run directory as `poreml train` leaves it: config.yaml, run_meta.json, ckpts/."""
    run_dir = train_case / "ckpts" / f"unet_drainage_all_{stamp}"
    (run_dir / "ckpts").mkdir(parents=True)
    cfg.to_yaml(run_dir / "config.yaml")
    (run_dir / "run_meta.json").write_text(json.dumps({"status": status}))
    for name in ("best.pt", "last.pt"):
        (run_dir / "ckpts" / name).write_bytes(b"")
    os.utime(run_dir / "run_meta.json", (mtime, mtime))
    return run_dir


def saved_config(tmp_path: Path, **train) -> Config:
    return Config.model_validate(
        {
            "name": "unet_drainage_all",
            "data": {"root": str(tmp_path / "data"), "campaign": "drainage", "split": "splits/drainage/all.yaml"},
            "task": {"name": "next_frame", "history": 1, "params": {"fields": [{"name": "phi"}], "roi": "rock"}},
            "model": {"name": "persistence"},
            "train": {"epochs": 3, "batch_size": 2, "device": "cpu", **train},
            "rollout": {"horizon": 8, "start_fraction": 0.5},
            "metrics": None,
        }
    )


def write_study(tmp_path: Path, *, root=None, horizon=4, stride=None, keyframes=2, run=None, which=None) -> tuple[Path, Path]:
    """A cell file with its shared data and scheme files under tmp_path; returns (cell, train_case)."""
    train_case = tmp_path / "case/train/drainage/unet_all"
    (tmp_path / "configs/scale/drainage").mkdir(parents=True)
    (tmp_path / "configs/scale/scheme.yaml").write_text(
        yaml.safe_dump(
            {
                "horizon": horizon,
                "stride": stride,
                "keyframes": keyframes,
                "metrics": [{"error": "mae", "channel": "phi"}, {"descriptor": "euler", "error": "target", "channel": "phi"}],
            }
        )
    )
    (tmp_path / "configs/scale/drainage/data.yaml").write_text(
        yaml.safe_dump({"root": str(root or tmp_path / "data"), "campaign": "drainage", "split": str(tmp_path / "split.yaml")})
    )
    cell = tmp_path / "configs/scale/drainage/unet_all.yaml"
    cell.write_text(
        yaml.safe_dump(
            {
                "name": "scale_drainage_unet_all",
                "data": str(tmp_path / "configs/scale/drainage/data.yaml"),
                "scheme": str(tmp_path / "configs/scale/scheme.yaml"),
                "checkpoint": {"train_case": str(train_case), "run": run, **({"which": which} if which else {})},
                "out_dir": str(tmp_path / "case/scale/drainage/unet_all"),
            }
        )
    )
    return cell, train_case


def test_cell_loads_its_shared_data_and_scheme_files(tmp_path):
    cell_path, _ = write_study(tmp_path)
    cell = Cell.load(cell_path)
    assert cell.config.name == "scale_drainage_unet_all"
    assert cell.data.campaign == "drainage" and cell.data.split == tmp_path / "split.yaml"
    assert cell.scheme.horizon == 4 and cell.scheme.window_stride == 4 and cell.scheme.keyframes == 2
    assert [m.key for m in cell.scheme.metrics] == ["mae@phi", "euler/target@phi"]


def test_cell_config_is_strict(tmp_path):
    cell_path, _ = write_study(tmp_path)
    payload = yaml.safe_load(cell_path.read_text())
    payload["stride"] = 8
    cell_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        CellConfig.from_yaml(cell_path)


def test_resolve_checkpoint_takes_the_newest_finished_run_and_never_a_running_one(tmp_path):
    cell_path, train_case = write_study(tmp_path)
    cfg = saved_config(tmp_path)
    old = write_run(train_case, "20260901-000000", "finished", cfg, mtime=1.0)
    write_run(train_case, "20260903-000000", "running", cfg, mtime=3.0)  # newest, but its best.pt still moves
    new = write_run(train_case, "20260902-000000", "finished", cfg, mtime=2.0)
    assert resolve_checkpoint(Cell.load(cell_path)) == new / "ckpts" / "best.pt"
    assert old != new


def test_resolve_checkpoint_refuses_a_case_with_no_finished_run(tmp_path):
    cell_path, train_case = write_study(tmp_path)
    write_run(train_case, "20260903-000000", "running", saved_config(tmp_path), mtime=3.0)
    with pytest.raises(FileNotFoundError, match="no finished run"):
        resolve_checkpoint(Cell.load(cell_path))


def test_resolve_checkpoint_honours_a_pinned_run(tmp_path):
    cell_path, train_case = write_study(tmp_path)
    cfg = saved_config(tmp_path)
    pinned = write_run(train_case, "20260901-000000", "preempted", cfg, mtime=1.0)  # pinning overrides the status rule
    write_run(train_case, "20260902-000000", "finished", cfg, mtime=2.0)
    cell_path, _ = write_study(tmp_path / "pinned", run=str(pinned))
    assert resolve_checkpoint(Cell.load(cell_path)) == pinned / "ckpts" / "best.pt"
    cell_path, _ = write_study(tmp_path / "file", run=str(pinned / "ckpts" / "last.pt"))
    assert resolve_checkpoint(Cell.load(cell_path)) == pinned / "ckpts" / "last.pt"


def test_resolve_checkpoint_can_take_the_rollout_selected_checkpoint(tmp_path):
    cell_path, train_case = write_study(tmp_path, which="best_rollout")
    cfg = saved_config(tmp_path)
    run = write_run(train_case, "20260901-000000", "finished", cfg, mtime=1.0)
    assert resolve_checkpoint(Cell.load(cell_path)) == run / "ckpts" / "best_rollout.pt"
    with pytest.raises(pydantic.ValidationError):
        write_study(tmp_path / "bad", which="worst") and Cell.load(tmp_path / "bad" / "configs/scale/drainage/unet_all.yaml")


def test_derive_keeps_the_saved_task_and_model_and_swaps_only_the_protocol(tmp_path):
    cell_path, train_case = write_study(tmp_path)
    saved = saved_config(tmp_path, num_workers=3)
    write_run(train_case, "20260901-000000", "finished", saved, mtime=1.0)
    cell = Cell.load(cell_path)
    derived = derive(cell, resolve_checkpoint(cell))

    assert derived.task == saved.task and derived.model == saved.model  # what the weights were trained on, untouched
    assert derived.name == "scale_drainage_unet_all" and derived.out_dir == tmp_path / "case/scale/drainage/unet_all"
    assert derived.data == cell.data
    assert (
        derived.rollout.horizon == 4 and derived.rollout.keyframes == 2 and derived.train.batch_size == saved.train.batch_size
    )
    assert [m.key for m in derived.metrics] == ["mae@phi", "euler/target@phi"]
    assert derived.train.num_workers == 3 and derived.train.epochs == 3  # everything else stays the training-time value


def test_derive_refuses_a_checkpoint_without_its_saved_config(tmp_path):
    cell_path, train_case = write_study(tmp_path)
    run_dir = write_run(train_case, "20260901-000000", "finished", saved_config(tmp_path), mtime=1.0)
    (run_dir / "config.yaml").unlink()
    cell = Cell.load(cell_path)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        derive(cell, run_dir / "ckpts" / "best.pt")


def trained_cell(fake_root, tmp_path):
    """The smoke model trained on the fixture and a cell scoring it: (cell, derived config, checkpoint)."""
    from poreml.train import train
    from tests.fixtures import write_fake_dataset

    run_ids = write_fake_dataset(fake_root)
    train_split = tmp_path / "train_split.yaml"
    train_split.write_text(yaml.safe_dump({"train": run_ids[:2], "val": run_ids[2:], "test": []}))
    cfg = Config.model_validate(
        {
            "name": "unet_drainage_all",
            "data": {"root": str(fake_root), "campaign": "drainage", "split": str(train_split)},
            "task": {"name": "next_frame", "params": {"fields": [{"name": "phi"}], "roi": None}},
            "model": {"name": "persistence"},
            "train": {"epochs": 1, "batch_size": 1, "device": "cpu", "max_batches": 1},
            "metrics": None,
            "out_dir": str(tmp_path / "case/train/drainage/unet_all/ckpts"),
        }
    )
    train(cfg)
    cell_path, _ = write_study(tmp_path, root=fake_root, horizon=2, keyframes=2)
    (tmp_path / "split.yaml").write_text(yaml.safe_dump({"train": [], "val": [], "test": run_ids}))
    cell = Cell.load(cell_path)
    ckpt = resolve_checkpoint(cell)
    return cell, derive(cell, ckpt), ckpt


def test_end_to_end_on_the_fixture(fake_root, tmp_path):
    """Train the smoke model on the fixture, then score it through a cell: the derived config
    stores keyframes for the scheme's protocol and the metric stage aggregates them."""
    from poreml.evaluate import inference_split, metric_split
    from poreml.study import classify

    cell, derived, ckpt = trained_cell(fake_root, tmp_path)
    inference_split(derived, ckpt=ckpt, split="test", out_dir=cell.config.out_dir, stride=cell.scheme.window_stride)
    assert classify(cell) == "metric"
    payload = metric_split(derived, ckpt=ckpt, split="test", out_dir=cell.config.out_dir, workers=1)
    assert classify(cell) == "finished"
    assert payload["checkpoint"] == str(ckpt) and payload["protocol"] == {
        "horizon": 2,
        "stride": 2,
        "keyframes": 2,
        "steps": [1, 2],
    }
    assert set(payload["at_horizon"]) >= {"mae@phi", "euler/target@phi"}


def _ref(run_id, family, shape, rock, status="finished"):
    return SimpleNamespace(run_id=run_id, status=status, geometry={"family": family}, shape=shape, regions={"rock": rock})


def test_rock_max_axis_crops_the_padded_domain_to_the_rock():
    assert rock_max_axis(_ref("a", "blob", (128, 128, 161), [7, 135])) == 128
    assert rock_max_axis(_ref("a", "fiber", (256, 256, 141), [7, 135])) == 256
    assert rock_max_axis(SimpleNamespace(shape=(), regions={})) is None


def test_pool_test_runs_filters_by_size_family_and_status_and_sorts():
    refs = [
        _ref("b", "blob", (128, 128, 161), [7, 135]),
        _ref("a", "bentheimer", (128, 128, 161), [7, 135]),
        _ref("c", "blob", (256, 256, 289), [7, 263]),
        _ref("d", "blob", (128, 128, 161), [7, 135], status="running"),
    ]
    assert pool_test_runs(refs, size_class=128, families=None) == ["a", "b"]
    assert pool_test_runs(refs, size_class=128, families=("blob",)) == ["b"]
    assert pool_test_runs(refs, size_class=256, families=None) == ["c"]


def _metrics_payload(steps, values, groups=None, sha="s"):
    return {
        "steps": steps,
        "n_runs": 2,
        "n_windows": 5,
        "n_runs_per_step": {"mae@phi": [2] * len(steps)},
        "n_windows_per_step": {"mae@phi": [5] * len(steps)},
        "summary": {"mae@phi": values, "curvature_hist/pred@phi": [[0.5, 0.5]] * len(steps)},
        "groups": groups or {},
        "at_horizon": {"mae@phi": values[-1]},
        "checkpoint_sha256": sha,
        "metric": {"checkpoint_sha256": sha, "poreml_commit": "abc"},
    }


def test_study_curves_write_one_svg_and_csv_per_scalar_metric(tmp_path):
    pytest.importorskip("matplotlib")
    for cell, values in (("unet_gen", [0.1, 0.2, 0.3]), ("fno_gen", [0.05, 0.1, 0.15])):
        (tmp_path / cell).mkdir()
        (tmp_path / cell / "metrics_test.json").write_text(json.dumps(_metrics_payload([1, 4, 8], values)))
    from poreml.study import study_curves

    written = study_curves(tmp_path)
    assert sorted(p.name for p in written) == ["mae@phi.csv", "mae@phi.svg"]  # the histogram is not a curve
    text = (tmp_path / "curves" / "mae@phi.csv").read_text().splitlines()
    assert text[0] == "step,fno_gen_mean,fno_gen_n_runs,fno_gen_n_windows,unet_gen_mean,unet_gen_n_runs,unet_gen_n_windows"
    assert text[1] == "1,0.05,2,5,0.1,2,5" and len(text) == 4
    assert "unet_gen" in (tmp_path / "curves" / "mae@phi.svg").read_text()


def test_study_report_tables_first_and_last_step_per_group(tmp_path):
    groups = {
        "gen": {
            "summary": {"mae@phi": [0.1, 0.3]},
            "n_runs_per_step": {"mae@phi": [1, 1]},
            "n_windows_per_step": {"mae@phi": [3, 3]},
            "runs": ["a"],
        }
    }
    (tmp_path / "unet_all").mkdir()
    (tmp_path / "unet_all" / "metrics_test.json").write_text(json.dumps(_metrics_payload([1, 64], [0.2, 0.4], groups)))
    from poreml.study import study_report

    text = study_report(tmp_path)
    assert (tmp_path / "report.md").read_text() == text
    assert "| metric | unet_all |" in text and "| mae@phi | 0.2000 / 0.4000 |" in text
    assert "gen runs" in text and "| mae@phi | 0.1000 / 0.3000 |" in text
    assert "curvature_hist" not in text and "abc" in text


def test_classify_walks_waiting_pending_metric_finished(tmp_path):
    from poreml.study import classify

    cell_path, train_case = write_study(tmp_path)
    cell = Cell.load(cell_path)
    assert classify(cell) == "waiting"
    write_run(train_case, "a", "finished", saved_config(tmp_path), mtime=1.0)
    assert classify(cell) == "pending"
    out = cell.config.out_dir
    out.mkdir(parents=True)
    (out / "inference_test.json").write_text(json.dumps({"checkpoint_sha256": "s", "windows": {}}))
    assert classify(cell) == "metric"
    (out / "metrics_test.json").write_text(
        json.dumps({"metric": {"checkpoint_sha256": "other", "metrics_version": METRICS_VERSION}})
    )
    assert classify(cell) == "metric"  # scored for another checkpoint: rescore
    (out / "metrics_test.json").write_text(json.dumps({"metric": {"checkpoint_sha256": "s", "metrics_version": "0"}}))
    assert classify(cell) == "metric"  # scored under an older metrics contract: rescore
    (out / "metrics_test.json").write_text(json.dumps({"metric": {"checkpoint_sha256": "s"}}))
    assert classify(cell) == "metric"  # no version recorded: unknown contract, rescore
    (out / "metrics_test.json").write_text(
        json.dumps({"metric": {"checkpoint_sha256": "s", "metrics_version": METRICS_VERSION}})
    )
    assert classify(cell) == "finished"


@pytest.mark.parametrize("study", sorted(STUDIES))
def test_every_shipped_cell_config_mirrors_its_training_case(study):
    """configs/<study>/<campaign>/<model>_<kind>.yaml names the training case it scores, the
    campaign's shared data file, the study's shared scheme, and an out_dir under case/<study>/."""
    scheme = REPO / "configs" / study / "scheme.yaml"
    cells = cell_paths(study, repo=REPO)
    assert len(cells) >= 15, f"{study}: expected 5 models x 3 campaigns"
    for path in cells:
        campaign = path.parent.name
        cell = Cell.load(path)
        stem = path.stem
        assert cell.config.name == f"{study}_{campaign}_{stem}", path
        assert cell.config.data == Path("configs") / study / campaign / "data.yaml", path
        assert cell.config.scheme == scheme.relative_to(REPO), path
        assert cell.config.checkpoint.train_case == Path("case/train") / campaign / stem, path
        assert cell.config.checkpoint.run is None, f"{path}: shipped cells never pin a run"
        assert cell.config.out_dir == Path("case") / study / campaign / stem, path
        assert cell.data.campaign == CAMPAIGN_FOLDERS[campaign] and cell.data.root == Path("data/case"), path
        assert cell.data.split == Path("splits") / study / f"{campaign}.yaml", path
        kind = STUDIES[study]
        assert stem.endswith(kind), f"{path}: a {study} main cell scores a {kind} checkpoint"
        # The scheme's protocol: stride 64, the 64-frame rollout scored every step, criterion first.
        assert cell.scheme.horizon == 64 and cell.scheme.window_stride == 64 and cell.scheme.keyframes == 12
        assert resolve(cell.scheme.metrics, CHANNELS)[0].name == "mae@phi"
        # The frozen test split exists, is eval-only, and holds only runs of the campaign.
        split = yaml.safe_load((REPO / cell.data.split).read_text())
        assert split["train"] == [] and split["val"] == [] and len(split["test"]) >= 16, path


# ---- render: 3 truth | prediction GIFs per cell, one run per rock family, like training's periodic render


def test_scheme_render_defaults_to_three_runs_per_cell():
    from poreml.study import SchemeConfig

    assert SchemeConfig(metrics=[{"error": "mae", "channel": "phi"}]).render == 3
    assert SchemeConfig(metrics=[{"error": "mae", "channel": "phi"}], render=0).render == 0
    with pytest.raises(ValueError):
        SchemeConfig(metrics=[{"error": "mae", "channel": "phi"}], render=-1)


def test_render_runs_takes_one_run_per_family_round_robin():
    from poreml.study import render_runs

    refs = [
        _ref("bent_0003", "bentheimer", (128, 128, 161), (16, 144)),
        _ref("blob_0002", "blob", (128, 128, 161), (16, 144)),
        _ref("castle_0005", "castlegate", (128, 128, 161), (16, 144)),
        _ref("bent_0001", "bentheimer", (128, 128, 161), (16, 144)),
        _ref("blob_0009", "blob", (128, 128, 161), (16, 144)),
    ]
    assert render_runs(refs, 3) == ["bent_0001", "blob_0002", "castle_0005"]
    assert render_runs(refs, 5) == ["bent_0001", "blob_0002", "castle_0005", "bent_0003", "blob_0009"]
    assert render_runs(refs, 0) == []


def test_render_runs_returns_every_run_when_fewer_than_wanted():
    from poreml.study import render_runs

    refs = [_ref("a_0001", None, (8, 8, 8), (0, 8)), _ref("a_0002", None, (8, 8, 8), (0, 8))]
    assert render_runs(refs, 3) == ["a_0001", "a_0002"]


def _render_cfg(fake_root: Path, tmp_path: Path, run_ids: list[str]) -> Config:
    (tmp_path / "split.yaml").write_text(yaml.safe_dump({"train": [], "val": [], "test": run_ids}))
    return Config.model_validate(
        {
            "name": "scale_drainage_unet_all",
            "data": {"root": str(fake_root), "campaign": "drainage", "split": str(tmp_path / "split.yaml")},
            "task": {"name": "next_frame", "params": {"fields": [{"name": "phi"}], "roi": None}},
            "model": {"name": "persistence"},
            "train": {"epochs": 1, "batch_size": 1, "device": "cpu"},
            "rollout": {"horizon": 2, "start_fraction": 0.25},
            "metrics": [{"error": "mae", "channel": "phi"}],
        }
    )


def test_render_cell_skips_existing_gifs_and_survives_a_failed_render(fake_root, tmp_path, monkeypatch, capsys):
    from poreml import evaluate
    from poreml.study import render_cell
    from tests.fixtures import write_fake_dataset

    run_ids = write_fake_dataset(fake_root)
    cfg = _render_cfg(fake_root, tmp_path, run_ids)
    out_dir = tmp_path / "cell"
    picked = sorted(run_ids)[:3]  # the fixture is one family, so the pick is the first three ids
    done = out_dir / "render" / picked[0] / f"rollout_{picked[0]}.gif"
    done.parent.mkdir(parents=True)
    done.write_bytes(b"GIF")
    calls: list[str] = []

    def fake_render(cfg, ckpt, run_id, out_dir, split="test", **kw):
        calls.append(run_id)
        if run_id == picked[1]:
            raise RuntimeError("render worker failed:\nbad X server connection")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"rollout_{run_id}.gif").write_bytes(b"GIF")
        return Path(out_dir)

    monkeypatch.setattr(evaluate, "render_rollout", fake_render)
    gifs = render_cell(cfg, ckpt=None, out_dir=out_dir, n=3)

    assert calls == [picked[1], picked[2]]  # the existing GIF is skipped, the failure does not stop the loop
    assert gifs == [done, out_dir / "render" / picked[2] / f"rollout_{picked[2]}.gif"]
    out = capsys.readouterr().out
    assert "!!" in out and "bad X server" in out  # the failure is reported, not raised
    render_cell(cfg, ckpt=None, out_dir=out_dir, n=3, force=True)
    assert calls[2:] == picked  # --force redoes all three


def test_render_cell_renders_a_gif_on_the_fixture(fake_root, tmp_path):
    pytest.importorskip("pyvista")
    from poreml.study import render_cell
    from tests.fixtures import write_fake_dataset

    run_ids = write_fake_dataset(fake_root)
    cfg = _render_cfg(fake_root, tmp_path, run_ids)
    gifs = render_cell(cfg, ckpt=None, out_dir=tmp_path / "cell", n=1)
    assert len(gifs) == 1 and gifs[0].exists() and gifs[0].stat().st_size > 0
    frames = sorted(gifs[0].parent.glob("frames/frame_*.png"))
    assert len(frames) == 2  # horizon 2: one truth | prediction image per step


def test_render_gifs_lists_a_cells_existing_gifs_sorted(tmp_path):
    from poreml.study import render_gifs

    assert render_gifs(tmp_path / "missing") == []
    for run in ("b_0002", "a_0001"):
        (tmp_path / "render" / run).mkdir(parents=True)
        (tmp_path / "render" / run / f"rollout_{run}.gif").write_bytes(b"GIF")
    (tmp_path / "render" / "c_0003").mkdir()  # a failed render leaves the directory without a GIF
    assert render_gifs(tmp_path) == [
        tmp_path / "render" / "a_0001" / "rollout_a_0001.gif",
        tmp_path / "render" / "b_0002" / "rollout_b_0002.gif",
    ]


def test_render_cell_scores_only_the_cheap_metrics(fake_root, tmp_path, monkeypatch):
    """Rendering scores every step in-process for the frame labels; the mesh/curvature family
    (`eval_only`, 30-60 s per side at 256) must not be in that loop."""
    from poreml import evaluate
    from poreml.study import render_cell
    from tests.fixtures import write_fake_dataset

    run_ids = write_fake_dataset(fake_root)
    cfg = _render_cfg(fake_root, tmp_path, run_ids).updated(
        metrics=[
            {"error": "mae", "channel": "phi"},
            {"descriptor": "saturation", "error": "abs_err", "channel": "phi"},
            {"descriptor": "mesh_area", "error": "pred", "channel": "phi", "eval_only": True},
        ]
    )
    seen = []

    def fake_render(cfg, ckpt, run_id, out_dir, split="test", **kw):
        seen.append([(m.descriptor, m.error) for m in cfg.metrics])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"rollout_{run_id}.gif").write_bytes(b"GIF")

    monkeypatch.setattr(evaluate, "render_rollout", fake_render)
    render_cell(cfg, ckpt=None, out_dir=tmp_path / "cell", n=1)
    assert seen == [[("voxels", "mae"), ("saturation", "abs_err")]]


def test_frame_store_claims_are_exclusive_until_finished_or_stale(tmp_path):
    """A window is one worker's from `claim` to `finish`; a claim nobody touches for `stale_after`
    seconds is a dead worker's and the next claimant takes it over."""
    import numpy as np

    from poreml.frames import H5FrameStore, MemoryFrameStore

    store = H5FrameStore(tmp_path / "frames", fields=("phi",), stale_after=60.0)
    assert store.claim("run", 0) is True
    assert store.claim("run", 0) is False, "a live claim is exclusive"
    marker = store.path("run", 0).with_suffix(".h5.claim")
    assert marker.is_file() and "pid=" in marker.read_text()
    assert store.claim("run", 8) is True, "another window is free"

    old = marker.stat().st_mtime - 120
    os.utime(marker, (old, old))
    assert store.claim("run", 0) is True, "a stale claim is taken over"
    assert marker.stat().st_mtime > old + 60, "the takeover refreshes the marker"

    store.open("run", 0, horizon=1, steps=[1], checkpoint_sha256=None)
    os.utime(marker, (old, old))
    store.write("run", 0, 1, np.zeros((1, 2, 2, 2), np.float32))
    assert marker.stat().st_mtime > old + 60, "every keyframe write touches the claim"
    store.finish("run", 0)
    assert not marker.exists() and store.has("run", 0)
    assert store.claim("run", 0) is True, "a finished window's claim is released"

    memory = MemoryFrameStore(fields=("phi",))
    assert memory.claim("run", 0) is True and memory.claim("run", 0) is False
    memory.open("run", 0)
    memory.finish("run", 0)
    assert memory.claim("run", 0) is True


def test_shared_inference_skips_held_windows_and_the_last_worker_writes_the_sentinel(fake_root, tmp_path):
    """Shared workers claim windows as they go: a window another worker holds is left to it and the
    sentinel appears only once every window is stored, so the metric stage never sees a partial cell."""
    from poreml.evaluate import inference_split
    from poreml.frames import H5FrameStore
    from poreml.study import classify

    cell, derived, ckpt = trained_cell(fake_root, tmp_path)
    out = Path(cell.config.out_dir)
    sentinel = out / "inference_test.json"
    kwargs = {"ckpt": ckpt, "split": "test", "out_dir": out, "stride": cell.scheme.window_stride}

    payload = inference_split(derived, shared=True, **kwargs)
    assert payload["shared"] and payload["complete"] and payload["n_missing"] == 0 and sentinel.is_file()
    assert not list(out.rglob("*.claim")), "a finished worker leaves no claim behind"
    run_id, t0 = next((r, starts[0]) for r, starts in payload["windows"].items() if starts)
    store = H5FrameStore(out / payload["frames_dir"])
    window = store.path(run_id, t0)

    # Another worker holds one window: this worker skips it and must not declare the cell complete.
    sentinel.unlink()
    window.unlink()
    assert store.claim(run_id, t0)
    payload = inference_split(derived, shared=True, **kwargs)
    assert not payload["complete"] and payload["n_missing"] == 1
    assert not sentinel.exists() and not window.exists() and classify(cell) == "pending"

    # That worker died: its claim goes stale, the next worker takes the window and completes the cell.
    marker = window.with_suffix(".h5.claim")
    old = marker.stat().st_mtime - 2 * 3600
    os.utime(marker, (old, old))
    payload = inference_split(derived, shared=True, **kwargs)
    assert payload["complete"] and window.is_file() and sentinel.is_file() and not marker.exists()
    assert classify(cell) == "metric"

    # An unshared run never claims: it recomputes whatever is missing, claims or not, and always writes the sentinel.
    sentinel.unlink()
    window.unlink()
    assert store.claim(run_id, t0)
    payload = inference_split(derived, **kwargs)
    assert payload["complete"] and not payload["shared"] and sentinel.is_file() and window.is_file()
