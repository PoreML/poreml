"""`rollout_inference` on the fixture: every window of every validation run timed and scored, the
per-step CSV, the stored keyframes, the sentinel JSON, resume after SIGTERM — and the driver's
checkpoint / run discovery rules."""

import csv
import importlib.util
import json
import os
import signal
from pathlib import Path

import pytest
import yaml

from poreml.config import Config
from poreml.frames import H5FrameStore
from poreml.inference import CSV_NAME, DIR_NAME, JSON_NAME, metric_inference, read_rows, rollout_inference
from poreml.metrics import MetricSpec, json_safe, resolve
from poreml.rollout import Interrupted, keyframe_steps, rollout, window_starts

VAL = ["tiny_0002_M10_theta140", "tiny_0000_M1_theta120"]


def inference_cfg(fake_root, tmp_path, horizon=2, keyframes=2, fields=({"name": "phi"}, {"name": "p"})):
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"train": [], "val": VAL, "test": []}))
    return Config.model_validate(
        {
            "name": "inference",
            "data": {"root": str(fake_root), "campaign": "drainage", "split": str(split)},
            "task": {"name": "next_frame", "history": 1, "params": {"fields": list(fields)}},
            "model": {"name": "persistence"},
            "train": {"epochs": 1, "batch_size": 1, "device": "cpu", "max_batches": 1, "num_workers": 0},
            "metrics": [{"error": "mae", "channel": "phi"}],
            "rollout": {"horizon": horizon, "keyframes": keyframes},
            "out_dir": str(tmp_path / "runs"),
        }
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_rollout_inference_writes_the_csv_the_keyframes_and_the_sentinel(fake_root, tmp_path):
    cfg = inference_cfg(fake_root, tmp_path)
    out = tmp_path / "run"
    payload = rollout_inference(cfg, split="val", out_dir=out)

    # the fixture has 6 frames per run: windows of 2 from frame 0 every 2 frames -> t0 = 0, 2 (4 has no room)
    assert window_starts(6, 1, 2, 2) == [0, 2]
    rows = read_csv(out / CSV_NAME)
    assert len(rows) == 2 * 2 * 2
    columns = [
        "run",
        "family",
        "group",
        "t0",
        "h",
        "t",
        "forward_s",
        "step_s",
        "stored",
        "mae@phi",
        "rel_mae@phi",
        "mae@p",
        "rel_mae@p",
    ]
    assert list(rows[0])[: len(columns)] == columns
    assert [(int(r["t0"]), int(r["h"])) for r in rows if r["run"] == VAL[0]] == [(0, 1), (0, 2), (2, 1), (2, 2)]
    assert all(int(r["t"]) == int(r["t0"]) + int(r["h"]) for r in rows)
    assert all(r["family"] == "tiny" and r["group"] == "" for r in rows)  # the fixture's family is in no study group
    assert all(float(r["forward_s"]) > 0 and float(r["step_s"]) >= float(r["forward_s"]) for r in rows)
    assert all(r["stored"] == "True" for r in rows)  # keyframes 2 on a horizon of 2: both steps

    store = H5FrameStore(out / DIR_NAME / "frames")
    assert store.runs() == sorted(VAL)
    for run in VAL:
        assert store.windows(run) == [0, 2] and store.steps(run, 2) == [1, 2]
        meta = store.meta(run, 2)
        assert meta["horizon"] == 2 and meta["steps"] == [1, 2] and meta["fields"] == ["phi", "p"]
        assert store.read(run, 2, 2).shape[0] == 2  # every field is stored

    written = json.loads((out / DIR_NAME / JSON_NAME).read_text())
    assert written == json.loads(json.dumps(json_safe(payload)))
    assert written["split"] == "val" and written["n_runs"] == 2 and written["checkpoint"] is None
    assert written["protocol"] == {"horizon": 2, "stride": 2, "keyframes": 2, "steps": [1, 2]}
    assert written["scored_steps"] == [1, 2]
    assert written["windows"] == {VAL[0]: [0, 2], VAL[1]: [0, 2]} and written["runs_without_windows"] == []
    assert written["n_windows"] == 4 and written["n_frames"] == 8
    assert written["fields"] == ["phi", "p"] and written["fields_missing"] == []
    assert written["csv"] == CSV_NAME and written["frames_dir"] == "frames"
    assert written["n_runs_per_step"]["mae@phi"] == [2, 2] and written["n_windows_per_step"]["mae@phi"] == [4, 4]
    assert written["groups"] == {}
    assert len(written["summary"]["mae@phi"]) == 2 and written["at_horizon"]["mae@phi"] == written["summary"]["mae@phi"][-1]
    assert written["timing"]["forward_s"]["n"] == 8 and written["timing"]["forward_s"]["mean"] > 0
    assert written["timing"]["seconds"] > 0 and written["timing"]["windows_timed_this_segment"] == 4


def test_metric_inference_scores_the_stored_windows_with_the_whole_metric_list(fake_root, tmp_path):
    cfg = inference_cfg(fake_root, tmp_path).updated(
        metrics=[{"error": "mae", "channel": "phi"}, {"descriptor": "saturation", "error": "rel_err", "channel": "phi"}]
    )
    out = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match=JSON_NAME):
        metric_inference(cfg, split="val", out_dir=out)  # nothing rolled out yet
    rolled = rollout_inference(cfg, split="val", out_dir=out)
    payload = metric_inference(cfg, split="val", out_dir=out)

    stage = out / DIR_NAME
    written = json.loads((stage / "metrics_val.json").read_text())
    assert written == json.loads(json.dumps(json_safe(payload)))
    assert written["windows"] == rolled["windows"] and written["steps"] == [1, 2] and written["n_frames"] == 8
    assert written["n_windows_per_step"]["saturation/rel_err@phi"] == [4, 4]
    # the descriptor metric is new, the field metric is the one the GPU stage scored at the same frames
    assert written["summary"]["mae@phi"] == pytest.approx(rolled["summary"]["mae@phi"])
    rows = read_csv(stage / "metrics_val.csv")
    assert [(r["run"], int(r["t0"]), int(r["h"])) for r in rows[:4]] == [(VAL[0], t0, h) for t0 in (0, 2) for h in (1, 2)]
    assert "saturation@phi/pred" in rows[0]  # traced values ride along, as in the studies
    assert (out / CSV_NAME).is_file() and (stage / JSON_NAME).is_file()  # the GPU stage's record is untouched


def test_rollout_inference_scores_exactly_what_the_rollout_scores_per_window(fake_root, tmp_path):
    cfg = inference_cfg(fake_root, tmp_path)
    out = tmp_path / "run"
    payload = rollout_inference(cfg, split="val", out_dir=out)

    from poreml.evaluate import _load

    device, runs, task, _, model = _load(cfg, None, "val")
    dataset = task.dataset(runs["val"])
    metrics = resolve([MetricSpec(error="mae", channel="phi"), MetricSpec(error="mae", channel="p")], task.target_channels)
    rows = read_rows(out / CSV_NAME)
    for i, ref in enumerate(dataset.runs):
        for t0 in (0, 2):
            reference = rollout(model, dataset, i, t0, 2, metrics, device)
            mine = [r for r in rows if r["run"] == ref.run_id and r["t0"] == t0]
            for name in ("mae@phi", "mae@p"):
                assert [r[name] for r in mine] == reference.values[name]
    # per run first, then over runs: with every run reaching every step this is the plain mean over rows
    for h in (1, 2):
        at_h = [r["mae@phi"] for r in rows if r["h"] == h]
        assert payload["summary"]["mae@phi"][h - 1] == pytest.approx(sum(at_h) / len(at_h))
    # the fixture invades one slab per frame: persistence is wrong by h slabs after h steps
    assert payload["summary"]["mae@phi"][1] > payload["summary"]["mae@phi"][0] > 0


def test_rollout_inference_skips_a_finished_stage_and_redoes_it_on_force(fake_root, tmp_path):
    cfg = inference_cfg(fake_root, tmp_path)
    out = tmp_path / "run"
    first = rollout_inference(cfg, split="val", out_dir=out)
    csv_path, json_path = out / CSV_NAME, out / DIR_NAME / JSON_NAME
    stamp = (csv_path.stat().st_mtime_ns, json_path.stat().st_mtime_ns)

    again = rollout_inference(cfg, split="val", out_dir=out)
    assert again == json.loads(json.dumps(json_safe(first)))
    assert (csv_path.stat().st_mtime_ns, json_path.stat().st_mtime_ns) == stamp  # nothing rewritten

    forced = rollout_inference(cfg, split="val", out_dir=out, force=True)
    assert len(read_csv(csv_path)) == 8 and forced["n_windows"] == 4
    assert forced["timing"]["windows_timed_this_segment"] == 4  # everything redone


def test_rollout_inference_resumes_at_the_first_missing_window_after_sigterm(fake_root, tmp_path, monkeypatch):
    cfg = inference_cfg(fake_root, tmp_path)
    out = tmp_path / "run"
    (out / DIR_NAME).mkdir(parents=True)
    stale = out / DIR_NAME / JSON_NAME
    stale.write_text("not a real payload")  # another checkpoint's sentinel, left by an interrupted rerun
    before = signal.getsignal(signal.SIGTERM)
    import poreml.inference as inf

    original = inf.rollout_window
    calls = {"n": 0}

    def sigterm_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            os.kill(os.getpid(), signal.SIGTERM)  # the stage polls its flag after the window in hand
        return original(*args, **kwargs)

    monkeypatch.setattr(inf, "rollout_window", sigterm_once)
    with pytest.raises(Interrupted):
        rollout_inference(cfg, split="val", out_dir=out)
    assert not stale.exists()  # unlinked at the start, before rolling out
    assert signal.getsignal(signal.SIGTERM) is before
    assert calls["n"] == 1 and len(read_csv(out / CSV_NAME)) == 2  # one window: two rows, no sentinel
    assert not (out / DIR_NAME / JSON_NAME).exists()

    monkeypatch.setattr(inf, "rollout_window", original)
    payload = rollout_inference(cfg, split="val", out_dir=out)  # the rerun keeps the finished window
    assert payload["n_windows"] == 4 and payload["n_frames"] == 8
    assert payload["timing"]["windows_timed_this_segment"] == 3
    rows = read_csv(out / CSV_NAME)
    assert len(rows) == 8 and len({(r["run"], r["t0"], r["h"]) for r in rows}) == 8  # no duplicate rows

    # a window whose rows were lost (torn CSV) is redone even though its frames exist
    (out / DIR_NAME / JSON_NAME).unlink()
    kept = [r for r in rows if not (r["run"] == VAL[0] and r["t0"] == "2")]
    with (out / CSV_NAME).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(kept)
    payload = rollout_inference(cfg, split="val", out_dir=out)
    assert payload["timing"]["windows_timed_this_segment"] == 1 and len(read_csv(out / CSV_NAME)) == 8


def test_rollout_inference_reports_fields_the_task_lacks_and_refuses_none(fake_root, tmp_path):
    cfg = inference_cfg(fake_root, tmp_path, fields=({"name": "phi"},))
    payload = rollout_inference(cfg, split="val", out_dir=tmp_path / "run")
    assert payload["fields"] == ["phi"] and payload["fields_missing"] == ["p"]
    rows = read_csv(tmp_path / "run" / CSV_NAME)
    assert "mae@phi" in rows[0] and "mae@p" not in rows[0]
    with pytest.raises(ValueError, match="none of the fields"):
        rollout_inference(cfg, split="val", out_dir=tmp_path / "other", fields=("rho",))


def test_rollout_inference_stores_twelve_keyframes_when_the_config_names_none(fake_root, tmp_path):
    cfg = inference_cfg(fake_root, tmp_path, horizon=4, keyframes=None)
    assert cfg.rollout.keyframes is None  # every training config predates the knob
    payload = rollout_inference(cfg, split="val", out_dir=tmp_path / "run")
    assert payload["protocol"]["keyframes"] == 12 and payload["protocol"]["steps"] == keyframe_steps(4, 12)
    payload = rollout_inference(cfg, split="val", out_dir=tmp_path / "other", keyframes=2)
    assert payload["protocol"]["keyframes"] == 2 and payload["protocol"]["steps"] == [1, 4]
    rows = read_csv(tmp_path / "other" / CSV_NAME)
    assert [int(r["h"]) for r in rows if r["stored"] == "True"] == [1, 4, 1, 4]


def test_rollout_inference_windows_from_frame_zero_at_the_stride_and_lists_short_runs(fake_root, tmp_path):
    cfg = inference_cfg(fake_root, tmp_path, horizon=3)
    payload = rollout_inference(cfg, split="val", out_dir=tmp_path / "a", stride=1)
    assert payload["protocol"]["stride"] == 1 and all(v == [0, 1, 2] for v in payload["windows"].values())
    assert payload["n_windows"] == 6 and payload["n_windows_per_step"]["mae@phi"] == [6, 6, 6]
    assert payload["n_runs_per_step"]["mae@phi"] == [2, 2, 2]
    payload = rollout_inference(cfg, split="val", out_dir=tmp_path / "b")  # stride = horizon: [0] only (3 has no room)
    assert all(v == [0] for v in payload["windows"].values())
    cfg = inference_cfg(fake_root, tmp_path, horizon=6)
    payload = rollout_inference(cfg, split="val", out_dir=tmp_path / "c")
    assert payload["runs_without_windows"] == sorted(VAL) and payload["n_windows"] == 0 and payload["n_frames"] == 0
    assert (tmp_path / "c" / DIR_NAME / JSON_NAME).is_file()  # a stage with no windows is still finished


# --- the driver: util/inference/inference.py -------------------------------------------------------

spec = importlib.util.spec_from_file_location("case_inference", Path("util/inference/inference.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def _run_dir(root: Path, job: str, name: str, status: str = "finished", ckpts=("best.pt", "last.pt")) -> Path:
    run = root / job / "ckpts" / name
    (run / "ckpts").mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"status": status}))
    (run / "config.yaml").write_text("name: x\n")
    for c in ckpts:
        (run / "ckpts" / c).write_bytes(b"ckpt")
    return run


def test_driver_picks_best_for_train_and_best_rollout_for_train_push(tmp_path):
    base = _run_dir(tmp_path / "case" / "train", "drainage/unet_gen", "unet_20260904", ckpts=("best.pt", "last.pt"))
    push = _run_dir(
        tmp_path / "case" / "train_push", "drainage/unet_gen", "unet_push", ckpts=("best.pt", "last.pt", "best_rollout.pt")
    )
    assert driver.choose_checkpoint(base, "auto") == base / "ckpts" / "best.pt"
    assert driver.choose_checkpoint(push, "auto") == push / "ckpts" / "best_rollout.pt"
    assert driver.choose_checkpoint(base, "last") == base / "ckpts" / "last.pt"
    assert driver.choose_checkpoint(base, "best_rollout") is None  # not finalised: nothing to roll out
    with pytest.raises(ValueError):
        driver.choose_checkpoint(base, "newest")


def test_driver_finds_the_newest_finished_run_per_job_and_skips_underscored_dirs(tmp_path):
    root = tmp_path / "case" / "train"
    old = _run_dir(root, "drainage/unet_gen", "unet_20260901")
    new = _run_dir(root, "drainage/unet_gen", "unet_20260904")
    running = _run_dir(root, "drainage/unet_gen", "unet_20260909", status="running")
    failed = _run_dir(root, "gdl/fno_gen", "fno_20260905", status="failed")
    _run_dir(root / "_archive", "drainage/unet_gen", "unet_20260830")
    found = driver.find_runs([root])
    assert found == [new]
    assert driver.find_runs([root / "drainage" / "unet_gen"]) == [new]
    assert driver.find_runs([running]) == []  # a run dir given directly still has to be finished
    assert driver.find_runs([old]) == [old]  # ...but an older finished one is taken as named
    assert driver.find_runs([failed.parents[1]]) == []


def test_driver_states(tmp_path):
    run = _run_dir(tmp_path / "case" / "train", "drainage/unet_gen", "unet_20260904")
    ckpt = run / "ckpts" / "best.pt"
    assert driver.state(run, ckpt) == "pending"
    assert driver.state(run, None) == "waiting"
    (run / DIR_NAME / "frames" / "some_run").mkdir(parents=True)
    (run / DIR_NAME / "frames" / "some_run" / "w000000.h5").write_bytes(b"")
    assert driver.state(run, ckpt) == "resume"  # windows stored, no sentinel yet
    (run / DIR_NAME / JSON_NAME).write_text(json.dumps({"checkpoint_sha256": driver.sha256_of(ckpt)}))
    assert driver.state(run, ckpt) == "finished"
    ckpt.write_bytes(b"other weights")
    assert driver.state(run, ckpt) == "stale"


# --- the submitter: util/inference/submit.py ----------------------------------------------------

_sspec = importlib.util.spec_from_file_location("case_submit", Path("util/inference/submit.py"))
submit = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(submit)


def test_submit_estimates_limits_and_qos():
    # drainage_all unet: 189 windows x 64 steps at 0.55 s -> 1.8 h: a short cell -> normal, smallest bucket above it
    est = submit.estimate_hours("unet3d", "voxel", "drainage", 189 * 64)
    assert est == pytest.approx(1.848)
    assert submit.qos_for(est) == "normal" and submit.time_limit(est, "normal") == "02:00:00"
    # underfill abupt: 44 windows, the point factor -> 5.3 h: long -> compact, doubled fits 16 h
    est = submit.estimate_hours("abupt", "points", "underfill", 44 * 64)
    assert est == pytest.approx(5.32, abs=0.01)
    assert submit.qos_for(est) == "compact" and submit.time_limit(est, "compact") == "16:00:00"
    assert submit.time_limit(0.3, "normal") == "01:00:00"  # trapping cells backfill into hour-long holes
    # beyond compact's day even undoubled: normal, and beyond every bucket the normal ceiling
    assert submit.qos_for(13.0) == "normal" and submit.time_limit(30.0, "normal") == submit.NORMAL_LIMIT
    assert submit.qos_for(0.3, "compact") == "compact" and submit.qos_for(5.0, "normal") == "normal"  # pinned
    assert submit.hours("1-02:30:00") == 26.5 and submit.hours("23:00:00") == 23


def test_submit_splits_the_throttle_by_array_hours():
    assert submit.split_throttle(12, [30.0, 15.0, 5.0]) == [7, 4, 1]
    assert submit.split_throttle(12, [16.0, 20.0, 21.0, 22.0]) == [2, 3, 3, 3]  # the 2026-09-12 sweep: near-equal hours
    assert submit.split_throttle(2, [30.0, 15.0, 5.0]) == [1, 1, 1]  # at least one task per array
    assert submit.split_throttle(0, [3.0, 3.0]) == [0, 0]  # no cap
    assert sum(submit.split_throttle(12, [50.0])) == 12
