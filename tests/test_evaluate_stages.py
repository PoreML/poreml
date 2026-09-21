"""The study stages on the fixture: inference stores windows and writes its sentinel JSON; the
metric stage (Task 5) scores them and equals in-process scoring."""

import json
import os
import signal

import pytest
import yaml

from poreml.config import Config
from poreml.evaluate import FRAMES_DIR, inference_split
from poreml.frames import H5FrameStore
from poreml.metrics import json_safe
from poreml.rollout import Interrupted, keyframe_steps


def staged_cfg(fake_root, tmp_path, horizon=2, keyframes=2, metrics=None):
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"train": [], "val": [], "test": ["tiny_0002_M10_theta140", "tiny_0000_M1_theta120"]}))
    metrics = metrics or [
        {"error": "mae", "channel": "phi"},
        {"descriptor": "saturation", "error": "rel_err", "channel": "phi"},
        {"descriptor": "euler", "error": "target", "channel": "phi", "eval_only": True},
    ]
    return Config.model_validate(
        {
            "name": "staged",
            "data": {"root": str(fake_root), "campaign": "drainage", "split": str(split)},
            "task": {"name": "next_frame", "history": 1},
            "model": {"name": "persistence"},
            "train": {"epochs": 1, "batch_size": 1, "device": "cpu", "max_batches": 1, "num_workers": 0},
            "metrics": metrics,
            "rollout": {"horizon": horizon, "keyframes": keyframes},
            "out_dir": str(tmp_path / "runs"),
        }
    )


def test_inference_split_stores_every_window_and_writes_the_sentinel(fake_root, tmp_path):
    cfg = staged_cfg(fake_root, tmp_path)
    out = tmp_path / "cell"
    payload = inference_split(cfg, out_dir=out, stride=2)
    written = json.loads((out / "inference_test.json").read_text())
    assert written == json.loads(json.dumps(json_safe(payload)))
    assert written["protocol"] == {"horizon": 2, "stride": 2, "keyframes": 2, "steps": keyframe_steps(2, 2)}
    assert written["windows"] == {"tiny_0002_M10_theta140": [0, 2], "tiny_0000_M1_theta120": [0, 2]}
    assert written["runs_without_windows"] == [] and written["n_windows"] == 4 and written["n_frames"] == 8
    assert written["frames_dir"] == FRAMES_DIR and written["checkpoint"] is None and written["precision"] == "tf32"
    store = H5FrameStore(out / FRAMES_DIR)
    assert store.runs() == sorted(written["windows"]) and store.steps("tiny_0000_M1_theta120", 2) == [1, 2]


def test_inference_split_defaults_the_stride_to_the_horizon_and_lists_short_runs(fake_root, tmp_path):
    cfg = staged_cfg(fake_root, tmp_path, horizon=3)
    payload = inference_split(cfg, out_dir=tmp_path / "a")
    assert payload["protocol"]["stride"] == 3 and all(v == [0] for v in payload["windows"].values())
    cfg = staged_cfg(fake_root, tmp_path, horizon=6)
    payload = inference_split(cfg, out_dir=tmp_path / "b")
    assert payload["runs_without_windows"] == sorted(payload["windows"]) and payload["n_windows"] == 0
    assert (tmp_path / "b" / "inference_test.json").is_file()  # a study with no windows is still a finished inference


def test_inference_split_stops_on_sigterm_without_the_sentinel_and_resumes(fake_root, tmp_path, monkeypatch):
    cfg = staged_cfg(fake_root, tmp_path)
    out = tmp_path / "cell"
    out.mkdir(parents=True)
    stale_sentinel = out / "inference_test.json"
    stale_sentinel.write_text("not a real inference payload")  # a stale sentinel from another checkpoint's interrupted run
    before = signal.getsignal(signal.SIGTERM)
    import poreml.evaluate as ev

    original = ev.infer_windows
    calls = {"n": 0}

    def sigterm_once(*args, **kwargs):
        # deliver SIGTERM to ourselves before the first window: infer_windows polls stop() after it
        calls["n"] += 1
        if calls["n"] == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return original(*args, **kwargs)

    monkeypatch.setattr(ev, "infer_windows", sigterm_once)
    with pytest.raises(Interrupted):
        inference_split(cfg, out_dir=out, stride=2)
    assert not stale_sentinel.exists()  # unlinked at the start, before rolling out
    assert signal.getsignal(signal.SIGTERM) is before
    stored = sum(len(H5FrameStore(out / FRAMES_DIR).windows(r)) for r in ("tiny_0002_M10_theta140", "tiny_0000_M1_theta120"))
    assert stored == 1
    payload = inference_split(cfg, out_dir=out, stride=2)  # the rerun completes the rest
    assert payload["n_windows"] == 4 and (out / "inference_test.json").is_file()


def test_metric_split_scores_the_stored_frames_and_equals_in_process_scoring(fake_root, tmp_path):
    from poreml.evaluate import metric_split
    from poreml.rollout import rollout, score_frame

    cfg = staged_cfg(fake_root, tmp_path)
    out = tmp_path / "cell"
    inference_split(cfg, out_dir=out, stride=2)
    payload = metric_split(cfg, out_dir=out, workers=1)
    written = json.loads((out / "metrics_test.json").read_text())
    assert written == json.loads(json.dumps(json_safe(payload)))
    assert written["steps"] == [1, 2] and written["protocol"]["stride"] == 2 and written["n_frames"] == 8
    assert written["n_windows"] == 4
    assert written["n_runs"] == 2
    names = ["mae@phi", "saturation/rel_err@phi", "euler/target@phi"]
    assert all(n in written["summary"] for n in names) and "saturation@phi/target" in written["summary"]  # traced values too
    assert written["n_runs_per_step"]["mae@phi"] == [2, 2] and written["n_windows_per_step"]["mae@phi"] == [4, 4]
    assert written["at_horizon"]["mae@phi"] == written["summary"]["mae@phi"][-1]
    assert (
        written["metric"]["workers"] == 1
        and written["metric"]["metrics_version"]
        and written["metric"]["checkpoint_sha256"] is None
    )
    assert written["groups"] == {}  # the fixture's family is "tiny": not in study.SOURCES, so no group

    # the CSV: one row per stored frame with the raw numbers
    import csv

    rows = list(csv.DictReader((out / "metrics_test.csv").open()))
    assert len(rows) == 8 and {r["run"] for r in rows} == set(written["windows"])
    assert sorted(set(r["h"] for r in rows)) == ["1", "2"] and "mae@phi" in rows[0] and "t0" in rows[0]

    # equals the in-process value: roll the same window out and score its frame directly
    from poreml.evaluate import _load

    _, runs, task, metrics, model = _load(cfg, None, "test")
    ds = task.dataset(runs["test"])
    full = rollout(model, ds, 0, 2, 2, metrics, "cpu", keep_frames=True)
    direct = score_frame(ds, metrics, 0, 2, 2, full.frames[1])
    row = next(r for r in rows if r["run"] == ds.runs[0].run_id and r["t0"] == "2" and r["h"] == "2")
    assert (
        float(row["mae@phi"]) == pytest.approx(direct["mae@phi"])
        and float(row["euler/target@phi"]) == direct["euler/target@phi"]
    )


def test_metric_split_parallel_equals_serial(fake_root, tmp_path):
    from poreml.evaluate import metric_split
    from poreml.metrics import json_safe

    cfg = staged_cfg(fake_root, tmp_path)
    inference_split(cfg, out_dir=tmp_path / "p", stride=2)
    inference_split(cfg, out_dir=tmp_path / "s", stride=2)
    par = metric_split(cfg, out_dir=tmp_path / "p", workers=2)
    ser = metric_split(cfg, out_dir=tmp_path / "s", workers=1)
    assert json_safe(par["summary"]) == json_safe(ser["summary"]) and par["n_windows"] == ser["n_windows"]
    assert (tmp_path / "p" / "metrics_test.csv").read_text() == (tmp_path / "s" / "metrics_test.csv").read_text()


def test_metric_split_refuses_a_missing_sentinel_a_foreign_checkpoint_and_a_missing_window(fake_root, tmp_path):
    from poreml.evaluate import metric_split

    cfg = staged_cfg(fake_root, tmp_path)
    with pytest.raises(FileNotFoundError, match="inference_test.json"):
        metric_split(cfg, out_dir=tmp_path / "none")

    out = tmp_path / "a"
    inference_split(cfg, out_dir=out, stride=2)
    sentinel = out / "inference_test.json"
    payload = json.loads(sentinel.read_text())
    payload["checkpoint_sha256"] = "deadbeef"
    sentinel.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint"):
        metric_split(cfg, out_dir=out)

    out = tmp_path / "b"
    inference_split(cfg, out_dir=out, stride=2)
    (out / FRAMES_DIR / "tiny_0000_M1_theta120" / "w000002.h5").unlink()
    with pytest.raises(ValueError, match="w000002"):
        metric_split(cfg, out_dir=out)


def test_metrics_payload_and_write_metrics_are_what_metric_split_uses(fake_root, tmp_path):
    """A synthetic study (case_dummy) builds its files through the same two helpers as the real
    metric stage, so the JSON keys and the CSV columns can never drift apart."""
    from poreml.evaluate import metric_split, metrics_payload, write_metrics

    cfg = staged_cfg(fake_root, tmp_path)
    out = tmp_path / "cell"
    inference_split(cfg, out_dir=out, stride=2)
    real = metric_split(cfg, out_dir=out, workers=1)

    rows = [{"run": "r", "family": "blob", "group": "gen", "t0": 0, "h": 1, "t": 1, "mae@phi": 0.1, "x@phi/pred": 2.0}]
    header = {
        k: real[k]
        for k in (
            "name",
            "split",
            "task",
            "model",
            "checkpoint",
            "checkpoint_sha256",
            "precision",
            "n_runs",
            "runs",
            "provenance",
        )
    }
    inference = json.loads((out / "inference_test.json").read_text())
    payload = metrics_payload(
        header, inference, real["metrics_spec"], rows, ["mae@phi", "x@phi/pred"], {"gen": ["r"]}, {"workers": 1}
    )
    assert list(payload) == list(real)  # same keys, same order
    assert payload["summary"]["mae@phi"] == [0.1, float("nan")] or payload["summary"]["mae@phi"][0] == 0.1
    assert payload["at_horizon"]["x@phi/pred"] != payload["at_horizon"]["x@phi/pred"]  # NaN at an unscored step
    target = tmp_path / "written"
    target.mkdir()
    write_metrics(target, "test", rows, payload)
    assert (target / "metrics_test.json").is_file() and (target / "metrics_test.csv").read_text().splitlines()[0].startswith(
        "run,family,group,t0,h,t,mae@phi"
    )
