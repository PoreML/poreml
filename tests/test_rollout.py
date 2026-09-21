import json
import math

import numpy as np
import pytest
import torch
import yaml

from poreml.config import Config
from poreml.data import SOLID_FILL, WindowDataset, discover
from poreml.evaluate import rollout_split
from poreml.metrics import MetricSpec, resolve
from poreml.models import Persistence
from poreml.rollout import RolloutResult, RolloutSummary, evaluate_rollout, rollout, start_index
from poreml.train import train


@pytest.fixture
def runs(fake_root):
    return list(discover(fake_root, "drainage").values())


@pytest.fixture
def metrics():
    return resolve([MetricSpec(error="mae"), MetricSpec(descriptor="saturation", error="abs_err")], ("phi",))


def test_persistence_rollout_error_grows_one_slab_per_step(runs, metrics):
    # The fixture invades one slab per frame. Persistence fed back on itself never moves,
    # so after h steps it is wrong on exactly the ROI slabs 0..h-1 (slab 0 of the domain
    # is the inlet, outside the rock ROI), by |dphi| = 2 on every pore voxel there.
    ds = WindowDataset(runs[:1], history=1)
    pore = ~ds.solid(0)

    result = rollout(Persistence(2), ds, run_idx=0, t0=0, horizon=4, metrics=metrics, device="cpu", keep_frames=True)

    assert isinstance(result, RolloutResult)
    assert result.run_id == runs[0].run_id and result.t0 == 0 and result.horizon == 4
    expected = [2.0 * pore[:, :, :h].sum() / pore.sum() for h in range(1, 5)]
    assert result.values["mae"] == pytest.approx(expected)
    assert all(a < b for a, b in zip(result.values["mae"], result.values["mae"][1:], strict=False))
    assert len(result.frames) == 4 and result.frames[0].shape == (1,) + pore.shape
    assert np.all(result.frames[0][0][~pore] == -1.0)  # solid is forced to the fill value


def test_rollout_uses_the_history_window_and_feeds_predictions_back(runs, metrics):
    # A model that returns the *older* of two history frames: after one step the window is
    # (frame 1, frame 0-as-prediction), so step two predicts frame 1 again, and so on.
    class Older(torch.nn.Module):
        def forward(self, x):
            return x[:, 1:2]

    ds = WindowDataset(runs[:1], history=2)

    result = rollout(Older(), ds, run_idx=0, t0=1, horizon=3, metrics=metrics, device="cpu", keep_frames=True)

    assert np.array_equal(result.frames[0], ds.frame(0, 0))
    assert np.array_equal(result.frames[1], ds.frame(0, 1))
    assert np.array_equal(result.frames[2], ds.frame(0, 0))


def test_rollout_caps_the_horizon_at_the_run_length_and_records_it(runs, metrics):
    ds = WindowDataset(runs[:1], history=1)  # 6 frames: from t0 = 0 at most 5 targets

    result = rollout(Persistence(2), ds, run_idx=0, t0=0, horizon=64, metrics=metrics, device="cpu")

    assert result.horizon == 5
    assert len(result.values["mae"]) == 5


def test_rollout_refuses_a_start_with_no_target_or_no_history(runs, metrics):
    ds = WindowDataset(runs[:1], history=2)

    with pytest.raises(ValueError, match="history"):
        rollout(Persistence(3), ds, run_idx=0, t0=0, horizon=1, metrics=metrics, device="cpu")
    with pytest.raises(ValueError, match="no frame"):
        rollout(Persistence(3), ds, run_idx=0, t0=5, horizon=1, metrics=metrics, device="cpu")


def test_evaluate_rollout_summarises_over_runs_per_horizon(runs, metrics):
    ds = WindowDataset(runs, history=1)

    summary = evaluate_rollout(Persistence(2), ds, metrics, horizon=3, device="cpu")

    assert isinstance(summary, RolloutSummary)
    assert set(summary.curves["mae"]) == {r.run_id for r in runs}
    # 6 frames: a quarter of the way in is round(0.25 * 5) = 1, leaving 4 true frames after it.
    assert summary.horizon == 3 and all(v["horizon"] == 3 and v["t0"] == 1 for v in summary.runs.values())
    per_h = np.array([summary.curves["mae"][r.run_id] for r in runs])
    assert summary.summary["mae"] == pytest.approx(per_h.mean(axis=0).tolist())
    assert summary.n_runs["mae"] == [3, 3, 3]
    assert summary.at_horizon["mae"] == pytest.approx(summary.summary["mae"][-1])


def test_evaluate_rollout_averages_over_the_runs_that_reach_each_horizon(tmp_path, metrics):
    from tests.fixtures import write_fake_run

    write_fake_run(tmp_path, "drainage", "long_0000_M1_theta120", n_steps=6)
    write_fake_run(tmp_path, "drainage", "short_0001_M1_theta120", n_steps=3)
    ds = WindowDataset(list(discover(tmp_path, "drainage").values()), history=1)

    summary = evaluate_rollout(Persistence(2), ds, metrics, horizon=4, device="cpu", start=0)

    assert summary.n_runs["mae"] == [2, 2, 1, 1]
    assert summary.runs["short_0001_M1_theta120"]["horizon"] == 2
    assert all(math.isfinite(v) for v in summary.summary["mae"])


def test_rollout_eval_stride_scores_the_grid_and_the_final_step(runs, metrics):
    # eval_stride k scores steps k, 2k, ... plus the final step; unscored steps are None.
    # The model still steps through every frame — only the metric computation is skipped.
    ds = WindowDataset(runs[:1], history=1)

    full = rollout(Persistence(2), ds, run_idx=0, t0=0, horizon=4, metrics=metrics, device="cpu")
    strided = rollout(Persistence(2), ds, run_idx=0, t0=0, horizon=4, metrics=metrics, device="cpu", eval_stride=2)

    assert [v is None for v in strided.values["mae"]] == [True, False, True, False]
    assert strided.values["mae"][1] == pytest.approx(full.values["mae"][1])
    assert strided.values["mae"][3] == pytest.approx(full.values["mae"][3])


def test_rollout_eval_stride_beyond_the_horizon_scores_only_the_final_step(runs, metrics):
    ds = WindowDataset(runs[:1], history=1)

    result = rollout(Persistence(2), ds, run_idx=0, t0=0, horizon=4, metrics=metrics, device="cpu", eval_stride=64)

    assert [v is None for v in result.values["mae"]] == [True, True, True, False]
    assert math.isfinite(result.values["mae"][-1])


def test_evaluate_rollout_with_eval_stride_keeps_at_horizon_and_counts_scored_runs(runs, metrics):
    ds = WindowDataset(runs, history=1)

    summary = evaluate_rollout(Persistence(2), ds, metrics, horizon=3, device="cpu", eval_stride=3)

    assert math.isnan(summary.summary["mae"][0]) and math.isnan(summary.summary["mae"][1])
    assert math.isfinite(summary.summary["mae"][2])
    assert summary.n_runs["mae"] == [0, 0, 3]
    assert summary.at_horizon["mae"] == pytest.approx(summary.summary["mae"][-1])


@pytest.fixture
def cfg(fake_root, tmp_path):
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {"train": ["tiny_0000_M1_theta120"], "val": ["tiny_0001_M1_theta130"], "test": ["tiny_0002_M10_theta140"]}
        )
    )
    return Config.model_validate(
        {
            "name": "smoke",
            "data": {"root": str(fake_root), "campaign": "drainage", "split": str(split)},
            "task": {"name": "next_frame", "history": 1},
            "model": {"name": "persistence"},
            "train": {"epochs": 1, "batch_size": 1, "device": "cpu", "max_batches": 2},
            "rollout": {"horizon": 3},
            "out_dir": str(tmp_path / "runs"),
        }
    )


def test_rollout_config_defaults_and_strictness():
    assert (
        Config.model_validate(
            {
                "name": "x",
                "data": {"root": ".", "campaign": "drainage", "split": "s.yaml"},
                "task": {"name": "next_frame"},
                "model": {"name": "persistence"},
            }
        ).rollout.horizon
        == 64
    )
    with pytest.raises(Exception, match="horizon"):
        Config.model_validate(
            {
                "name": "x",
                "data": {"root": ".", "campaign": "drainage", "split": "s.yaml"},
                "task": {"name": "next_frame"},
                "model": {"name": "persistence"},
                "rollout": {"horizon": 0},
            }
        )


def test_rollout_split_writes_a_strict_json_artifact_beside_the_checkpoint(cfg):
    run_dir = train(cfg)
    ckpt = run_dir / "ckpts" / "best.pt"

    summary = rollout_split(cfg, ckpt=ckpt, split="test")

    path = ckpt.parent / "rollout_test.json"
    written = json.loads(path.read_text(), parse_constant=lambda t: pytest.fail(f"non-JSON token {t}"))
    assert written["split"] == "test" and written["horizon"] == 3
    assert written["summary"]["mae"] == pytest.approx(summary.summary["mae"])
    assert written["curves"]["mae"]["tiny_0002_M10_theta140"] == pytest.approx(summary.curves["mae"]["tiny_0002_M10_theta140"])
    assert written["runs"][0]["run_id"] == "tiny_0002_M10_theta140" and "provenance" in written
    assert written["at_horizon"]["mae"] == pytest.approx(summary.summary["mae"][-1])


def test_rollout_split_honours_a_horizon_override(cfg):
    run_dir = train(cfg)

    summary = rollout_split(cfg, ckpt=run_dir / "ckpts" / "best.pt", split="test", horizon=2)

    assert summary.horizon == 2 and len(summary.summary["mae"]) == 2


def test_start_index_defaults_to_a_quarter_of_the_run_and_respects_the_bounds(runs):
    ds1 = WindowDataset(runs[:1], history=1)  # 6 frames
    ds3 = WindowDataset(runs[:1], history=3)

    assert start_index(ds1, 0) == 1  # round(0.25 * 5)
    assert start_index(ds1, 0, start_fraction=0.0) == 0
    assert start_index(ds1, 0, start_fraction=0.99) == 4  # never past the last frame with a target
    assert start_index(ds3, 0) == 2  # never before the history window
    assert start_index(ds1, 0, start=3) == 3  # an explicit start wins


def test_rollout_config_start_fraction_is_bounded():
    base = {
        "name": "x",
        "data": {"root": ".", "campaign": "drainage", "split": "s.yaml"},
        "task": {"name": "next_frame"},
        "model": {"name": "persistence"},
    }
    assert Config.model_validate(base).rollout.start_fraction == 0.25
    with pytest.raises(Exception, match="start_fraction"):
        Config.model_validate({**base, "rollout": {"start_fraction": 1.0}})


def test_rollout_split_writes_the_model_alone(cfg):
    class _Zero(torch.nn.Module):
        def __init__(self, in_channels, out_channels: int = 1):
            super().__init__()
            self.out_channels = out_channels

        def forward(self, x):
            return torch.zeros_like(x[:, -self.out_channels :])

    from poreml.registry import MODELS

    MODELS.register("_test_zero")(_Zero)
    try:
        cfg.model.name = "_test_zero"
        run_dir = train(cfg)
        summary = rollout_split(cfg, ckpt=run_dir / "ckpts" / "best.pt", split="test")
    finally:
        MODELS._items.pop("_test_zero")

    assert not hasattr(summary, "baseline")
    written = json.loads((run_dir / "ckpts" / "rollout_test.json").read_text())
    assert "baseline" not in written
    assert written["summary"]["mae"] == pytest.approx(summary.summary["mae"])


def test_rollout_feeds_back_every_channel_and_forces_solid_to_each_fill(runs):
    from poreml.data import FieldSpec

    fields = [FieldSpec(name="phi"), FieldSpec(name="p", offset=0.33, scale=0.001, fill=5.0)]
    ds = WindowDataset(runs, history=1, fields=fields)
    metrics = resolve([MetricSpec(error="mae", channel="phi"), MetricSpec(error="mae", channel="p")], ds.channels)
    model = Persistence(in_channels=3, out_channels=2)

    result = rollout(model, ds, 0, 0, 3, metrics, "cpu", keep_frames=True)

    solid = ds.solid(0)
    assert result.frames[0].shape == (2,) + solid.shape
    assert (result.frames[-1][0][solid] == SOLID_FILL).all() and (result.frames[-1][1][solid] == 5.0).all()
    # persistence on p never moves, so its p error grows with the fixture's per-frame rise
    assert result.values["mae@p"][2] > result.values["mae@p"][0]


def test_persistence_rollout_on_the_point_stream_matches_the_voxel_stream(runs, metrics):
    from poreml.points import PointWindowDataset

    vox = WindowDataset(runs[:1], history=1)
    pts = PointWindowDataset(runs[:1], history=1)
    a = rollout(Persistence(2), vox, run_idx=0, t0=0, horizon=3, metrics=metrics, device="cpu")
    b = rollout(Persistence(4), pts, run_idx=0, t0=0, horizon=3, metrics=metrics, device="cpu")
    assert a.values == pytest.approx(b.values)


# --- keyframes ---------------------------------------------------------------------------


def test_scored_steps_is_the_eval_stride_grid_plus_the_final_step():
    from poreml.rollout import scored_steps

    assert scored_steps(4, 1) == [1, 2, 3, 4]
    assert scored_steps(4, 2) == [2, 4]
    assert scored_steps(4, 64) == [4]


def test_keyframe_steps_none_means_the_scored_steps():

    assert keyframe_steps(4, None, eval_stride=2) == [2, 4]
    assert keyframe_steps(64, None, eval_stride=64) == [64]


def test_keyframe_steps_take_first_last_and_evenly_spaced_between():

    assert keyframe_steps(64, 2) == [1, 64]
    steps = keyframe_steps(64, 12)
    assert steps[0] == 1 and steps[-1] == 64 and len(steps) == 12 and steps == sorted(set(steps))
    assert max(b - a for a, b in zip(steps, steps[1:], strict=False)) <= 6


def test_keyframe_steps_short_horizon_yields_every_step():

    assert keyframe_steps(5, 12) == [1, 2, 3, 4, 5]
    assert keyframe_steps(1, 12) == [1]


# --- windows and the inference stage ----------------------------------------------------

from poreml.frames import MemoryFrameStore  # noqa: E402
from poreml.rollout import Interrupted, infer_windows, keyframe_steps, score_frame, window_starts  # noqa: E402

MESH = MetricSpec(descriptor="euler", error="target", eval_only=True)  # cheap stand-in for the mesh family


@pytest.fixture
def mixed():
    return resolve([MetricSpec(error="mae"), MetricSpec(descriptor="saturation", error="abs_err"), MESH], ("phi",))


def test_window_starts_tile_the_run_with_full_windows_only():
    assert window_starts(n_steps=6, history=1, horizon=2, stride=2) == [0, 2]  # t0=4: 4 + 2 = 6 > 5, discarded
    assert window_starts(n_steps=7, history=1, horizon=2, stride=2) == [0, 2, 4]
    assert window_starts(n_steps=200, history=1, horizon=64, stride=64) == [0, 64, 128]  # 128 + 64 = 192 <= 199
    assert window_starts(n_steps=193, history=1, horizon=64, stride=64) == [0, 64, 128]  # 192 <= 192: just fits
    assert window_starts(n_steps=192, history=1, horizon=64, stride=64) == [0, 64]  # 192 > 191: discarded
    assert window_starts(n_steps=6, history=2, horizon=2, stride=2) == [1, 3]  # the history window comes first
    assert window_starts(n_steps=64, history=1, horizon=64, stride=64) == []  # too short for one window
    assert window_starts(n_steps=6, history=1, horizon=2, stride=3) == [0, 3]  # a stride wider than the horizon leaves gaps


def test_full_stage_scores_eval_only_metrics_at_keyframes_only(runs, mixed):
    ds = WindowDataset(runs[:1], history=1)
    r = rollout(Persistence(2), ds, 0, 0, 4, mixed, "cpu", keyframes=2)
    assert r.keyframes == [1, 4]
    assert [v is None for v in r.values["mae"]] == [False] * 4
    assert [v is None for v in r.values["euler/target"]] == [False, True, True, False]


def test_full_stage_without_keyframes_is_unchanged(runs, mixed):
    ds = WindowDataset(runs[:1], history=1)
    r = rollout(Persistence(2), ds, 0, 0, 4, mixed, "cpu", eval_stride=2)
    assert r.keyframes == [2, 4]
    assert [v is None for v in r.values["euler/target"]] == [True, False, True, False]


def test_inference_stores_the_keyframes_and_scores_nothing(runs, mixed):
    ds = WindowDataset(runs[:1], history=1)
    store = MemoryFrameStore(fields=ds.channels)
    r = rollout(Persistence(2), ds, 0, 2, 3, mixed, "cpu", keyframes=2, store=store, stage="inference", checkpoint_sha256="abc")
    assert r.keyframes == [1, 3] and store.windows(r.run_id) == [2] and store.steps(r.run_id, 2) == [1, 3]
    meta = store.meta(r.run_id, 2)
    assert meta["t0"] == 2 and meta["horizon"] == 3 and meta["steps"] == [1, 3] and meta["checkpoint_sha256"] == "abc"
    assert all(v is None for name in r.values for v in r.values[name])  # nothing scored
    full = rollout(Persistence(2), ds, 0, 2, 3, mixed, "cpu", keep_frames=True)
    for h in r.keyframes:
        assert np.array_equal(store.read(r.run_id, 2, h), full.frames[h - 1])


def test_score_frame_equals_the_in_process_value(runs, mixed):
    ds = WindowDataset(runs[:1], history=1)
    full = rollout(Persistence(2), ds, 0, 0, 4, mixed, "cpu", keep_frames=True)
    scored = score_frame(ds, mixed, 0, 0, 3, full.frames[2])
    assert scored["mae"] == full.values["mae"][2] and scored["euler/target"] == full.values["euler/target"][2]
    assert any(k.startswith("euler") and k != "euler/target" for k in scored)  # traced values ride along


def test_inference_requires_a_store(runs, mixed):
    ds = WindowDataset(runs[:1], history=1)
    with pytest.raises(ValueError, match="store"):
        rollout(Persistence(2), ds, 0, 0, 4, mixed, "cpu", stage="inference")


def test_infer_windows_covers_every_run_skips_stored_windows_and_lists_short_runs(runs):
    ds = WindowDataset(runs, history=1)  # three 6-frame runs
    store = MemoryFrameStore(fields=ds.channels)
    windows, short = infer_windows(
        Persistence(2), ds, horizon=2, stride=2, keyframes=2, store=store, device="cpu", checkpoint_sha256="s"
    )
    assert windows == {r.run_id: [0, 2] for r in runs} and short == []
    assert all(store.windows(r.run_id) == [0, 2] for r in runs)

    calls = []
    original = store.open

    def counting_open(run_id, t0, **meta):
        calls.append((run_id, t0))
        original(run_id, t0, **meta)

    store.open = counting_open
    again, _ = infer_windows(
        Persistence(2), ds, horizon=2, stride=2, keyframes=2, store=store, device="cpu", checkpoint_sha256="s"
    )
    assert again == windows and calls == []  # every window already stored for this checkpoint: nothing redone

    infer_windows(Persistence(2), ds, horizon=2, stride=2, keyframes=2, store=store, device="cpu", checkpoint_sha256="other")
    assert len(calls) == 6  # another checkpoint: every window rewritten

    _, short = infer_windows(Persistence(2), ds, horizon=6, stride=6, keyframes=2, store=store, device="cpu")
    assert short == sorted(r.run_id for r in runs)  # 6 frames: a 6-step window needs 7


def test_infer_windows_rewrites_every_window_when_keyframes_changes(runs):
    ds = WindowDataset(runs, history=1)  # three 6-frame runs
    store = MemoryFrameStore(fields=ds.channels)
    infer_windows(Persistence(2), ds, horizon=4, stride=4, keyframes=2, store=store, device="cpu", checkpoint_sha256="s")
    assert keyframe_steps(4, 2) != keyframe_steps(4, 3)  # the two calls must store genuinely different step sets
    for r in runs:
        assert store.steps(r.run_id, 0) == keyframe_steps(4, 2)

    calls = []
    original = store.open

    def counting_open(run_id, t0, **meta):
        calls.append((run_id, t0))
        original(run_id, t0, **meta)

    store.open = counting_open
    infer_windows(Persistence(2), ds, horizon=4, stride=4, keyframes=3, store=store, device="cpu", checkpoint_sha256="s")
    assert len(calls) == len(runs)  # same checkpoint, different keyframes: every window rewritten, not trusted as stored
    for r in runs:
        assert store.steps(r.run_id, 0) == keyframe_steps(4, 3)


def test_infer_windows_stops_between_windows_when_asked(runs):
    ds = WindowDataset(runs, history=1)
    store = MemoryFrameStore(fields=ds.channels)
    with pytest.raises(Interrupted):
        infer_windows(Persistence(2), ds, horizon=2, stride=2, keyframes=2, store=store, device="cpu", stop=lambda: True)
    assert sum(len(store.windows(r.run_id)) for r in runs) == 1  # the window in hand was finished, then it stopped
