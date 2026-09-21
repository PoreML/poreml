"""Push-forward fine-tuning (`train.push_forward`, `train.init_from`, `pushforward.py`).

The trick (Brandstetter et al. 2022; BubbleML, Hassan et al. 2023): on a pushed batch the
model first steps on its own output `steps` times without gradients, then takes one graded
step scored against the true frame that far ahead. Batches that are not pushed get Gaussian
noise on their history channels instead. The default path (`push_forward: null`) must stay
byte-identical to what every benchmark run trained with.
"""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError
from torch import nn

from poreml import pushforward
from poreml.config import Config, PushForwardConfig, TrainConfig
from poreml.data import WindowDataset, discover
from poreml.points import Points, PointWindowDataset, gather
from poreml.registry import MODELS
from poreml.train import train


@pytest.fixture
def runs(fake_root):
    return list(discover(fake_root, "drainage").values())


def rows(run_dir: Path) -> list[dict]:
    return list(csv.DictReader((run_dir / "metrics.csv").open()))


def only_run(out_dir: Path) -> Path:
    (run_dir,) = Path(out_dir).iterdir()
    return run_dir


# --- config ---------------------------------------------------------------------------------


def test_push_forward_defaults_follow_the_protocol():
    p = PushForwardConfig()
    assert (p.steps, p.prob_start, p.prob_end, p.noise) == (1, 0.5, 1.0, 0.01)
    assert TrainConfig().push_forward is None and TrainConfig().init_from is None


def test_push_forward_config_is_validated():
    with pytest.raises(ValidationError):
        PushForwardConfig(steps=0)
    with pytest.raises(ValidationError):
        PushForwardConfig(prob_start=1.5)
    with pytest.raises(ValidationError):
        PushForwardConfig(noise=-0.1)
    with pytest.raises(ValidationError):
        TrainConfig(push_forward={"stepz": 2})


# --- dataset: targets `future` frames ahead ---------------------------------------------------


def test_window_dataset_future_targets_stack_the_next_frames(runs):
    one = WindowDataset(runs[:1], history=1, roi=None)
    two = WindowDataset(runs[:1], history=1, roi=None, future=2)
    assert len(two) == len(one) - 1  # the last window has no frame two steps ahead
    inputs1, target1, mask1, meta1 = one[0]
    inputs2, target2, mask2, meta2 = two[0]
    assert torch.equal(inputs1, inputs2) and torch.equal(mask1, mask2) and meta1 == meta2
    assert target2.shape == (2, *target1.shape)
    assert torch.equal(target2[0], target1)
    assert torch.equal(target2[1], torch.from_numpy(two.frame(0, 2)))


def test_window_dataset_future_one_is_the_historical_shape(runs):
    ds = WindowDataset(runs[:1], history=1)
    _, target, _, _ = ds[0]
    assert target.ndim == 4  # (F, D, H, W): no leading future axis


def test_window_dataset_rejects_a_run_too_short_for_the_future(runs):
    with pytest.raises(ValueError, match="too few"):
        WindowDataset(runs[:1], history=1, future=6)  # the fixture runs have 6 steps


def test_point_dataset_future_targets_are_per_point(runs):
    one = PointWindowDataset(runs[:1], history=1)
    two = PointWindowDataset(runs[:1], history=1, future=2)
    assert len(two) == len(one) - 1
    pts, target, mask, _ = two[0]
    n = len(pts.index)
    assert target.shape == (n, 2, 1) and mask.shape == (n, 1)
    assert torch.equal(target[:, 0], one[0][1])
    assert np.allclose(target[:, 1].numpy(), gather(two.frame(0, 2), pts.index.numpy()))


def test_task_threads_future_to_the_dataset(runs):
    from poreml.config import TaskConfig
    from poreml.tasks import build_task

    task = build_task(TaskConfig(name="next_frame"))
    assert task.dataset(runs, future=3).future == 3
    assert task.dataset(runs).future == 1


# --- schedule ---------------------------------------------------------------------------------


def test_probability_ramps_linearly_from_start_to_end():
    cfg = PushForwardConfig(prob_start=0.5, prob_end=1.0)
    assert pushforward.probability(cfg, 0, 100) == 0.5
    assert pushforward.probability(cfg, 50, 100) == pytest.approx(0.75)
    assert pushforward.probability(cfg, 100, 100) == 1.0
    assert pushforward.probability(cfg, 150, 100) == 1.0  # never beyond the end


def test_n_pushes_is_a_pure_function_of_seed_and_step():
    cfg = PushForwardConfig(steps=2, prob_start=0.5, prob_end=0.5)
    draws = [pushforward.n_pushes(cfg, seed=0, step=s, total_steps=100) for s in range(200)]
    assert set(draws) == {0, 2}
    assert 60 < draws.count(2) < 140  # about half
    assert draws == [pushforward.n_pushes(cfg, seed=0, step=s, total_steps=100) for s in range(200)]
    assert draws != [pushforward.n_pushes(cfg, seed=1, step=s, total_steps=100) for s in range(200)]


def test_n_pushes_honours_the_extremes():
    never = PushForwardConfig(prob_start=0.0, prob_end=0.0)
    always = PushForwardConfig(steps=3, prob_start=1.0, prob_end=1.0)
    assert all(pushforward.n_pushes(never, 0, s, 10) == 0 for s in range(20))
    assert all(pushforward.n_pushes(always, 0, s, 10) == 3 for s in range(20))


# --- perturb / advance / unroll on voxels -----------------------------------------------------


def _voxel_batch(runs, history=1, batch=2, future=1):
    ds = WindowDataset(runs[:1], history=history, roi=None, future=future)
    items = [ds[i] for i in range(batch)]
    inputs = torch.stack([it[0] for it in items])
    target = torch.stack([it[1] for it in items])
    mask = torch.stack([it[2] for it in items])
    return ds, inputs, target, mask


def test_perturb_touches_only_fluid_history_voxels(runs):
    ds, inputs, _, _ = _voxel_batch(runs)
    n_static = 1  # solid mask; the fixture task has no conditions
    torch.manual_seed(0)
    noisy = pushforward.perturb(inputs, 0.01, n_static)
    solid = inputs[:, :1] == 1
    assert noisy.shape == inputs.shape
    assert torch.equal(noisy[:, :n_static], inputs[:, :n_static])  # static channels untouched
    assert torch.equal(
        noisy[:, n_static:][solid.expand_as(noisy[:, n_static:])], inputs[:, n_static:][solid.expand_as(noisy[:, n_static:])]
    )
    fluid = ~solid
    diff = (noisy[:, n_static:] - inputs[:, n_static:])[fluid.expand_as(noisy[:, n_static:])]
    assert diff.abs().max() > 0 and diff.std() == pytest.approx(0.01, rel=0.3)
    assert torch.equal(pushforward.perturb(inputs, 0.0, n_static), inputs)  # std 0: a no-op


class _Shift(nn.Module):
    """Predicts the newest frame plus one, everywhere (solid included, to test the fill)."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        feats = x.feats if isinstance(x, Points) else x
        return feats[:, -self.out_channels :] + 1.0


def test_advance_slides_the_window_and_fills_solid(runs):
    ds, inputs, _, _ = _voxel_batch(runs, history=2)
    n_static, n_fields = 1, 1
    pred = _Shift(3)(inputs)
    nxt = pushforward.advance(ds, inputs, pred, n_static, n_fields)
    assert nxt.shape == inputs.shape
    assert torch.equal(nxt[:, :1], inputs[:, :1])  # static: the solid mask
    assert torch.equal(nxt[:, 1:2], inputs[:, 2:3])  # the old newest frame is now the older one
    assert torch.equal(nxt[:, 2:], ds.decode(inputs, pred))  # newest = the prediction, solid at fill
    solid = inputs[:, :1] == 1
    assert (nxt[:, 2:][solid] == ds.frame_fill[0]).all()


def test_unroll_takes_n_ungraded_steps_and_keeps_the_graph_off(runs):
    ds, inputs, _, _ = _voxel_batch(runs, history=1)
    model = _Shift(2)
    autocast = torch.autocast("cpu", enabled=False)
    out = pushforward.unroll(model, ds, inputs, 3, 1, 1, autocast)
    assert model.calls == 3 and not out.requires_grad
    fluid = inputs[:, :1] == 0
    assert torch.allclose(out[:, 1:][fluid], inputs[:, 1:][fluid] + 3.0)
    assert torch.equal(pushforward.unroll(model, ds, inputs, 0, 1, 1, autocast), inputs)


# --- the same on points ----------------------------------------------------------------------


def test_perturb_and_advance_on_points(runs):
    ds = PointWindowDataset(runs[:1], history=2, geometry_radii=(1,), future=1)
    pts, _, _, _ = ds[0]
    n_static, n_fields = 1, 1
    torch.manual_seed(0)
    noisy = pushforward.perturb(pts, 0.01, n_static)
    assert isinstance(noisy, Points) and torch.equal(noisy.feats[:, :1], pts.feats[:, :1])
    assert (noisy.feats[:, 1:] - pts.feats[:, 1:]).std() == pytest.approx(0.01, rel=0.3)
    assert torch.equal(noisy.index, pts.index) and torch.equal(noisy.pos, pts.pos)

    pred = _Shift(3)(pts)
    nxt = pushforward.advance(ds, pts, pred, n_static, n_fields)
    assert nxt.feats.shape == pts.feats.shape
    assert torch.equal(nxt.feats[:, 0], pts.feats[:, 0])
    assert torch.equal(nxt.feats[:, 1], pts.feats[:, 2])
    assert torch.equal(nxt.feats[:, 2:], pred)


# --- training end to end ----------------------------------------------------------------------


@pytest.fixture
def push_cfg(fake_root, tmp_path, interruptible):
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"train": ["tiny_0000_M1_theta120"], "val": ["tiny_0001_M1_theta130"]}))
    return Config.model_validate(
        {
            "name": "push",
            "data": {"root": str(fake_root), "campaign": "drainage", "split": str(split)},
            "task": {"name": "next_frame", "history": 1, "params": {"roi": None}},
            "model": {"name": "_test_interruptible", "params": {"base": 4, "depth": 1}},
            "train": {
                "epochs": 3,
                "batch_size": 1,
                "lr": 1e-3,
                "device": "cpu",
                "checkpoint_every": 0,
                "push_forward": {"steps": 1, "prob_start": 0.5, "prob_end": 1.0, "noise": 0.01},
            },
            "out_dir": str(tmp_path / "runs"),
        }
    )


def test_push_forward_training_runs_and_reports_the_pushed_fraction(push_cfg, interruptible):
    run_dir = train(push_cfg)
    table = rows(run_dir)
    assert len(table) == 3
    fracs = [float(r["push_frac"]) for r in table]
    assert all(0.0 <= f <= 1.0 for f in fracs)
    assert sum(fracs) > 0  # some batches were pushed
    # a pushed batch costs one extra forward: 4 windows per epoch, 3 epochs
    assert interruptible.forwards == 12 + round(sum(f * 4 for f in fracs))
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "finished" and meta["config"]["train"]["push_forward"]["steps"] == 1


def test_push_forward_does_not_change_the_default_path(push_cfg, tmp_path):
    """`push_forward: null` trains exactly as before: no extra random draws, no new column."""
    plain = push_cfg.model_copy(update={"train": push_cfg.train.model_copy(update={"push_forward": None})})
    a = rows(train(plain.model_copy(update={"out_dir": tmp_path / "a"})))
    b = rows(train(plain.model_copy(update={"out_dir": tmp_path / "b"})))
    assert "push_frac" not in a[0]
    for ra, rb in zip(a, b, strict=True):
        assert ra["train_loss"] == rb["train_loss"]


def test_resumed_push_forward_training_is_bit_equal(push_cfg, interruptible, tmp_path):
    reference = rows(train(push_cfg.model_copy(update={"out_dir": tmp_path / "ref"})))
    interruptible.forwards, interruptible.crash_at = 0, 6  # inside epoch 1
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(push_cfg)
    run_dir = only_run(push_cfg.out_dir)
    interruptible.crash_at = None
    assert train(push_cfg, resume=run_dir) == run_dir
    resumed = rows(run_dir)
    assert [r["epoch"] for r in resumed] == [r["epoch"] for r in reference]
    for ra, rb in zip(reference, resumed, strict=True):
        for key in ra.keys() - {"epoch_seconds", "peak_memory_mb"}:
            assert float(ra[key]) == pytest.approx(float(rb[key]), rel=1e-6), key


def test_init_from_loads_the_checkpoint_weights_and_is_recorded(push_cfg, tmp_path):
    base = train(push_cfg.model_copy(update={"out_dir": tmp_path / "base"}))
    ckpt = base / "ckpts" / "best.pt"
    # One batch at a vanishing lr: last.pt is then the weights the run started from.
    still = {"epochs": 1, "max_batches": 1, "lr": 1e-12}
    tuned = train(push_cfg.model_copy(update={"train": push_cfg.train.model_copy(update={**still, "init_from": ckpt})}))
    fresh = train(push_cfg.model_copy(update={"train": push_cfg.train.model_copy(update=still), "out_dir": tmp_path / "fresh"}))
    meta = json.loads((tuned / "run_meta.json").read_text())
    assert meta["model"]["init_from"]["path"] == str(ckpt)
    assert meta["model"]["init_from"]["epoch"] == torch.load(ckpt, weights_only=False)["epoch"]
    assert len(meta["model"]["init_from"]["sha256"]) == 64
    assert "init_from" not in json.loads((fresh / "run_meta.json").read_text())["model"]
    w0 = torch.load(ckpt, weights_only=False)["model"]
    w_tuned = torch.load(tuned / "ckpts" / "last.pt", weights_only=False)["model"]
    w_fresh = torch.load(fresh / "ckpts" / "last.pt", weights_only=False)["model"]
    assert all(torch.allclose(w_tuned[k], w0[k], atol=1e-6) for k in w0)
    assert not all(torch.allclose(w_fresh[k], w0[k], atol=1e-6) for k in w0)  # a fresh init is not the checkpoint


def test_init_from_must_exist_before_a_run_directory_is_made(push_cfg, tmp_path):
    cfg = push_cfg.model_copy(update={"train": push_cfg.train.model_copy(update={"init_from": tmp_path / "nope.pt"})})
    with pytest.raises(FileNotFoundError):
        train(cfg)
    assert not Path(cfg.out_dir).exists()


class _PointShift(nn.Module):
    representation = "points"

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: Points):
        return x.feats[:, -self.out_channels :] + self.bias


def test_push_forward_trains_a_point_model(push_cfg):
    MODELS.register("_test_point_shift")(_PointShift)
    try:
        cfg = push_cfg.model_copy(
            update={"model": push_cfg.model.model_copy(update={"name": "_test_point_shift", "params": {}})}
        )
        run_dir = train(cfg)
    finally:
        MODELS._items.pop("_test_point_shift")
    assert json.loads((run_dir / "run_meta.json").read_text())["status"] == "finished"
    assert "push_frac" in rows(run_dir)[0]


# --- case/train_push/make_configs.py ---------------------------------------------------------


spec = importlib.util.spec_from_file_location("make_configs", Path("case/train_push/make_configs.py"))
make_configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_configs)


def _run(out_dir: Path, name: str, status: str, best: bool = True) -> Path:
    run = out_dir / name
    (run / "ckpts").mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"status": status}))
    if best:
        (run / "ckpts" / "best.pt").write_bytes(b"x")
    return run


def test_finished_best_picks_the_newest_finished_run_with_a_checkpoint(tmp_path):
    assert make_configs.finished_best(tmp_path / "missing") is None
    _run(tmp_path, "a", "failed")
    assert make_configs.finished_best(tmp_path) is None
    old = _run(tmp_path, "b", "finished")
    assert make_configs.finished_best(tmp_path) == old / "ckpts" / "best.pt"
    _run(tmp_path, "c", "running")
    assert make_configs.finished_best(tmp_path) == old / "ckpts" / "best.pt"


@pytest.mark.parametrize(
    "path",
    [
        "configs/drainage/unet_gen.yaml",
        "configs/drainage/fno_all.yaml",
        "configs/gdl/p3d_gen.yaml",
        "configs/underfill/abupt_all.yaml",
    ],
)
def test_derive_halves_the_epochs_and_switches_on_the_fine_tune(path):
    base = Config.from_yaml(path)
    best = Path("case/train") / "x" / "ckpts" / "best.pt"
    cfg = make_configs.derive(base, best)
    epochs = base.train.epochs // 2
    assert cfg.name == base.name + "_push"
    assert cfg.train.epochs == epochs
    # the periodic eval is the rollout alone (2026-09-10): every epoch, except for AB-UPT whose
    # rollout is the expensive part (40-90 min per eval at 128, ~4 h on underfill)
    assert cfg.train.eval.every == (max(1, epochs // 5) if base.model.name == "abupt" else 1)
    assert cfg.train.push_forward == PushForwardConfig(steps=1, prob_start=0.5, prob_end=1.0, noise=0.01)
    assert cfg.train.init_from == best
    assert str(cfg.out_dir) == str(base.out_dir).replace("case/train/", "case/train_push/")
    if base.model.name == "fno3d":
        assert cfg.train.lr_step_size == max(1, epochs // 5)
    if base.model.name == "p3d":
        assert cfg.train.lr_step_size == round(epochs * 2 / 3)
    # everything else is the base recipe
    assert (cfg.task, cfg.data, cfg.metrics, cfg.rollout) == (base.task, base.data, base.metrics, base.rollout)
    assert cfg.seed == base.seed
    if base.model.name in make_configs.NO_CHECKPOINT:  # transolver: same model, activation checkpointing off
        assert cfg.model.params == {**base.model.params, "checkpoint": False}
        assert cfg.model.model_copy(update={"params": base.model.params}) == base.model
    else:
        assert cfg.model == base.model
    assert cfg.train.lr == pytest.approx(base.train.lr / 3, rel=1e-2) and cfg.train.lr_schedule == base.train.lr_schedule
    assert cfg.train.batch_size == base.train.batch_size and cfg.train.weight_decay == base.train.weight_decay


def test_shipped_push_configs_match_their_base(tmp_path):
    """Every generated config under configs/push is what make_configs derives from its base today.

    The checkpoint each one starts from is only checked where the base trainings are on disk:
    `case/train/` is generated output, so a fresh clone has the configs but none of the weights.
    """
    for path in sorted(Path("configs/push").rglob("*.yaml")):
        base = Config.from_yaml(Path("configs") / path.relative_to("configs/push"))
        cfg = Config.from_yaml(path)
        assert cfg == make_configs.derive(base, cfg.train.init_from), path
        run_dir = cfg.train.init_from.parents[1]
        if run_dir.is_dir():
            assert cfg.train.init_from.is_file(), f"{path}: {run_dir} exists but holds no {cfg.train.init_from.name}"
