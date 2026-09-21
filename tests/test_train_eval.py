import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pydantic
import pytest
import torch
import yaml
from torch import nn

from poreml.config import Config, EvalConfig, ModelConfig, TaskConfig
from poreml.evaluate import evaluate, render_rollout
from poreml.metrics import MetricSpec
from poreml.registry import MODELS
from poreml.train import Evaluation, build_run_dir, evaluate_loader, load_runs, masked_h1, masked_mse, train


def strict_loads(text: str):
    """Parse JSON the way jq, JSON.parse, Go and R do: bare NaN/Infinity are not JSON."""

    def reject(token: str) -> float:
        raise AssertionError(f"non-JSON token {token!r} was written to an artifact")

    return json.loads(text, parse_constant=reject)


class _Worsening(nn.Module):
    """Persistence that drifts further from the truth with every batch it trains on."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drift = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.drift += 0.25
        return x[:, -self.out_channels :] + self.drift


class _Counting(nn.Module):
    """Persistence that records how many batches it was trained on."""

    train_batches = 0

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            type(self).train_batches += 1
        return x[:, -self.out_channels :]


@pytest.fixture
def stub_models():
    """Register the test-only models and take them out again afterwards.

    The registry is global and other tests assert on what it holds, so these must not
    outlive the test that asked for them. Registry has no public removal — a benchmark
    should not be able to unregister a model at runtime — hence the private pop here.
    """
    MODELS.register("_test_worsening")(_Worsening)
    MODELS.register("_test_counting")(_Counting)
    _Counting.train_batches = 0
    yield
    MODELS._items.pop("_test_worsening")
    MODELS._items.pop("_test_counting")


@pytest.fixture
def cfg(fake_root, tmp_path):
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {
                "train": ["tiny_0000_M1_theta120"],
                "val": ["tiny_0001_M1_theta130"],
                "test": ["tiny_0002_M10_theta140"],
            }
        )
    )
    return Config.model_validate(
        {
            "name": "smoke",
            "data": {"root": str(fake_root), "campaign": "drainage", "split": str(split)},
            "task": {"name": "next_frame", "history": 1},
            "model": {"name": "persistence"},
            "train": {"epochs": 1, "batch_size": 1, "device": "cpu", "max_batches": 2},
            # the fixture runs hold 6 frames: a 4-step horizon is reachable, so the rollout's
            # at-horizon values (what the final stage selects on) are finite
            "rollout": {"horizon": 4},
            "out_dir": str(tmp_path / "runs"),
        }
    )


def test_empty_metrics_list_is_rejected(fake_root, tmp_path):
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {
                "train": ["tiny_0000_M1_theta120"],
                "val": ["tiny_0001_M1_theta130"],
                "test": ["tiny_0002_M10_theta140"],
            }
        )
    )

    with pytest.raises(pydantic.ValidationError):
        Config.model_validate(
            {
                "name": "smoke",
                "data": {"root": str(fake_root), "campaign": "drainage", "split": str(split)},
                "task": {"name": "next_frame", "history": 1},
                "model": {"name": "persistence"},
                "train": {"epochs": 1, "batch_size": 1, "device": "cpu", "max_batches": 2},
                "metrics": [],
                "out_dir": str(tmp_path / "runs"),
            }
        )


def test_masked_mse_ignores_masked_out_voxels():
    pred = torch.tensor([[1.0, 100.0]])
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])

    assert masked_mse(pred, target, mask).item() == pytest.approx(1.0)


def test_masked_mse_is_zero_for_a_perfect_prediction():
    x = torch.tensor([[0.3, -0.7]])
    mask = torch.ones_like(x, dtype=torch.bool)

    assert masked_mse(x, x, mask).item() == pytest.approx(0.0)


def test_masked_mse_on_an_all_solid_batch_is_zero_and_still_differentiable():
    # A hard torch.zeros(()) here detaches the graph, so backward() fails for any model
    # with parameters the moment one batch is all rock.
    weight = torch.zeros(1, 2, requires_grad=True)
    pred = weight + 1.0
    target = torch.zeros(1, 2)
    mask = torch.zeros(1, 2, dtype=torch.bool)

    loss = masked_mse(pred, target, mask)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert loss.requires_grad
    assert torch.equal(weight.grad, torch.zeros(1, 2))


def test_masked_h1_is_zero_for_a_perfect_prediction():
    torch.manual_seed(0)
    x = torch.rand(1, 2, 4, 4, 4)
    mask = torch.ones(1, 1, 4, 4, 4, dtype=torch.bool)

    assert masked_h1(x, x, mask).item() == pytest.approx(0.0, abs=1e-4)


def test_masked_h1_ignores_masked_out_voxels():
    # Solid voxels must not leak into the loss through the finite-difference terms
    # either: a pore voxel's derivative at the wall may only ever see the target's fill.
    torch.manual_seed(0)
    target = torch.rand(1, 2, 4, 4, 4)
    mask = torch.rand(1, 1, 4, 4, 4) > 0.5
    pred = target + 0.1 * torch.rand_like(target)
    vandalised = torch.where(mask.expand_as(pred), pred, torch.full_like(pred, 99.0))

    assert masked_h1(vandalised, target, mask).item() == pytest.approx(masked_h1(pred, target, mask).item())


def test_masked_h1_penalises_a_gradient_mismatch_that_mse_cannot_see():
    # Two predictions with identical masked MSE: a constant offset, and the same offset
    # with alternating sign along the flow axis. Only the second has spurious gradients.
    target = torch.linspace(0.5, 1.5, 8).view(1, 1, 1, 1, 8).expand(1, 1, 8, 8, 8)
    mask = torch.ones(1, 1, 8, 8, 8, dtype=torch.bool)
    smooth = target + 0.1
    checker = target + 0.1 * (-1.0) ** torch.arange(8).view(1, 1, 1, 1, 8)

    assert masked_mse(smooth, target, mask).item() == pytest.approx(masked_mse(checker, target, mask).item())
    assert masked_h1(checker, target, mask).item() > 2 * masked_h1(smooth, target, mask).item()


def test_masked_h1_is_one_norm_over_all_channels_not_a_per_channel_sum():
    # Growing an error-free channel's target enlarges the shared denominator, so the
    # relative loss falls. Upstream's per-channel reduction would not move: channel 0's
    # ratio is unchanged and channel 1 contributes zero either way.
    torch.manual_seed(0)
    target = torch.rand(1, 2, 4, 4, 4)
    pred = target.clone()
    pred[:, 0] += 0.1
    mask = torch.ones(1, 1, 4, 4, 4, dtype=torch.bool)
    grown_target = target.clone()
    grown_target[:, 1] *= 10
    grown_pred = pred.clone()
    grown_pred[:, 1] = grown_target[:, 1]

    assert masked_h1(grown_pred, grown_target, mask).item() < 0.9 * masked_h1(pred, target, mask).item()


def test_masked_h1_is_relative_so_scaling_both_sides_changes_nothing():
    torch.manual_seed(0)
    target = torch.rand(1, 2, 4, 4, 4) + 0.5
    pred = target + 0.1 * torch.rand_like(target)
    mask = torch.ones(1, 1, 4, 4, 4, dtype=torch.bool)

    assert masked_h1(3 * pred, 3 * target, mask).item() == pytest.approx(masked_h1(pred, target, mask).item(), rel=1e-4)


def test_masked_h1_on_an_all_solid_batch_is_zero_and_still_differentiable():
    # Same contract as masked_mse: an all-rock batch must yield a connected zero, not NaN
    # from a square root at exactly zero.
    weight = torch.zeros(1, 1, 4, 4, 4, requires_grad=True)
    pred = weight + 1.0
    target = torch.zeros(1, 1, 4, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)

    loss = masked_h1(pred, target, mask)
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=1e-4)
    assert loss.requires_grad
    assert torch.equal(weight.grad, torch.zeros(1, 1, 4, 4, 4))


def test_build_run_dir_is_named_and_created(cfg):
    path = build_run_dir(cfg, now="20260814-120000")

    assert path.is_dir()
    assert path.name == "smoke_20260814-120000"
    assert (path / "ckpts").is_dir()


def test_two_runs_with_the_same_stamp_get_separate_directories(cfg):
    # A scripted sweep starts two runs of one name inside a second; sharing a directory
    # would interleave their metrics.jsonl lines and overwrite their checkpoints.
    first = build_run_dir(cfg, now="20260814-120000")
    second = build_run_dir(cfg, now="20260814-120000")

    assert first != second
    assert second.name == "smoke_20260814-120000-1"
    assert (second / "ckpts").is_dir()
    assert build_run_dir(cfg, now="20260814-120000").name == "smoke_20260814-120000-2"


def test_load_runs_returns_all_three_sections(cfg):
    runs = load_runs(cfg)

    assert [r.run_id for r in runs["train"]] == ["tiny_0000_M1_theta120"]
    assert [r.run_id for r in runs["val"]] == ["tiny_0001_M1_theta130"]
    assert [r.run_id for r in runs["test"]] == ["tiny_0002_M10_theta140"]


def test_train_writes_all_expected_artifacts(cfg):
    run_dir = train(cfg)

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "run_meta.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "train_log.csv").is_file()
    assert (run_dir / "ckpts" / "last.pt").is_file()
    assert (run_dir / "ckpts" / "best.pt").is_file()


def test_val_stride_subsamples_the_per_epoch_validation(cfg):
    # The fixture's val run has 6 frames -> windows t=0..4. val_stride 5 keeps only t=0,
    # whose persistence error differs from the mean over all five windows, so the change
    # must show up in the reported validation value — proof the stride reaches the loader.
    import csv

    def val_mae(val_stride):
        run_dir = train(cfg.updated(train={"epochs": 1, "batch_size": 1, "device": "cpu", "val_stride": val_stride}))
        row = list(csv.DictReader((run_dir / "metrics.csv").open()))[-1]
        return float(next(v for k, v in row.items() if k.startswith("mae")))

    assert val_mae(5) != pytest.approx(val_mae(1))


def test_periodic_eval_threads_rollout_stride_to_the_rollout(cfg):
    # rollout_stride 3 on a 4-step rollout (6-frame run, t0=1) scores steps 3 and 4 only;
    # the written summary carries null at the unscored steps and a number at the horizon.
    c = cfg.updated(
        train={
            "epochs": 1,
            "batch_size": 1,
            "device": "cpu",
            "eval": {"every": 1, "stride": 1, "rollout": True, "render": False, "rollout_stride": 3},
        }
    )

    run_dir = train(c)

    payload = json.loads((run_dir / "eval" / "epoch_000" / "rollout_val.json").read_text())
    curve = next(v for k, v in payload["summary"].items() if k.startswith("mae"))
    assert curve[0] is None and curve[1] is None
    assert isinstance(curve[2], float) and isinstance(curve[3], float)


def test_written_config_reloads_identically(cfg):
    run_dir = train(cfg)

    assert Config.from_yaml(run_dir / "config.yaml") == cfg


def test_metrics_jsonl_has_one_line_per_epoch_with_task_metrics(cfg):
    cfg.train.epochs = 2

    run_dir = train(cfg)

    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["epoch"] == 0
    assert {"mae", "iou", "saturation/abs_err"} <= set(lines[0])
    assert "train_loss" in lines[0]


def test_training_a_parameter_free_model_does_not_crash(cfg):
    # persistence has no parameters; the loop must skip the optimizer rather than fail.
    run_dir = train(cfg)

    assert (run_dir / "ckpts" / "last.pt").is_file()


def test_checkpoint_carries_config_and_epoch(cfg):
    run_dir = train(cfg)

    ckpt = torch.load(run_dir / "ckpts" / "best.pt", weights_only=True)

    assert ckpt["config"]["name"] == "persistence"
    assert isinstance(ckpt["epoch"], int)


def test_evaluate_writes_results_json_with_task_metrics(cfg):
    run_dir = train(cfg)
    ckpt = run_dir / "ckpts" / "best.pt"

    results = evaluate(cfg, ckpt=ckpt)

    assert isinstance(results, Evaluation)
    assert {"mae", "iou", "saturation/abs_err"} <= set(results.summary)
    written = json.loads((ckpt.parent / "results_test.json").read_text())
    assert written["metrics"]["mae"] == pytest.approx(results.summary["mae"])
    assert written["split"] == "test"


def test_results_are_named_per_split_so_a_second_scoring_does_not_clobber_the_first(cfg):
    run_dir = train(cfg)
    ckpt = run_dir / "ckpts" / "best.pt"

    evaluate(cfg, ckpt=ckpt, split="test")
    evaluate(cfg, ckpt=ckpt, split="val")

    assert json.loads((ckpt.parent / "results_test.json").read_text())["split"] == "test"
    assert json.loads((ckpt.parent / "results_val.json").read_text())["split"] == "val"


def test_results_land_beside_a_foreign_checkpoint_not_two_levels_up(cfg, tmp_path):
    run_dir = train(cfg)
    foreign = tmp_path / "models"
    foreign.mkdir()
    (foreign / "best.pt").write_bytes((run_dir / "ckpts" / "best.pt").read_bytes())

    evaluate(cfg, ckpt=foreign / "best.pt")

    assert (foreign / "results_test.json").is_file()
    assert not (tmp_path / "results_test.json").exists()


def test_out_dir_overrides_where_results_are_written(cfg, tmp_path):
    run_dir = train(cfg)
    elsewhere = tmp_path / "scores" / "nested"

    evaluate(cfg, ckpt=run_dir / "ckpts" / "best.pt", out_dir=elsewhere)

    assert (elsewhere / "results_test.json").is_file()


def test_evaluate_honours_an_explicit_metric_list(cfg):
    cfg.metrics = [MetricSpec(error="mae")]
    run_dir = train(cfg)

    results = evaluate(cfg, ckpt=run_dir / "ckpts" / "best.pt")

    assert set(results.summary) == {"mae", "loss"}  # the configured metric plus the training loss


def test_a_distribution_metric_survives_the_loader_and_the_results_file_as_a_list(cfg):
    # `hist/pred` is per-sample (B, K): the mean over samples is a K-vector, which has to
    # travel through evaluate_loader, json_safe and the results file as a JSON array.
    cfg.metrics = [
        MetricSpec(error="mae"),
        MetricSpec(descriptor="hist", error="pred", bins=4, range=(-1.0, 1.0)),
        MetricSpec(descriptor="hist", error="w1", bins=4, range=(-1.0, 1.0)),
    ]
    run_dir = train(cfg)
    ckpt = run_dir / "ckpts" / "best.pt"

    results = evaluate(cfg, ckpt=ckpt).summary

    bins = results["hist/pred"]
    assert isinstance(bins, list)
    assert [type(v) for v in bins] == [float] * 4
    assert sum(bins) == pytest.approx(1.0)  # every sample's histogram is normalised
    assert isinstance(results["hist/w1"], float) and math.isfinite(results["hist/w1"])

    written = strict_loads((ckpt.parent / "results_test.json").read_text())
    assert written["metrics"]["hist/pred"] == pytest.approx(bins)
    assert '"hist/pred": [' in (ckpt.parent / "results_test.json").read_text()
    logged = strict_loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert isinstance(logged["hist/pred"], list)

    # ... and as a long CSV beside the JSON, bins as rows with their edges, a model column.
    rows = list(csv.DictReader((ckpt.parent / "distributions_test.csv").open()))
    assert [r["metric"] for r in rows] == ["hist/pred"] * 4
    assert [float(r["lo"]) for r in rows] == pytest.approx([-1.0, -0.5, 0.0, 0.5])
    assert [float(r["model"]) for r in rows] == pytest.approx(bins)
    assert "baseline" not in rows[0]


def test_eval_only_metrics_are_skipped_in_validation_but_scored_by_evaluate(cfg):
    cfg.metrics = [MetricSpec(error="mae"), MetricSpec(error="iou", eval_only=True)]
    run_dir = train(cfg)
    logged = strict_loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert "iou" not in logged
    results = evaluate(cfg, ckpt=run_dir / "ckpts" / "best.pt").summary
    assert {"mae", "iou"} <= set(results)


def test_persistence_is_wrong_by_exactly_one_slab(cfg):
    # Persistence predicts frame t against the target at t + 1. The fixture invades one
    # slab per step, so the two frames differ only on the pore voxels of slab t + 1, and
    # there by |dphi| = |1 - (-1)| = 2.
    (test_run,) = load_runs(cfg)["test"]
    with h5py.File(test_run.h5_path, "r") as f:
        pore = np.asarray(f["rock"][:]) == 0
        n_steps = len(f["steps"])
    # The task scores the rock ROI, which in the fixture is slabs 1.. — slab 0 is the inlet.
    pore = pore[:, :, 1:]
    n_pore = pore.sum()
    windows = range(n_steps - 1)  # t = 0 .. n_steps - 2; the differing slab t + 1 is ROI slab t
    expected_mae = float(np.mean([2.0 * pore[:, :, t].sum() / n_pore for t in windows]))
    # Predicted invaded voxels are a strict subset of the target's, so intersection over
    # union is n_pore(ROI slabs < t) / n_pore(ROI slabs <= t).
    expected_iou = float(np.mean([pore[:, :, :t].sum() / pore[:, :, : t + 1].sum() for t in windows]))

    run_dir = train(cfg)
    results = evaluate(cfg, ckpt=run_dir / "ckpts" / "best.pt").summary

    assert results["mae"] == pytest.approx(expected_mae, rel=1e-6)
    assert results["iou"] == pytest.approx(expected_iou, rel=1e-6)


def test_an_unknown_metric_name_fails_before_any_training(cfg):
    cfg.metrics = [MetricSpec(error="ioU")]

    with pytest.raises(KeyError, match="ioU"):
        train(cfg)

    assert list(Path(cfg.out_dir).glob("*")) == []  # not even a run directory was made


def test_evaluate_rejects_an_unknown_metric_name(cfg):
    cfg.metrics = [MetricSpec(error="ioU")]

    with pytest.raises(KeyError, match="ioU"):
        evaluate(cfg)


def test_artifacts_are_strict_json_when_nothing_can_be_scored(cfg, tmp_path):
    # Empty val and test sections make every metric NaN, which json.dumps would otherwise
    # write as a bare NaN token that no parser outside Python accepts.
    split = tmp_path / "empty_split.yaml"
    split.write_text(yaml.safe_dump({"train": ["tiny_0000_M1_theta120"], "val": [], "test": []}))
    cfg.data.split = split

    run_dir = train(cfg)
    ckpt = run_dir / "ckpts" / "best.pt"
    evaluate(cfg, ckpt=ckpt)

    record = strict_loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    payload = strict_loads((ckpt.parent / "results_test.json").read_text())
    assert record["mae"] is None
    assert payload["metrics"] == {"mae": None, "iou": None, "saturation/abs_err": None, "loss": None}


def test_a_worse_later_epoch_does_not_overwrite_best(cfg, stub_models):
    cfg.model = ModelConfig(name="_test_worsening")
    cfg.train.epochs = 2

    run_dir = train(cfg)

    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[1]["mae"] > lines[0]["mae"]  # epoch 1 really is the worse one
    assert torch.load(run_dir / "ckpts" / "last.pt", weights_only=True)["epoch"] == 1
    assert torch.load(run_dir / "ckpts" / "best.pt", weights_only=True)["epoch"] == 0


def test_max_batches_caps_the_batches_trained_on(cfg, stub_models):
    cfg.model = ModelConfig(name="_test_counting")
    cfg.train.max_batches = 2

    train(cfg)

    assert _Counting.train_batches == 2

    _Counting.train_batches = 0
    cfg.train.max_batches = None

    train(cfg)

    assert _Counting.train_batches == 5  # the train run holds 5 windows at batch_size 1


def test_reported_metrics_do_not_depend_on_batch_size(cfg):
    run_dir = train(cfg)
    ckpt = run_dir / "ckpts" / "best.pt"

    cfg.train.batch_size = 1
    at_one = evaluate(cfg, ckpt=ckpt).summary
    cfg.train.batch_size = 4
    at_four = evaluate(cfg, ckpt=ckpt).summary

    for name in at_one:
        assert at_four[name] == pytest.approx(at_one[name], rel=1e-6), name


def test_results_json_echoes_the_metric_specs_and_sample_count(cfg):
    run_dir = train(cfg)
    ckpt = run_dir / "ckpts" / "best.pt"

    evaluate(cfg, ckpt=ckpt)

    written = json.loads((ckpt.parent / "results_test.json").read_text())
    assert [s["name"] for s in written["metrics_spec"]] == ["mae", "iou", "saturation/abs_err"]
    assert written["metrics_spec"][2]["descriptor"] == "saturation"
    assert written["n_samples"] == 5  # the fixture run has 6 steps -> 5 windows at history=1


class _NaNSecond(nn.Module):
    """Persistence whose second sample in every batch is non-finite: a failed prediction."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = x[:, -1:].clone()
        if pred.shape[0] > 1:
            pred[1] = float("nan")
        return pred


@pytest.fixture
def uneven_runs(tmp_path):
    """Two finished runs of different lengths: 6 steps (5 windows) and 4 steps (3 windows)."""
    from tests.fixtures import write_fake_run

    write_fake_run(tmp_path, "drainage", "long_0000_M1_theta120", n_steps=6)
    write_fake_run(tmp_path, "drainage", "short_0001_M1_theta120", n_steps=4)
    return load_runs_from(tmp_path)


def load_runs_from(root):
    from poreml.data import discover

    return list(discover(root, "drainage").values())


def test_evaluate_loader_averages_per_run_first_so_every_trajectory_counts_once(uneven_runs):
    from torch.utils.data import DataLoader

    from poreml.data import WindowDataset
    from poreml.metrics import MetricSpec, compute, resolve
    from poreml.models import Persistence

    ds = WindowDataset(uneven_runs, history=1)
    metrics = resolve([MetricSpec(error="mae")], ("phi",))
    model = Persistence(2)

    result = evaluate_loader(model, DataLoader(ds, batch_size=2), metrics, "cpu", [r.run_id for r in uneven_runs])

    # Recompute per sample by hand and aggregate both ways.
    per_sample = {}
    for i in range(len(ds)):
        inputs, target, mask, meta = ds[i]
        value = compute(metrics, model(inputs[None]), target[None], mask[None])["mae"].item()
        per_sample.setdefault(meta["run"], []).append(value)
    per_run = {uneven_runs[k].run_id: float(np.mean(v)) for k, v in per_sample.items()}
    macro = float(np.mean(list(per_run.values())))
    micro = float(np.mean([v for vs in per_sample.values() for v in vs]))

    assert result.per_run["mae"] == pytest.approx(per_run)
    assert result.summary["mae"] == pytest.approx(macro)
    assert macro != pytest.approx(micro)  # the two runs differ in length, so the choice is visible
    assert result.counts["mae"] == {"n_runs": 2, "n_samples": 8, "n_invalid": 0}


def test_evaluate_loader_counts_failed_samples_instead_of_dropping_them(uneven_runs):
    from torch.utils.data import DataLoader

    from poreml.data import WindowDataset
    from poreml.metrics import MetricSpec, resolve

    ds = WindowDataset(uneven_runs, history=1)
    metrics = resolve([MetricSpec(error="mae")], ("phi",))

    result = evaluate_loader(_NaNSecond(2), DataLoader(ds, batch_size=2), metrics, "cpu", [r.run_id for r in uneven_runs])

    # Batches of 2 over 8 windows: the second sample of each full batch fails -> 4 failures.
    assert result.counts["mae"]["n_invalid"] == 4
    assert result.counts["mae"]["n_samples"] == 8
    assert math.isfinite(result.summary["mae"])  # the surviving samples still yield a number


def test_evaluate_loader_on_an_empty_loader_is_nan_with_zero_counts():
    from torch.utils.data import DataLoader, TensorDataset

    from poreml.metrics import MetricSpec, resolve
    from poreml.models import Persistence

    metrics = resolve([MetricSpec(error="mae")], ("phi",))
    empty = DataLoader(TensorDataset(torch.zeros(0, 2, 2, 2, 2)), batch_size=1)

    result = evaluate_loader(Persistence(2), empty, metrics, "cpu", [])

    assert math.isnan(result.summary["mae"])
    assert result.per_run["mae"] == {}
    assert result.counts["mae"] == {"n_runs": 0, "n_samples": 0, "n_invalid": 0}


def test_results_json_carries_per_run_values_counts_runs_and_provenance(cfg):
    run_dir = train(cfg)
    ckpt = run_dir / "ckpts" / "best.pt"

    evaluate(cfg, ckpt=ckpt)

    written = strict_loads((ckpt.parent / "results_test.json").read_text())
    assert written["n_runs"] == 1
    assert written["per_run"]["mae"] == {"tiny_0002_M10_theta140": pytest.approx(written["metrics"]["mae"])}
    assert written["counts"]["mae"] == {"n_runs": 1, "n_samples": 5, "n_invalid": 0}
    assert written["runs"][0]["run_id"] == "tiny_0002_M10_theta140"
    assert written["runs"][0]["params"]["M"] == pytest.approx(10.0)
    assert written["runs"][0]["geometry"]["family"] == "tiny"
    prov = written["provenance"]
    assert prov["poreml_version"] and prov["torch"] and prov["metrics_version"]
    assert prov["split"]["sha256"] and prov["split"]["path"].endswith("split.yaml")
    assert Path(prov["data_root"]).is_absolute()


def test_train_writes_run_meta_with_provenance_and_epoch_timing(cfg):
    run_dir = train(cfg)

    meta = strict_loads((run_dir / "run_meta.json").read_text())
    prov = meta["provenance"]
    assert prov["seed"] == 0 and prov["device"] == "cpu"
    logged = strict_loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert logged["epoch_seconds"] >= 0.0
    assert "peak_memory_mb" in logged  # None on CPU, a number on CUDA


def test_run_meta_documents_config_data_model_and_progress(cfg):
    import csv

    cfg.train.epochs = 2
    run_dir = train(cfg)

    meta = strict_loads((run_dir / "run_meta.json").read_text())
    assert meta["name"] == "smoke" and meta["status"] == "finished"
    assert meta["config"]["model"]["name"] == "persistence" and meta["config"]["task"]["history"] == 1
    assert meta["data"]["split"]["path"].endswith("split.yaml") and meta["data"]["split"]["sha256"]
    assert [r["run_id"] for r in meta["data"]["runs"]["train"]] == ["tiny_0000_M1_theta120"]
    assert meta["data"]["runs"]["train"][0]["params"]["M"] == pytest.approx(1.0)
    assert meta["data"]["n_windows"] == {"train": 5, "val": 1}  # val_stride 8 (the default) keeps t=0 of 5 windows
    assert meta["model"] == {
        "name": "persistence",
        "params": {},
        "in_channels": 2,
        "out_channels": 1,
        "n_parameters": 0,
        "precision": "tf32",
    }
    assert meta["metrics"][0]["name"] == "mae" and meta["primary_metric"] == "mae"
    assert meta["progress"]["epochs_done"] == 2 and meta["progress"]["epochs_total"] == 2
    assert meta["progress"]["best"]["epoch"] in (0, 1) and math.isfinite(meta["progress"]["best"]["mae"])
    assert meta["run"]["start_time"] <= meta["run"]["end_time"]

    # The flat CSV view of the epoch log: one row per epoch, header first, plottable as is.
    with (run_dir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["0", "1"]
    assert {"train_loss", "epoch_seconds", "peak_memory_mb", "mae", "iou", "saturation/abs_err"} <= set(rows[0])
    assert float(rows[0]["mae"]) == pytest.approx(strict_loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])["mae"])


def test_run_meta_is_written_before_training_and_marks_failure(cfg, stub_models):
    class _Boom(nn.Module):
        def __init__(self, in_channels: int, out_channels: int = 1) -> None:
            super().__init__()

        def forward(self, x):
            raise RuntimeError("boom")

    MODELS.register("_test_boom")(_Boom)
    try:
        cfg.model = ModelConfig(name="_test_boom")
        with pytest.raises(RuntimeError, match="boom"):
            train(cfg)
    finally:
        MODELS._items.pop("_test_boom")

    (run_dir,) = Path(cfg.out_dir).iterdir()
    meta = strict_loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "failed" and "boom" in meta["run"]["error"]
    assert meta["progress"]["epochs_done"] == 0


def test_train_log_csv_records_batches(cfg):
    import csv

    cfg.train.max_batches = None  # 5 windows at batch_size 1 -> 5 batches; log_every rows
    cfg.train.log_every = 2
    run_dir = train(cfg)

    with (run_dir / "train_log.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["batch"] for r in rows] == ["2", "4", "5"]  # every log_every batches plus the epoch's last batch
    assert {"epoch", "step", "loss", "elapsed_seconds"} <= set(rows[0])


def test_distribution_metrics_are_flattened_into_indexed_csv_columns(cfg):
    import csv

    cfg.metrics = [MetricSpec(error="mae"), MetricSpec(descriptor="hist", error="pred", bins=3, range=(-1.0, 1.0))]
    run_dir = train(cfg)

    with (run_dir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {"hist/pred[0]", "hist/pred[1]", "hist/pred[2]"} <= set(rows[0])


def test_a_split_naming_an_unfinished_run_is_refused_before_training(cfg, fake_root):
    from tests.fixtures import write_fake_run

    write_fake_run(fake_root, "drainage", "live_0009_M1_theta120", status="running")
    payload = yaml.safe_load(cfg.data.split.read_text())
    payload["test"].append("live_0009_M1_theta120")
    cfg.data.split.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="live_0009_M1_theta120"):
        train(cfg)

    cfg.data.require_finished = False
    train(cfg)  # explicitly opting in works


class _Zero(nn.Module):
    """Predicts all-wetting everywhere: strictly worse than persistence on the fixture."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[:, -self.out_channels :], -1.0)


def test_evaluate_scores_the_model_alone(cfg):
    MODELS.register("_test_zero")(_Zero)
    try:
        cfg.model = ModelConfig(name="_test_zero")
        run_dir = train(cfg)
        ckpt = run_dir / "ckpts" / "best.pt"
        results = evaluate(cfg, ckpt=ckpt)
    finally:
        MODELS._items.pop("_test_zero")

    assert not hasattr(results, "baseline")
    written = strict_loads((ckpt.parent / "results_test.json").read_text())
    assert "baseline" not in written
    assert written["counts"]["mae"]["n_samples"] == 5


def test_baseline_is_not_a_config_option():
    with pytest.raises(pydantic.ValidationError, match="baseline"):
        Config.model_validate(
            {
                "name": "x",
                "data": {"root": ".", "campaign": "drainage", "split": "s.yaml"},
                "task": {"name": "next_frame"},
                "model": {"name": "persistence"},
                "baseline": "persistence",
            }
        )


def test_evaluate_stride_subsamples_the_windows_and_records_it(cfg, tmp_path):
    full = evaluate(cfg, split="test", out_dir=tmp_path / "full")
    strided = evaluate(cfg, split="test", out_dir=tmp_path / "strided", stride=2)

    assert strided.counts["mae"]["n_samples"] < full.counts["mae"]["n_samples"]
    payload = strict_loads((tmp_path / "strided" / "results_test.json").read_text())
    assert payload["stride"] == 2 and payload["n_samples"] == strided.counts["mae"]["n_samples"]


def test_render_rollout_checks_phi_channel_before_rolling_out(cfg, monkeypatch, tmp_path):
    # A field-less-than-phi task must fail immediately, before the (expensive) autonomous
    # rollout runs at all — not deep inside frame assembly after paying for it.
    from poreml import evaluate as evaluate_mod

    cfg = Config.model_validate(
        {
            **cfg.model_dump(mode="json"),
            "task": {**cfg.task.model_dump(mode="json"), "params": {"fields": [{"name": "p"}]}},
        }
    )

    def _boom(*args, **kwargs):
        raise AssertionError("rollout must not run before the phi-channel check")

    monkeypatch.setattr(evaluate_mod, "rollout", _boom)

    with pytest.raises(ValueError, match="phi"):
        render_rollout(cfg, ckpt=None, run_id="tiny_0002_M10_theta140", out_dir=tmp_path / "render", split="test")


def test_scoring_module_owns_the_evaluator_and_train_still_exports_it():
    from poreml import scoring
    from poreml import train as train_mod

    assert train_mod.evaluate_loader is scoring.evaluate_loader
    assert train_mod.Evaluation is scoring.Evaluation
    assert train_mod.load_runs is scoring.load_runs


def test_eval_config_defaults_and_strictness():
    from poreml.config import EvalConfig, TrainConfig

    assert TrainConfig().eval is None
    ev = EvalConfig(every=2)
    assert (ev.stride, ev.rollout, ev.render) == (1, True, True)
    with pytest.raises(pydantic.ValidationError):
        EvalConfig(every=0)
    with pytest.raises(pydantic.ValidationError):
        EvalConfig(every=1, split="test")


def test_periodic_eval_writes_artifacts_csv_and_progress(cfg, stub_models):
    cfg = cfg.model_copy(
        update={
            "model": ModelConfig(name="_test_counting"),
            "train": cfg.train.model_copy(update={"epochs": 3, "eval": EvalConfig(every=2, stride=2, render=False)}),
        }
    )

    run_dir = train(cfg)

    # every=2 over 3 epochs: after epoch 1 (2nd) and after the last epoch (2)
    assert sorted(p.name for p in (run_dir / "eval").glob("epoch_*")) == ["epoch_001", "epoch_002"]
    for name in ("epoch_001", "epoch_002"):
        # the periodic eval is the rollout alone: no val pass, no render (2026-09-10)
        assert (run_dir / "eval" / name / "rollout_val.json").exists()
        assert not (run_dir / "eval" / name / "results_val.json").exists()
        assert not (run_dir / "eval" / name / "frames").exists()
    with (run_dir / "eval" / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert {"mae", "rollout/mae"} <= set(rows[0])
    assert not any(c.startswith(("baseline/", "rollout/baseline/")) for c in rows[0])
    assert (run_dir / "eval" / "curves.svg").exists()
    # the `<m>` columns are the epoch's own selection-pass metrics
    with (run_dir / "metrics.csv").open() as f:
        per_epoch = {r["epoch"]: r for r in csv.DictReader(f)}
    assert rows[0]["mae"] == per_epoch["1"]["mae"] and rows[1]["mae"] == per_epoch["2"]["mae"]
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert [e["epoch"] for e in meta["progress"]["evals"]] == [1, 2]
    assert meta["progress"]["last_eval"]["epoch"] == 2
    # the final stage: the rollout-best checkpoint marked and fully evaluated under final/
    chosen = meta["progress"]["best_rollout"]
    assert chosen["epoch"] in (1, 2) and chosen["candidates"] == 2 and "rollout/mae" in chosen
    marked = run_dir / "ckpts" / "best_rollout.pt"
    assert marked.read_bytes() == (run_dir / "ckpts" / f"epoch_{chosen['epoch']:03d}.pt").read_bytes()
    final = run_dir / "final"
    assert meta["progress"]["final"]["dir"] == str(final) and meta["progress"]["final"]["epoch"] == chosen["epoch"]
    for name in ("results_val.json", "samples_val.csv", "rollout_val.json", "metrics.json"):
        assert (final / name).exists(), name
    payload = strict_loads((final / "results_val.json").read_text())
    assert payload["stride"] == 2 and payload["checkpoint"].endswith("best_rollout.pt")
    summary = strict_loads((final / "metrics.json").read_text())
    assert summary["epoch"] == chosen["epoch"] and "mae" in summary["one_step"] and "mae" in summary["rollout_at_horizon"]
    assert not (final / "frames").exists()  # render=False


def test_eval_only_metrics_are_left_to_the_final_stage(cfg):
    """The periodic rollout scores the validation metrics only (no meshing during training);
    the marked checkpoint's final val pass and rollout score everything."""
    cfg = Config.model_validate(
        {
            **cfg.model_dump(mode="json"),
            "metrics": [{"error": "mae"}, {"descriptor": "saturation", "error": "abs_err", "eval_only": True}],
            "train": {
                **cfg.train.model_dump(mode="json"),
                "epochs": 1,
                "eval": {"every": 1, "rollout": True, "render": False},
            },
        }
    )

    run_dir = train(cfg)

    with (run_dir / "metrics.csv").open() as f:
        assert "saturation/abs_err" not in csv.DictReader(f).fieldnames
    with (run_dir / "eval" / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert "rollout/mae" in rows[0] and "saturation/abs_err" not in rows[0] and "rollout/saturation/abs_err" not in rows[0]
    periodic = strict_loads((run_dir / "eval" / "epoch_000" / "rollout_val.json").read_text())
    assert [m["name"] for m in periodic["metrics_spec"]] == ["mae"]
    final = strict_loads((run_dir / "final" / "results_val.json").read_text())
    assert "saturation/abs_err" in final["metrics"]
    final_roll = strict_loads((run_dir / "final" / "rollout_val.json").read_text())
    assert "saturation/abs_err" in final_roll["at_horizon"]


def test_a_checkpoint_is_kept_for_every_epoch(cfg):
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"epochs": 3})})

    run_dir = train(cfg)

    assert sorted(p.name for p in (run_dir / "ckpts").glob("epoch_*.pt")) == ["epoch_000.pt", "epoch_001.pt", "epoch_002.pt"]
    assert torch.load(run_dir / "ckpts" / "epoch_001.pt", weights_only=False)["epoch"] == 1
    # the same payload as last.pt (torch.save's zip container stamps the bytes, so compare the contents)
    kept = torch.load(run_dir / "ckpts" / "epoch_002.pt", weights_only=False)
    last = torch.load(run_dir / "ckpts" / "last.pt", weights_only=False)
    assert kept["epoch"] == last["epoch"] == 2 and kept["metrics"] == last["metrics"]
    assert all(torch.equal(kept["model"][k], last["model"][k]) for k in last["model"])


def test_without_a_periodic_rollout_nothing_is_selected(cfg):
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"eval": EvalConfig(every=1, rollout=False, render=False)})}
    )

    run_dir = train(cfg)

    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "finished"
    assert meta["progress"]["best_rollout"] is None and meta["progress"]["final"] is None
    assert not (run_dir / "ckpts" / "best_rollout.pt").exists() and not (run_dir / "final").exists()
    with (run_dir / "eval" / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert "mae" in rows[0] and not any(c.startswith("rollout/") for c in rows[0])


def test_periodic_eval_failure_does_not_kill_training(cfg, monkeypatch):
    # Periodic eval is reporting, not training: a bug inside it (a metric, the rollout,
    # the renderer) must not fail a multi-hour training run. `_periodic_eval` imports
    # `evaluate` from `.evaluate` at call time, so patching `poreml.evaluate.evaluate`
    # is what actually reaches it.
    import poreml.evaluate as evaluate_mod

    monkeypatch.setattr(evaluate_mod, "rollout_split", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"eval": EvalConfig(every=1, rollout=True, render=False)})}
    )

    run_dir = train(cfg)  # must return normally, not raise

    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "finished"
    assert "boom" in meta["progress"]["evals"][0]["error"]
    assert "boom" in meta["progress"]["last_eval"]["error"]
    assert meta["progress"]["best_rollout"] is None  # no rollout row to select on


def test_final_stage_failure_does_not_fail_the_run(cfg, monkeypatch):
    """The final stage is reporting like the periodic eval: the run is trained and stays
    `finished`; the error is recorded and `poreml finalise` reruns the stage."""
    import poreml.evaluate as evaluate_mod

    monkeypatch.setattr(evaluate_mod, "evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mesh boom")))
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"eval": EvalConfig(every=1, rollout=True, render=False)})}
    )

    run_dir = train(cfg)

    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "finished"
    assert meta["progress"]["best_rollout"]["epoch"] == 0  # selection and the mark happen before the evaluation
    assert (run_dir / "ckpts" / "best_rollout.pt").exists()
    assert "mesh boom" in meta["progress"]["final"]["error"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="train() moves batches to the forced cuda:0; needs a driver")
def test_periodic_eval_forks_rng_only_on_the_current_cuda_device(cfg, monkeypatch):
    # torch's fork_rng default (devices=None) captures *every visible* CUDA device and
    # warns when there is more than one; on a CUDA run this must be pinned to just the
    # process's current device instead of taking that default.
    import poreml.evaluate as evaluate_mod
    import poreml.train as train_mod
    from poreml.evaluate import Evaluation

    captured = {}
    real_fork_rng = torch.random.fork_rng

    def _spying_fork_rng(devices=None, **kwargs):
        captured["devices"] = devices
        return real_fork_rng(devices=[], **kwargs)  # actually fork nothing: no real CUDA device here

    monkeypatch.setattr(train_mod, "resolve_device", lambda spec: "cuda:0")
    monkeypatch.setattr(train_mod.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(train_mod.torch.random, "fork_rng", _spying_fork_rng)
    from poreml.rollout import RolloutSummary

    monkeypatch.setattr(evaluate_mod, "rollout_split", lambda *a, **k: RolloutSummary(horizon=1))
    monkeypatch.setattr(evaluate_mod, "evaluate", lambda *a, **k: Evaluation())
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"eval": EvalConfig(every=1, rollout=True, render=False)})}
    )

    train(cfg)  # the periodic-eval failure wrapper covers whatever plot_eval_curves does with an empty row

    assert captured["devices"] == [0]


def test_periodic_eval_render_true_fails_fast_when_viz_is_unavailable(cfg, monkeypatch):
    # `render: True` (the default) must not let a config with a missing viz group train
    # for N epochs and only then die inside render_rollout at the end of epoch N, after
    # hours of GPU time and a full rollout: the check has to happen before anything else.
    from poreml import viz

    monkeypatch.setattr(viz, "require_viz", lambda: (_ for _ in ()).throw(ImportError("no viz group")))
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"eval": EvalConfig(every=1, render=True)})})

    with pytest.raises(ImportError):
        train(cfg)

    assert list(Path(cfg.out_dir).glob("*")) == []  # failed before any run directory was made


def test_final_stage_render_writes_frames_and_a_gif(cfg):
    pytest.importorskip("pyvista")
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"epochs": 1, "eval": EvalConfig(every=1, rollout=True, render=True)})}
    )

    run_dir = train(cfg)

    assert not (run_dir / "eval" / "epoch_000" / "frames").exists()  # never during training
    frames = list((run_dir / "final" / "frames").glob("frame_*.png"))
    assert frames
    assert (run_dir / "final" / "rollout_tiny_0001_M1_theta130.gif").exists()
    assert (run_dir / "progress.svg").exists()


class _PointDouble(nn.Module):
    """A point model for the loop tests: returns twice the most recent point features."""

    representation = "points"

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, points):
        return points.feats[:, -self.out_channels :] * self.scale


@pytest.fixture
def point_model():
    MODELS.register("_test_point_double")(_PointDouble)
    yield
    MODELS._items.pop("_test_point_double")


def test_train_and_evaluate_run_on_the_point_stream(cfg, point_model):
    from poreml.evaluate import rollout_split

    cfg = cfg.model_copy(
        update={
            "model": ModelConfig(name="_test_point_double"),
            "task": cfg.task.model_copy(update={"params": {"points": {"train_points": 6}}}),
            "train": cfg.train.model_copy(update={"grad_clip": 1.0, "lr_schedule": "onecycle", "epochs": 2}),
        }
    )
    run_dir = train(cfg)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["task"]["representation"] == "points"
    assert meta["task"]["in_channels"] == 3 + 1
    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert len(rows) == 2 and float(rows[0]["lr"]) != float(rows[1]["lr"])  # onecycle moves the lr
    results = evaluate(cfg, ckpt=run_dir / "ckpts" / "best.pt", split="test")
    assert set(results.summary) == {"mae", "iou", "saturation/abs_err", "loss"}
    assert math.isfinite(results.summary["mae"])
    summary = rollout_split(cfg, ckpt=run_dir / "ckpts" / "best.pt", split="test")
    assert math.isfinite(summary.summary["mae"][0])  # step 1 is scored on every run


def test_steplr_decays_the_lr_every_step_size_epochs(cfg, point_model):
    cfg = cfg.model_copy(
        update={
            "model": ModelConfig(name="_test_point_double"),
            "train": cfg.train.model_copy(update={"epochs": 3, "lr_schedule": "steplr", "lr_step_size": 2, "lr_gamma": 0.5}),
        }
    )

    run_dir = train(cfg)

    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    lrs = [float(row["lr"]) for row in rows]
    # metrics.csv reports the lr each epoch actually trained at: two epochs at the
    # configured lr, then the decay fires.
    assert lrs == pytest.approx([cfg.train.lr, cfg.train.lr, cfg.train.lr * 0.5])


def test_train_loss_h1_trains_on_a_different_loss_than_mse(cfg):
    # persistence has no parameters, so train_loss is exactly the loss function's value
    # on the same windows: a differing number proves the knob reaches the batch loop.
    losses = {}
    for name in ("mse", "h1"):
        c = cfg.model_copy(update={"train": cfg.train.model_copy(update={"loss": name})})
        run_dir = train(c)
        rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
        losses[name] = float(rows[0]["train_loss"])

    assert losses["mse"] != losses["h1"]


def test_h1_loss_on_the_point_stream_fails_before_a_run_dir_exists(cfg, point_model):
    cfg = cfg.model_copy(
        update={
            "model": ModelConfig(name="_test_point_double"),
            "train": cfg.train.model_copy(update={"loss": "h1"}),
        }
    )

    with pytest.raises(ValueError, match="point"):
        train(cfg)

    assert not list(Path(cfg.out_dir).glob("*"))


def test_train_and_evaluate_with_conditions(cfg):
    cfg = cfg.model_copy(
        update={
            "task": TaskConfig(
                name="next_frame",
                params={"conditions": [{"name": "M", "transform": "log10"}, {"name": "theta", "scale": 180.0}]},
            )
        }
    )
    run_dir = train(cfg)
    results = evaluate(cfg, ckpt=run_dir / "ckpts" / "best.pt")
    assert math.isfinite(results.summary["mae"])


def test_results_record_the_checkpoint_hash(cfg, tmp_path):
    from poreml.provenance import sha256_of

    run_dir = train(cfg)
    ckpt = run_dir / "ckpts" / "best.pt"

    evaluate(cfg, ckpt=ckpt)
    written = json.loads((ckpt.parent / "results_test.json").read_text())
    assert written["checkpoint_sha256"] == sha256_of(ckpt)

    bare = tmp_path / "bare"
    evaluate(cfg, ckpt=None, out_dir=bare)  # no checkpoint: hash is null, not an error
    assert json.loads((bare / "results_test.json").read_text())["checkpoint_sha256"] is None


def test_training_under_bf16_keeps_fp32_weights_and_records_the_precision(cfg):
    """`model.precision: bf16` autocasts the forward pass only: the loss is computed in fp32,
    checkpoints and optimiser states stay fp32, run_meta.json says what the run trained in."""
    bf16 = cfg.model_copy(
        update={
            "task": TaskConfig(name="next_frame", history=1, params={"roi": None}),
            "model": ModelConfig(name="unet3d", params={"base": 4, "depth": 1}, precision="bf16"),
        }
    )
    run_dir = train(bf16)

    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert math.isfinite(float(rows[-1]["train_loss"]))
    state = torch.load(run_dir / "ckpts" / "last.pt", weights_only=True)["model"]
    assert all(t.dtype == torch.float32 for t in state.values() if t.is_floating_point())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["model"]["precision"] == "bf16"
    assert json.loads((run_dir / "run_meta.json").read_text())["config"]["model"]["precision"] == "bf16"
