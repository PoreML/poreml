"""Everything the evaluator computes is kept: descriptor values, their intermediates, the
validation loss — per sample, in a log beside the summary numbers.

The trace layer (`metrics.trace`) is a context the scoring loop opens around `compute`:
descriptors `record` intermediates into it, `compute` records every descriptor value per
side, and `evaluate_loader` turns the lot into one row per scored sample.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from poreml.config import Config
from poreml.losses import loss_function, masked_mse, per_sample_loss
from poreml.metrics import MetricSpec, compute, resolve
from poreml.metrics import trace as tr
from poreml.metrics.descriptors import saturation
from poreml.models import Persistence
from poreml.scoring import evaluate_loader
from poreml.tables import append_rows


def _field(values):
    field = torch.tensor(values, dtype=torch.float32)
    mask = torch.ones_like(field, dtype=torch.bool)
    return field, mask


# ---------------------------------------------------------------------------
# The trace context


def test_record_is_a_no_op_outside_a_trace():
    tr.record("x", torch.zeros(2))  # nothing to record into, nothing raised
    assert tr.active() is None


def test_trace_collects_records_under_the_current_scope():
    with tr.tracing() as trace:
        with tr.scoped("saturation@phi/pred/"):
            tr.record("pore_voxels", torch.tensor([3.0, 4.0]))
        with tr.scoped("saturation@phi/target/"):
            tr.record("pore_voxels", torch.tensor([3.0, 4.0]))
            tr.record("phase_voxels", torch.tensor([1.0, 2.0]))
    assert list(trace.values) == [
        "saturation@phi/pred/pore_voxels",
        "saturation@phi/target/pore_voxels",
        "saturation@phi/target/phase_voxels",
    ]
    assert trace.values["saturation@phi/target/phase_voxels"].tolist() == [1.0, 2.0]
    assert tr.active() is None  # closed


def test_traces_nest_without_leaking():
    with tr.tracing() as outer:
        tr.record("a", torch.ones(1))
        with tr.tracing() as inner:
            tr.record("b", torch.ones(1))
        tr.record("c", torch.ones(1))
    assert list(outer.values) == ["a", "c"]
    assert list(inner.values) == ["b"]


def test_saturation_records_its_intermediates():
    field, mask = _field([[[[1.0, -1.0, 1.0, 1.0]]]])
    mask[0, 0, 0, 3] = False
    with tr.tracing() as trace:
        assert saturation(field, mask).tolist() == pytest.approx([2 / 3])
    assert trace.values["pore_voxels"].tolist() == [3.0]
    assert trace.values["phase_voxels"].tolist() == [2.0]


# ---------------------------------------------------------------------------
# compute: descriptor values per side, intermediates scoped, meshes shared


def test_compute_records_every_descriptor_value_per_side_once():
    specs = resolve(
        [
            MetricSpec(error="mae", channel="phi"),
            MetricSpec(descriptor="saturation", error="abs_err", channel="phi"),
            MetricSpec(descriptor="saturation", error="rel_err", channel="phi"),
            MetricSpec(descriptor="saturation", error="abs_err", channel="phi", phase="w"),
        ],
        ["phi"],
    )
    pred, mask = _field([[[[1.0, -1.0, 1.0, -1.0]]], [[[1.0, 1.0, 1.0, -1.0]]]])
    target, _ = _field([[[[1.0, 1.0, 1.0, -1.0]]], [[[1.0, 1.0, 1.0, -1.0]]]])
    with tr.tracing() as trace:
        out = compute(specs, pred, target, mask)
    assert set(out) == {"mae@phi", "saturation/abs_err@phi", "saturation/rel_err@phi", "saturation/abs_err@phi@w"}
    # The descriptor's value once per side (two errors share it), under the descriptor key.
    assert trace.values["saturation@phi/pred"].tolist() == pytest.approx([0.5, 0.75])
    assert trace.values["saturation@phi/target"].tolist() == pytest.approx([0.75, 0.75])
    assert trace.values["saturation@phi@w/pred"].tolist() == pytest.approx([0.5, 0.25])
    # Intermediates are scoped by descriptor key and side; `voxels` metrics record nothing.
    assert trace.values["saturation@phi/target/pore_voxels"].tolist() == [4.0, 4.0]
    assert trace.values["saturation@phi@w/pred/phase_voxels"].tolist() == [2.0, 1.0]
    assert not [k for k in trace.values if k.startswith("mae")]


def test_curvature_descriptors_share_one_mesh_per_side_and_record_mesh_stats(monkeypatch):
    pytest.importorskip("igl")
    from poreml.metrics import curvature as cv
    from tests.test_curvature import sphere_field

    calls = []
    real = cv.interface_curvature

    def counting(field, solid, **kw):
        calls.append(field.shape)
        return real(field, solid, **kw)

    monkeypatch.setattr(cv, "interface_curvature", counting)
    r = 10
    phi, _ = sphere_field(r)
    target = torch.tensor(phi, dtype=torch.float32)[None, None]
    pred = -target  # a bubble: the same surface, H of the other sign
    mask = torch.ones_like(target, dtype=torch.bool)
    specs = resolve(
        [
            MetricSpec(descriptor="curvature_hist", error="w1", channel="phi", bins=10, range=(0.0, 0.2)),
            MetricSpec(descriptor="curvature_mean_integral", error="rel_err", channel="phi"),
            MetricSpec(descriptor="curvature_mean_integral_norm", error="target", channel="phi"),
            MetricSpec(descriptor="mesh_area", error="rel_err", channel="phi"),
        ],
        ["phi"],
    )
    with tr.tracing() as trace:
        out = compute(specs, pred, target, mask)
    assert len(calls) == 2, "one mesh per side however many curvature descriptors read it"
    assert out["curvature_mean_integral_norm/target@phi"].item() == pytest.approx(1 / r, rel=0.05)
    assert out["mesh_area/rel_err@phi"].item() == pytest.approx(0.0, abs=0.02)
    for side in ("pred", "target"):
        stats = {
            k.rsplit("/", 1)[1]: v for k, v in trace.values.items() if k.startswith(f"curvature_mean_integral_norm@phi/{side}/")
        }
        assert set(stats) >= {
            "n_vertices",
            "n_components_dropped",
            "n_scored",
            "scored_area",
            "area",
            "area_fluid",
            "area_solid",
        }
        assert stats["n_components_dropped"].item() == 0.0
        assert stats["area_solid"].item() == 0.0
        assert stats["scored_area"].item() == pytest.approx(stats["area_fluid"].item())
        assert stats["area"].item() == pytest.approx(4 * np.pi * r**2, rel=0.03)
    # The norm is the traced integral over the traced area.
    integral = trace.values["curvature_mean_integral@phi/target"]
    area = trace.values["curvature_mean_integral_norm@phi/target/scored_area"]
    assert (integral / area).item() == pytest.approx(out["curvature_mean_integral_norm/target@phi"].item(), rel=1e-5)
    assert trace.values["curvature_mean_integral@phi/pred"].item() == pytest.approx(-integral.item(), rel=1e-5)


def test_outside_compute_the_mesh_cache_is_off(monkeypatch):
    pytest.importorskip("igl")
    from poreml.metrics import curvature as cv
    from poreml.metrics.descriptors import curvature_mean, curvature_mean_integral
    from tests.test_curvature import sphere_field

    calls = []
    real = cv.interface_curvature
    monkeypatch.setattr(cv, "interface_curvature", lambda f, s, **kw: (calls.append(1), real(f, s, **kw))[1])
    phi, _ = sphere_field(8)
    field = torch.tensor(phi, dtype=torch.float32)[None, None]
    mask = torch.ones_like(field, dtype=torch.bool)
    curvature_mean(field, mask)
    curvature_mean_integral(field, mask)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Losses per sample


def test_per_sample_loss_matches_the_batch_loss_sample_by_sample():
    pred = torch.randn(3, 1, 4, 4, 4)
    target = torch.randn(3, 1, 4, 4, 4)
    mask = torch.rand(3, 1, 4, 4, 4) > 0.3
    per = per_sample_loss(masked_mse, pred, target, mask)
    assert per.shape == (3,)
    for i in range(3):
        assert per[i].item() == pytest.approx(masked_mse(pred[i : i + 1], target[i : i + 1], mask[i : i + 1]).item())


def test_loss_function_resolves_the_config_names():
    from poreml.losses import masked_h1

    assert loss_function("mse") is masked_mse
    assert loss_function("h1") is masked_h1
    with pytest.raises(ValueError):
        loss_function("l1")


# ---------------------------------------------------------------------------
# evaluate_loader: samples and loss


def _loader_and_metrics(fake_root):
    from poreml.data import WindowDataset, discover

    runs = list(discover(fake_root, "drainage").values())
    ds = WindowDataset(runs, history=1)
    loader = DataLoader(ds, batch_size=2, collate_fn=ds.collate)
    metrics = resolve([MetricSpec(error="mae"), MetricSpec(descriptor="saturation", error="abs_err")], ("phi",))
    return ds, loader, metrics


def test_evaluate_loader_keeps_one_row_per_sample_with_everything(fake_root):
    ds, loader, metrics = _loader_and_metrics(fake_root)
    ev = evaluate_loader(Persistence(1), loader, metrics, "cpu", [r.run_id for r in ds.runs], loss_fn=masked_mse)
    assert len(ev.samples) == len(ds)
    row = ev.samples[0]
    assert row["run"] == ds.runs[0].run_id and row["t"] == 0  # t: the last input frame's index
    assert set(row) >= {
        "mae",
        "saturation/abs_err",
        "loss",
        "saturation/pred",
        "saturation/target",
        "saturation/target/pore_voxels",
    }
    # The loss is aggregated like a metric: per run, then over runs.
    assert "loss" in ev.summary and "loss" in ev.per_run and ev.counts["loss"]["n_samples"] == len(ds)
    per_run_loss = [np.mean([r["loss"] for r in ev.samples if r["run"] == run]) for run in ev.per_run["loss"]]
    assert ev.summary["loss"] == pytest.approx(float(np.mean(per_run_loss)))
    # Sample order is the loader's; every row carries the same columns.
    assert all(set(r) == set(row) for r in ev.samples)


def test_evaluate_loader_without_a_loss_reports_none(fake_root):
    ds, loader, metrics = _loader_and_metrics(fake_root)
    ev = evaluate_loader(Persistence(1), loader, metrics, "cpu", [r.run_id for r in ds.runs])
    assert "loss" not in ev.summary
    assert "loss" not in ev.samples[0]


# ---------------------------------------------------------------------------
# tables.append_rows: a growing CSV whose header is fixed by its first rows


def test_append_rows_keeps_the_header_and_fills_missing_columns(tmp_path):
    path = tmp_path / "t.csv"
    append_rows(path, [{"epoch": 0, "a": 1.0, "h": [0.5, 0.5]}, {"epoch": 0, "a": float("nan"), "h": [1.0, 0.0]}])
    append_rows(path, [{"epoch": 1, "a": 2.0}])  # a later row lacking a column: blank, not a crash
    rows = list(csv.DictReader(path.open()))
    assert list(rows[0]) == ["epoch", "a", "h[0]", "h[1]"]
    assert [r["a"] for r in rows] == ["1.0", "", "2.0"]
    assert rows[2]["h[0]"] == ""
    append_rows(path, [])  # nothing to add, nothing written
    assert len(list(csv.DictReader(path.open()))) == 3


# ---------------------------------------------------------------------------
# End to end: evaluate, train, rollout


def _cfg(fake_root, tmp_path, **train):
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
            "train": {"epochs": 2, "batch_size": 1, "device": "cpu", "max_batches": 2, **train},
            "out_dir": str(tmp_path / "runs"),
        }
    )


def test_evaluate_writes_samples_csv_and_the_loss(fake_root, tmp_path):
    from poreml.evaluate import evaluate
    from poreml.train import train

    cfg = _cfg(fake_root, tmp_path)
    run_dir = train(cfg)
    results = evaluate(cfg, ckpt=run_dir / "ckpts" / "best.pt", split="test")
    payload = json.loads((run_dir / "ckpts" / "results_test.json").read_text())
    assert "loss" in payload["metrics"] and payload["metrics"]["loss"] == results.summary["loss"]
    rows = list(csv.DictReader((run_dir / "ckpts" / "samples_test.csv").open()))
    assert len(rows) == payload["n_samples"]
    assert {"run", "t", "loss", "mae", "saturation/pred"} <= set(rows[0])
    assert {r["run"] for r in rows} == {"tiny_0002_M10_theta140"}


def test_training_logs_the_validation_loss_and_the_validation_samples(fake_root, tmp_path):
    from poreml.train import train

    cfg = _cfg(fake_root, tmp_path)
    run_dir = train(cfg)
    epochs = list(csv.DictReader((run_dir / "metrics.csv").open()))
    assert len(epochs) == 2 and "val_loss" in epochs[0] and float(epochs[0]["val_loss"]) >= 0
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert "val_loss" in meta["progress"]["last"]
    samples = list(csv.DictReader((run_dir / "val_samples.csv").open()))
    assert {r["epoch"] for r in samples} == {"0", "1"}
    assert {"run", "t", "loss", "mae", "saturation/target"} <= set(samples[0])
    per_epoch = [float(r["loss"]) for r in samples if r["epoch"] == "0"]
    assert np.mean(per_epoch) == pytest.approx(float(epochs[0]["val_loss"]))
    # The loss is reporting only: selection still follows metrics[0].
    assert "val_loss" not in meta["progress"]["best"]


def test_rollout_carries_descriptor_values_and_intermediates_per_step(fake_root, tmp_path):
    from poreml.evaluate import rollout_split
    from poreml.train import train

    cfg = _cfg(fake_root, tmp_path, epochs=1)
    run_dir = train(cfg)
    summary = rollout_split(cfg, ckpt=run_dir / "ckpts" / "best.pt", split="test", eval_stride=2)
    payload = json.loads((run_dir / "ckpts" / "rollout_test.json").read_text())
    for key in ("saturation/pred", "saturation/target", "saturation/target/pore_voxels"):
        assert key in summary.curves and key in payload["curves"] and key in payload["summary"]
        curve = payload["curves"][key]["tiny_0002_M10_theta140"]
        assert len(curve) == summary.runs["tiny_0002_M10_theta140"]["horizon"]
        assert curve[0] is None and curve[1] is not None  # unscored steps stay null, like the metrics
    metric_names = [m["name"] for m in payload["metrics_spec"]]
    assert list(payload["summary"])[: len(metric_names)] == metric_names  # metrics first, then the extras


def test_every_shipped_config_scores_the_curvature_integrals():
    """Every config that meshes for `curvature_hist` also reports M2 = ∫H dS and the
    area-weighted ⟨H⟩, as values and relative errors, on the same phase — eval-only, so
    the periodic evaluation and `poreml eval` carry them and selection does not."""
    from poreml.study import SchemeConfig

    paths = sorted(Path("configs").rglob("*.yaml"))
    for path in paths:
        if path.parts[1] in (
            "scale",
            "shift",
            "scale_push",
            "shift_push",
        ):  # study trees: only the scheme carries a metric list
            if path.name != "scheme.yaml":
                continue
            specs = SchemeConfig.model_validate(yaml.safe_load(path.read_text())).metrics
        else:
            specs = Config.from_yaml(path).metrics or []
        hists = [m for m in specs if m.descriptor == "curvature_hist"]
        if not hists:
            continue
        phase = hists[0].phase
        for descriptor in ("curvature_mean_integral", "curvature_mean_integral_norm"):
            entries = {m.error: m for m in specs if m.descriptor == descriptor}
            assert {"rel_err", "target", "pred"} <= set(entries), f"{path}: {descriptor} is not scored as value and error"
            for m in entries.values():
                assert m.eval_only and m.phase == phase and m.channel == "phi", f"{path}: {m}"
