import math

import pytest
import torch

from poreml.metrics.compose import ResolvedMetric, compute, is_better, resolve, validation_metrics
from poreml.metrics.spec import MetricSpec
from poreml.registry import DESCRIPTORS

CH = ("phi",)


def test_resolve_returns_one_resolved_metric_per_spec_with_keys():
    specs = [MetricSpec(error="mae"), MetricSpec(descriptor="saturation", error="abs_err")]
    out = resolve(specs, CH)
    assert [m.name for m in out] == ["mae", "saturation/abs_err"]
    assert all(isinstance(m, ResolvedMetric) for m in out)
    assert out[1].params == {"phase": "nw"}
    assert out[0].params == {}


def test_resolve_maps_channel_names_and_indices():
    assert resolve([MetricSpec(error="mae", channel="uy")], ("ux", "uy", "uz"))[0].channel == 1
    assert resolve([MetricSpec(error="mae", channel=2)], ("ux", "uy", "uz"))[0].channel == 2
    assert resolve([MetricSpec(error="mae")], ("ux", "uy", "uz"))[0].channel is None


def test_resolve_rejects_unknown_names_channels_and_mismatches():
    with pytest.raises(KeyError, match="ioU"):
        resolve([MetricSpec(error="ioU")], CH)
    with pytest.raises(KeyError, match="volumen"):
        resolve([MetricSpec(descriptor="volumen", error="abs_err")], CH)
    with pytest.raises(ValueError, match="channel"):
        resolve([MetricSpec(error="mae", channel="ux")], CH)
    with pytest.raises(ValueError, match="channel"):
        resolve([MetricSpec(error="mae", channel=1)], CH)
    with pytest.raises(ValueError, match="accepts"):
        resolve([MetricSpec(descriptor="volume", error="mae")], CH)  # voxel error on a scalar
    with pytest.raises(ValueError, match="accepts"):
        resolve([MetricSpec(descriptor="hist", error="abs_err", bins=4, range=(0.0, 1.0))], CH)
    with pytest.raises(ValueError, match="accepts"):
        resolve([MetricSpec(error="pred")], CH)  # raw pass-through of ragged voxels


def test_resolve_rejects_params_the_descriptor_does_not_take():
    with pytest.raises(ValueError, match="phase"):
        resolve([MetricSpec(descriptor="area_interface", error="abs_err", phase="w")], CH)
    with pytest.raises(ValueError, match="bins"):
        resolve([MetricSpec(descriptor="volume", error="abs_err", bins=4)], CH)
    with pytest.raises(ValueError, match="bins"):
        resolve([MetricSpec(descriptor="hist", error="w1")], CH)  # required params missing


def test_resolve_rejects_duplicate_names_and_a_non_rankable_primary():
    with pytest.raises(ValueError, match="duplicate"):
        resolve([MetricSpec(error="mae"), MetricSpec(error="mae")], CH)
    with pytest.raises(ValueError, match="rankable"):
        resolve([MetricSpec(descriptor="volume", error="pred"), MetricSpec(error="mae")], CH)
    # non-primary raw reports are fine
    resolve([MetricSpec(error="mae"), MetricSpec(descriptor="volume", error="pred")], CH)


def test_resolve_requires_a_channel_for_a_scalar_descriptor_on_a_multichannel_task():
    # A scalar descriptor would raise inside `compute` on a multi-channel target; that has
    # to be a config error before the run starts, not a crash at the end of epoch 0.
    with pytest.raises(ValueError, match="channel"):
        resolve([MetricSpec(descriptor="saturation", error="abs_err")], ("ux", "uy", "uz"))
    assert resolve([MetricSpec(descriptor="saturation", error="abs_err", channel="ux")], ("ux", "uy", "uz"))[0].channel == 0
    # A single-channel task needs no channel, and voxel errors span channels by design.
    assert resolve([MetricSpec(descriptor="saturation", error="abs_err")], CH)[0].channel is None
    assert resolve([MetricSpec(error="mae")], ("ux", "uy", "uz"))[0].channel is None


def test_resolve_rejects_an_empty_list_and_names_reserved_by_metrics_jsonl():
    with pytest.raises(ValueError, match="at least one"):
        resolve([], CH)
    with pytest.raises(ValueError, match="reserved"):
        resolve([MetricSpec(error="mae", name="epoch")], CH)
    with pytest.raises(ValueError, match="reserved"):
        resolve([MetricSpec(error="mae"), MetricSpec(error="rmse", name="train_loss")], CH)


def test_resolve_accepts_all_valid_combinations_from_the_spec():
    specs = [
        MetricSpec(error="mae"),
        MetricSpec(error="mae", channel="ux"),
        MetricSpec(error="iou"),
        MetricSpec(descriptor="saturation", error="abs_err", channel="phi"),
        MetricSpec(descriptor="area_interface", error="rel_err", channel="phi"),
        MetricSpec(descriptor="area_contact", error="rel_err", phase="w", channel="phi"),
        MetricSpec(descriptor="volume", error="pred", channel="phi"),
        MetricSpec(descriptor="trapped_volume", error="abs_err", channel="phi"),
        MetricSpec(descriptor="hist", error="w1", channel="umag", bins=32, range=(0.0, 0.05)),
    ]
    assert len(resolve(specs, ("phi", "ux", "umag"))) == 9


def sample_batch():
    # (B=2, C=2, H=1, W=4); channel 0 is phi, channel 1 is a scalar field.
    phi_p = torch.tensor([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, -1.0]])
    phi_t = torch.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0]])
    other_p = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    other_t = torch.zeros(2, 4)
    pred = torch.stack([phi_p, other_p], dim=1)[:, :, None, :]
    target = torch.stack([phi_t, other_t], dim=1)[:, :, None, :]
    mask = torch.ones(2, 1, 1, 4, dtype=torch.bool)
    return pred, target, mask


def test_compute_returns_per_sample_tensors_keyed_by_name():
    pred, target, mask = sample_batch()
    metrics = resolve(
        [
            MetricSpec(error="mae", channel="other"),
            MetricSpec(error="iou", channel="phi"),
            MetricSpec(descriptor="volume", error="abs_err", channel="phi"),
            MetricSpec(descriptor="volume", error="rel_err", channel="phi"),
            MetricSpec(descriptor="hist", error="pred", channel="other", bins=2, range=(0.0, 1.0)),
        ],
        ("phi", "other"),
    )
    out = compute(metrics, pred, target, mask)
    assert set(out) == {"mae@other", "iou@phi", "volume/abs_err@phi", "volume/rel_err@phi", "hist/pred@other"}
    assert out["mae@other"].tolist() == pytest.approx([0.0, 1.0])
    assert out["iou@phi"].tolist() == pytest.approx([1.0 / 3.0, 0.25])
    assert out["volume/abs_err@phi"].tolist() == [0.0, 3.0]  # sample 1: pred 1 NW voxel vs target 4
    assert out["volume/rel_err@phi"].tolist() == pytest.approx([0.0, 0.75])
    assert out["hist/pred@other"].shape == (2, 2)


def test_compute_evaluates_a_descriptor_once_per_side_and_passes_error_params(monkeypatch):
    from poreml.metrics import distributions

    calls = []
    original = distributions.hist

    def counting(field, mask, **params):
        calls.append(params)
        return original(field, mask, **params)

    counting.kind, counting.params = original.kind, original.params
    monkeypatch.setitem(DESCRIPTORS._items, "hist", counting)
    pred, target, mask = sample_batch()
    metrics = resolve(
        [
            MetricSpec(descriptor="hist", error="w1", channel="other", bins=2, range=(0.0, 1.0)),
            MetricSpec(descriptor="hist", error="w1_norm", channel="other", bins=2, range=(0.0, 1.0)),
            MetricSpec(descriptor="hist", error="target", channel="other", bins=2, range=(0.0, 1.0)),
        ],
        ("phi", "other"),
    )
    out = compute(metrics, pred, target, mask)
    assert len(calls) == 2  # pred side + target side, shared by all three errors
    assert set(out) == {"hist/w1@other", "hist/w1_norm@other", "hist/target@other"}


def test_resolve_rejects_an_eval_only_primary_and_filters_validation_metrics():
    with pytest.raises(ValueError, match="eval_only"):
        resolve([MetricSpec(error="mae", eval_only=True)], CH)
    metrics = resolve([MetricSpec(error="mae"), MetricSpec(error="iou", eval_only=True)], CH)
    assert [m.name for m in validation_metrics(metrics)] == ["mae"]


def test_compute_without_channel_scores_all_channels_for_voxel_errors():
    pred, target, mask = sample_batch()
    (m,) = resolve([MetricSpec(error="mae")], ("phi", "other"))
    out = compute([m], pred, target, mask)["mae"]
    # sample 0: phi |diff| = [0,2,2,0] mean .5 over 4 voxels, other 0 -> (2+2+0)/8 = 0.5
    # sample 1: phi diffs [0,2,2,2] sum 6, other sum 4 -> 10/8 = 1.25
    assert out.tolist() == pytest.approx([0.5, 1.25])


def test_compute_rejects_bad_shapes():
    (m,) = resolve([MetricSpec(error="mae")], CH)
    with pytest.raises(ValueError):
        compute([m], torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 4, dtype=torch.bool))


def test_is_better_uses_the_errors_direction_and_beats_nan():
    lower, higher = resolve([MetricSpec(error="mae"), MetricSpec(error="iou")], CH)
    assert is_better(lower, 0.1, 0.2)
    assert not is_better(lower, 0.2, 0.1)
    assert is_better(higher, 0.9, 0.5)
    assert not is_better(higher, 0.5, 0.9)
    assert is_better(lower, 0.5, math.nan)
    assert not is_better(lower, math.nan, 0.5)


def test_resolve_maps_channel_u_to_the_velocity_component_range():
    channels = ("phi", "p", "ux", "uy", "uz")
    m = resolve([MetricSpec(error="mae_vec", channel="u"), MetricSpec(error="rel_mae_vec", channel="u")], channels)
    assert [x.name for x in m] == ["mae_vec@u", "rel_mae_vec@u"] and m[0].channel == (2, 5)
    pred, target = torch.rand(2, 5, 3, 3, 3), torch.rand(2, 5, 3, 3, 3)
    mask = torch.ones(2, 1, 3, 3, 3, dtype=torch.bool)
    from poreml.metrics.errors import mae_vec

    assert torch.allclose(compute(m, pred, target, mask)["mae_vec@u"], mae_vec(pred[:, 2:5], target[:, 2:5], mask))


def test_resolve_refuses_a_range_on_a_scalar_descriptor_and_an_incomplete_vector():
    with pytest.raises(ValueError, match="single channel"):
        resolve([MetricSpec(descriptor="saturation", error="abs_err", channel="u")], ("phi", "ux", "uy", "uz"))
    with pytest.raises(ValueError, match="not among target channels"):
        resolve([MetricSpec(error="mae_vec", channel="u")], ("phi", "ux", "uy"))
