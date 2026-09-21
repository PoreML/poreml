import math

import pytest
import torch

from poreml.metrics.errors import abs_err, iou, mae, pred, rel_err, rel_mae, rmse, target
from poreml.registry import ERRORS


def test_registered_errors_and_their_flags():
    for name in ("pred", "target", "abs_err", "rel_err", "mae", "rmse", "iou"):
        assert name in ERRORS
    assert ERRORS.get("iou").higher_is_better and ERRORS.get("mae").higher_is_better is False
    assert ERRORS.get("pred").rankable is False and ERRORS.get("abs_err").rankable
    assert ERRORS.get("mae").accepts == {"voxels"}
    assert ERRORS.get("abs_err").accepts == {"scalar"}
    assert ERRORS.get("pred").accepts == {"scalar", "distribution"}


def test_pred_and_target_pass_their_side_through():
    p, t = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    assert torch.equal(pred(p, t), p)
    assert torch.equal(target(p, t), t)


def test_abs_err_and_rel_err_are_per_sample():
    p = torch.tensor([1100.0, 900.0, 20.0])
    t = torch.tensor([1000.0, 1000.0, 10.0])
    assert abs_err(p, t).tolist() == [100.0, 100.0, 10.0]
    assert rel_err(p, t).tolist() == pytest.approx([0.1, 0.1, 1.0])


def test_rel_err_is_nan_where_target_is_zero_and_uses_abs_target():
    out = rel_err(torch.tensor([1.0, -2.0]), torch.tensor([0.0, -4.0]))
    assert math.isnan(out[0].item())
    assert out[1].item() == pytest.approx(0.5)


def voxel_case():
    # (B=2, C=1, H=1, W=3)
    p = torch.tensor([[[[1.0, 1.0, 99.0]]], [[[0.0, 0.0, 0.0]]]])
    t = torch.tensor([[[[0.0, 0.5, -99.0]]], [[[0.0, 0.0, 0.0]]]])
    m = torch.tensor([[[[True, True, False]]], [[[False, False, False]]]])
    return p, t, m


def test_mae_ignores_masked_out_voxels_and_is_nan_when_nothing_scored():
    out = mae(*voxel_case())
    assert out[0].item() == pytest.approx(0.75)  # (1.0 + 0.5) / 2; the 99 voxel is masked out
    assert math.isnan(out[1].item())


def test_rel_mae_is_mae_over_mean_abs_target():
    pred = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
    target = torch.tensor([[[[0.5, 0.5, 0.5, 0.5]]]])  # mean |t| = 0.5, mae = 0.5 -> 1.0
    mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    assert rel_mae(pred, target, mask).item() == pytest.approx(1.0)
    assert torch.isnan(rel_mae(pred, torch.zeros_like(target), mask))  # nothing to be relative to


def test_rmse_matches_hand_computation():
    out = rmse(*voxel_case())
    assert out[0].item() == pytest.approx(math.sqrt((1.0 + 0.25) / 2))


def test_mae_averages_over_all_channels_of_a_sample():
    p = torch.tensor([[[[1.0]], [[3.0]]]])  # (1, 2, 1, 1)
    t = torch.zeros(1, 2, 1, 1)
    m = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    assert mae(p, t, m).item() == pytest.approx(2.0)


def test_iou_hand_case_and_empty_case():
    p = torch.tensor([[[[1.0, 1.0, -1.0, -1.0]]], [[[-1.0, -1.0, -1.0, -1.0]]]])
    t = torch.tensor([[[[1.0, -1.0, 1.0, -1.0]]], [[[-1.0, -1.0, -1.0, -1.0]]]])
    m = torch.ones(2, 1, 1, 4, dtype=torch.bool)
    out = iou(p, t, m)
    assert out[0].item() == pytest.approx(1.0 / 3.0)  # intersection {0}, union {0,1,2}
    assert out[1].item() == pytest.approx(1.0)  # both empty: perfect agreement


def test_iou_is_nan_when_no_voxel_is_scored():
    # An all-solid sample scores nothing at all, which is not the same as "both phases are
    # empty here": mae and saturation report NaN for it, and iou has to agree.
    p = torch.tensor([[[[1.0, -1.0]]]])
    t = torch.tensor([[[[-1.0, -1.0]]]])
    m = torch.zeros(1, 1, 1, 2, dtype=torch.bool)
    assert math.isnan(iou(p, t, m).item())


def test_iou_ignores_solid_voxels():
    p = torch.tensor([[[[1.0, 1.0]]]])
    t = torch.tensor([[[[1.0, -1.0]]]])
    m = torch.tensor([[[[True, False]]]])
    assert iou(p, t, m).item() == pytest.approx(1.0)


def test_voxel_errors_reject_bad_shapes():
    with pytest.raises(ValueError):
        mae(torch.ones(1, 1, 2), torch.ones(1, 1, 2), torch.ones(1, 2, 2, dtype=torch.bool))


def test_mae_vec_is_the_mean_norm_of_the_error_vector_and_equals_mae_on_one_channel():
    from poreml.metrics.errors import mae_vec, rel_mae_vec

    mask = torch.ones(1, 1, 1, 1, 2, dtype=torch.bool)
    pred = torch.zeros(1, 3, 1, 1, 2)
    target = torch.zeros(1, 3, 1, 1, 2)
    target[0, :, 0, 0, 0] = torch.tensor([3.0, 4.0, 0.0])  # |Δu| = 5 at voxel 0, 0 at voxel 1
    assert mae_vec(pred, target, mask).tolist() == pytest.approx([2.5])
    # mean target speed is (5 + 0) / 2 = 2.5, so the relative error is 1
    assert rel_mae_vec(pred, target, mask).tolist() == pytest.approx([1.0])

    one_p, one_t = torch.rand(2, 1, 3, 3, 3), torch.rand(2, 1, 3, 3, 3)
    one_mask = torch.rand(2, 1, 3, 3, 3) > 0.3
    assert torch.allclose(mae_vec(one_p, one_t, one_mask), mae(one_p, one_t, one_mask))
    assert torch.allclose(rel_mae_vec(one_p, one_t, one_mask), rel_mae(one_p, one_t, one_mask))


def test_rel_mae_vec_is_nan_when_the_target_is_still():
    from poreml.metrics.errors import rel_mae_vec

    mask = torch.ones(1, 1, 2, 2, 2, dtype=torch.bool)
    assert math.isnan(rel_mae_vec(torch.rand(1, 3, 2, 2, 2), torch.zeros(1, 3, 2, 2, 2), mask).item())
    assert ERRORS.get("mae_vec").accepts == {"voxels"} and ERRORS.get("rel_mae_vec").rankable
