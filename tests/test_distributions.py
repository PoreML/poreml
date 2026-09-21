import math

import pytest
import torch

from poreml.metrics.distributions import hist
from poreml.metrics.errors import w1, w1_norm
from poreml.registry import DESCRIPTORS, ERRORS


def test_hist_is_registered_as_a_distribution_with_its_params():
    fn = DESCRIPTORS.get("hist")
    assert fn.kind == "distribution"
    assert fn.params == {"bins", "range"}
    assert ERRORS.get("w1").accepts == {"distribution"}


def test_hist_counts_masked_voxels_into_fixed_bins():
    # (B=1, C=1, H=1, W=6). Bins over [0, 1) with 4 bins: edges 0, .25, .5, .75, 1.
    x = torch.tensor([[[[0.1, 0.1, 0.3, 0.6, 0.9, 0.9]]]])
    m = torch.tensor([[[[True, True, True, True, True, False]]]])  # last voxel is solid
    out = hist(x, m, bins=4, range=(0.0, 1.0))
    assert out.shape == (1, 4)
    assert out[0].tolist() == pytest.approx([0.4, 0.2, 0.2, 0.2])


def test_hist_clips_out_of_range_values_into_the_end_bins():
    x = torch.tensor([[[[-5.0, 5.0]]]])
    m = torch.ones_like(x, dtype=torch.bool)
    assert hist(x, m, bins=2, range=(0.0, 1.0))[0].tolist() == pytest.approx([0.5, 0.5])


def test_hist_is_batched_and_nan_for_an_empty_sample():
    x = torch.tensor([[[[0.1, 0.9]]], [[[0.1, 0.9]]]])
    m = torch.tensor([[[[True, True]]], [[[False, False]]]])
    out = hist(x, m, bins=2, range=(0.0, 1.0))
    assert out[0].tolist() == pytest.approx([0.5, 0.5])
    assert all(math.isnan(v) for v in out[1].tolist())


def test_hist_pools_all_channels_of_a_sample():
    x = torch.tensor([[[[0.1]], [[0.9]]]])  # (1, 2, 1, 1)
    m = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    assert hist(x, m, bins=2, range=(0.0, 1.0))[0].tolist() == pytest.approx([0.5, 0.5])


def test_w1_is_the_shift_in_field_units():
    # 3 bins over [0, 3): a one-bin shift is 1.0, a two-bin shift 2.0; over [0, 0.3) they scale by 0.1.
    p = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    t = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert w1(p, t, bins=3, range=(0.0, 3.0)).tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert w1(p, t, bins=3, range=(0.0, 0.3)).tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert ERRORS.get("w1").params == {"bins", "range"}


def test_w1_norm_is_w1_over_target_spread_and_scale_free():
    # Target spread over bins 0 and 2 (σ = 1 bin); pred one bin to the right: W1 = 1 bin → 1.0 whatever the range.
    p = torch.tensor([[0.0, 0.5, 0.0, 0.5]])
    t = torch.tensor([[0.5, 0.0, 0.5, 0.0]])
    assert w1_norm(p, t).item() == pytest.approx(1.0)
    assert ERRORS.get("w1_norm").params == frozenset()
    # A single-bin target has no spread: NaN, not inf.
    assert torch.isnan(w1_norm(torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]])))
