"""Layer 2: errors between a prediction-side and a target-side value.

Scalar and distribution errors take `(p, t)` — two tensors of shape `(B,)` or `(B, K)`
produced by a descriptor — and return `(B,)` (raw `pred`/`target` pass `(B, K)` through).
Voxel errors take `(pred, target, mask)` fields directly and score each sample over its
scored voxels and all selected channels. `accepts` says which of the two shapes an
error understands; `resolve` rejects mismatched pairs before a run starts.
"""

from collections.abc import Callable, Iterable

import torch
from torch import Tensor

from ..registry import ERRORS
from .descriptors import PHASE_THRESHOLD, check_field


def error(
    name: str,
    *,
    accepts: Iterable[str],
    higher_is_better: bool = False,
    rankable: bool = True,
    params: Iterable[str] = (),
) -> Callable:
    """Register an error and stamp it with what it accepts and how it ranks.

    `rankable=False` marks raw reports (`pred`, `target`) that have no direction and so
    cannot select a best checkpoint. `params` names descriptor parameters the error also
    needs (a histogram's `bins` and `range` to speak in the field's units); `resolve`
    checks the paired descriptor takes them and `compute` passes them through.
    """

    def decorate(fn):
        fn.accepts = frozenset(accepts)
        fn.higher_is_better = higher_is_better
        fn.rankable = rankable
        fn.params = frozenset(params)
        return ERRORS.register(name)(fn)

    return decorate


@error("pred", accepts=("scalar", "distribution"), rankable=False)
def pred(p: Tensor, t: Tensor) -> Tensor:
    """The raw descriptor value of the prediction."""
    return p


@error("target", accepts=("scalar", "distribution"), rankable=False)
def target(p: Tensor, t: Tensor) -> Tensor:
    """The raw descriptor value of the target."""
    return t


@error("abs_err", accepts=("scalar",))
def abs_err(p: Tensor, t: Tensor) -> Tensor:
    """|p - t| per sample, in the descriptor's units."""
    return (p - t).abs()


@error("rel_err", accepts=("scalar",))
def rel_err(p: Tensor, t: Tensor) -> Tensor:
    """|p - t| / |t| per sample; NaN where the target is 0."""
    denominator = t.abs()
    return (p - t).abs() / denominator.masked_fill(denominator == 0, float("nan"))


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Per-sample mean of `values` over scored voxels and all channels; NaN when none."""
    weights = mask.to(values.dtype).expand_as(values)
    dims = tuple(range(1, values.ndim))
    count = weights.sum(dim=dims)
    return (values * weights).sum(dim=dims) / count.masked_fill(count == 0, float("nan"))


@error("mae", accepts=("voxels",))
def mae(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean absolute error over scored voxels, per sample."""
    check_field(pred, mask)
    return _masked_mean((pred - target).abs(), mask)


@error("rel_mae", accepts=("voxels",))
def rel_mae(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Relative L1 error: `sum |pred - target| / sum |target|` over scored voxels — mae as a fraction.

    NaN when the target is 0 on every scored voxel. For phi in [-1, 1] the denominator is
    close to the scored voxel count, so this reads as mae per unit of |phi|.
    """
    check_field(pred, mask)
    scale = _masked_mean(target.abs(), mask)
    return _masked_mean((pred - target).abs(), mask) / scale.masked_fill(scale == 0, float("nan"))


def _error_norm(pred: Tensor, target: Tensor) -> Tensor:
    """Per-voxel L2 norm of the error across channels: `(B, 1, *spatial)`."""
    return torch.linalg.vector_norm(pred - target, dim=1, keepdim=True)


@error("mae_vec", accepts=("voxels",))
def mae_vec(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean over scored voxels of |pred - target| taken as a vector across channels.

    On a velocity `channel: u` (ux, uy, uz) this is the mean speed error; on one channel it
    equals `mae`. Direction errors count where a per-component mean would cancel them.
    """
    check_field(pred, mask)
    return _masked_mean(_error_norm(pred, target), mask)


@error("rel_mae_vec", accepts=("voxels",))
def rel_mae_vec(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """`mae_vec` divided by the mean target vector norm (the mean speed for `u`); NaN when that is 0."""
    check_field(pred, mask)
    scale = _masked_mean(torch.linalg.vector_norm(target, dim=1, keepdim=True), mask)
    return _masked_mean(_error_norm(pred, target), mask) / scale.masked_fill(scale == 0, float("nan"))


@error("rmse", accepts=("voxels",))
def rmse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Root mean squared error over scored voxels, per sample."""
    check_field(pred, mask)
    return _masked_mean((pred - target) ** 2, mask).sqrt()


@error("iou", accepts=("voxels",), higher_is_better=True)
def iou(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Intersection over union of the non-wetting phase, thresholded at PHASE_THRESHOLD.

    1.0 when the sample has pore space but neither side puts any invading phase in it:
    two predictions that agree there is nothing to invade agree perfectly. NaN when the
    sample scores no voxel at all (an all-solid mask) — nothing was compared, which is
    what mae and saturation report for the same sample.
    """
    check_field(pred, mask)
    p = (pred > PHASE_THRESHOLD) & mask
    t = (target > PHASE_THRESHOLD) & mask
    dims = tuple(range(1, p.ndim))
    intersection = (p & t).sum(dim=dims).to(torch.float32)
    union = (p | t).sum(dim=dims).to(torch.float32)
    scored = mask.sum(dim=tuple(range(1, mask.ndim)))
    out = torch.where(union == 0, torch.ones_like(union), intersection / union.clamp(min=1))
    return out.masked_fill(scored == 0, float("nan"))


def _w1_bins(p: Tensor, t: Tensor) -> Tensor:
    """Wasserstein-1 between two histograms with shared edges, in units of bins: `sum_k |CDF_p[k] - CDF_t[k]|`."""
    return (p.cumsum(dim=1) - t.cumsum(dim=1)).abs().sum(dim=1)


def _bin_moments(h: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and standard deviation of a normalised histogram, in bin units (bin k sits at k + 0.5)."""
    centres = torch.arange(h.shape[1], device=h.device, dtype=h.dtype) + 0.5
    mean = (h * centres).sum(dim=1)
    var = (h * (centres - mean[:, None]) ** 2).sum(dim=1)
    return mean, var.clamp(min=0).sqrt()


@error("w1", accepts=("distribution",), params=("bins", "range"))
def w1(p: Tensor, t: Tensor, *, bins: int, range: tuple[float, float]) -> Tensor:
    """Wasserstein-1 between two histograms with shared edges, in the field's units.

    The bin-unit `sum_k |CDF_p[k] - CDF_t[k]|` scaled by the bin width `(hi - lo) / bins`
    of the descriptor that made the histograms, so a curvature W1 reads in 1/voxel and a
    speed W1 in the field's speed unit. Interpretable, but not comparable across fields
    or scales — see `w1_norm` for that.
    """
    lo, hi = range
    return _w1_bins(p, t) * ((hi - lo) / bins)


@error("w1_norm", accepts=("distribution",))
def w1_norm(p: Tensor, t: Tensor) -> Tensor:
    """Wasserstein-1 divided by the target distribution's standard deviation: error in units of natural spread.

    Scale-free (both terms are in bin units), so one number serves every field: below
    ~0.1 the distributions are practically the same, ~0.5 visibly different. NaN when the
    target has no spread (a single occupied bin) — then W1/σ says nothing.
    """
    _, sigma = _bin_moments(t)
    return _w1_bins(p, t) / sigma.masked_fill(sigma == 0, float("nan"))
