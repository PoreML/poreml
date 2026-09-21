"""Training losses, masked to the pore space.

Kept apart from the loop so the evaluator can score a split with the same loss the run
trained on (`evaluate_loader(loss_fn=...)`) without importing `train`.
"""

import torch
from torch import Tensor


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over pore voxels only.

    Written arithmetically rather than by boolean indexing: an all-solid batch then still
    returns a zero that carries a gradient (a hard `torch.zeros(())` detaches the graph
    and breaks `backward()`), and nothing has to sync the device to count the mask. The
    data layer NaN-fills, so `diff` is finite everywhere and multiplying by a 0/1 mask is
    exactly the masked mean whenever anything is scored.
    """
    diff = (pred - target) ** 2
    weights = mask.to(diff.dtype)
    return (diff * weights).sum() / weights.sum().clamp(min=1)


def _spatial_gradient(x: Tensor, axis: int) -> Tensor:
    """d/dx along one spatial axis of a (B, C, *spatial) tensor, on the unit interval.

    Second-order central differences inside, first-order one-sided at the ends: the rock
    ROI is periodic on no axis, so wrapping would difference the outlet against the inlet.
    Spacing is 1/n as in upstream's uniform quadrature, which is what makes the H1 norm's
    gradient terms dominate at high resolution there and here alike.
    """
    n = x.size(axis)
    if n < 2:
        return torch.zeros_like(x)
    interior = (x.narrow(axis, 2, n - 2) - x.narrow(axis, 0, n - 2)) / 2
    first = x.narrow(axis, 1, 1) - x.narrow(axis, 0, 1)
    last = x.narrow(axis, n - 1, 1) - x.narrow(axis, n - 2, 1)
    return torch.cat([first, interior, last], dim=axis) * n


def masked_h1(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Masked relative H1 (Sobolev) norm over pore voxels, one norm over all channels.

    Upstream neuraloperator's FNO training loss (`H1Loss.rel`), adapted to the benchmark
    in three deliberate ways. Solid voxels are substituted with the target before any
    differencing, so neither the value term nor a pore voxel's wall-adjacent derivative
    ever sees unscored model output. The channel dimension folds into the same norm as
    space instead of upstream's per-channel sum — a per-channel relative norm would
    reweight every field by its own magnitude and quietly change the loss semantics for
    the whole zoo. And the batch is reduced by mean, not sum, so the loss does not scale
    with batch_size. An all-solid sample contributes a connected zero (the eps inside the
    square roots keeps the gradient finite where masked_mse's arithmetic trick cannot).
    """
    solid_free = mask.to(torch.bool)
    pred = torch.where(solid_free, pred, target)
    weights = mask.to(target.dtype)
    reduce_dims = tuple(range(1, pred.ndim))
    diff = ((pred - target) ** 2 * weights).sum(dim=reduce_dims)
    ynorm = ((target**2) * weights).sum(dim=reduce_dims)
    for axis in range(2, pred.ndim):
        dp, dt = _spatial_gradient(pred, axis), _spatial_gradient(target, axis)
        diff = diff + ((dp - dt) ** 2 * weights).sum(dim=reduce_dims)
        ynorm = ynorm + ((dt**2) * weights).sum(dim=reduce_dims)
    scored = (weights.sum(dim=reduce_dims) > 0).to(target.dtype)
    rel = (diff + eps**2).sqrt() / ((ynorm + eps**2).sqrt() + eps)
    return (rel * scored).mean()


LOSSES = {"mse": masked_mse, "h1": masked_h1}


def loss_function(name: str):
    """The loss `train.loss` names."""
    try:
        return LOSSES[name]
    except KeyError:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(LOSSES)}") from None


def per_sample_loss(loss_fn, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """`(B,)`: the loss of each sample on its own — what a batch loss averages over."""
    return torch.stack([loss_fn(pred[i : i + 1], target[i : i + 1], mask[i : i + 1]) for i in range(pred.shape[0])])
