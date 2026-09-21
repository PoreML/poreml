"""Distribution descriptors: a field summarised as a fixed-bin histogram per sample.

Edges come from the spec (`bins`, `range`) so pred and target share them and their
histograms are directly comparable. The histogram is built for the whole batch at once
with `bucketize` + `scatter_add` — no per-sample loop — over every scored voxel of
every selected channel.
"""

import torch
from torch import Tensor

from .descriptors import DISTRIBUTION_KIND, check_field, descriptor


@descriptor("hist", kind=DISTRIBUTION_KIND, params=("bins", "range"))
def hist(field: Tensor, mask: Tensor, *, bins: int, range: tuple[float, float]) -> Tensor:
    """Normalised histogram `(B, bins)` of the scored voxels; NaN row when none are scored.

    Values below `range[0]` land in the first bin and above `range[1]` in the last, so
    nothing is silently dropped when a prediction overshoots the configured range.
    """
    check_field(field, mask)
    lo, hi = range
    if not hi > lo:
        raise ValueError(f"hist range must satisfy lo < hi; got {range}")
    edges = torch.linspace(lo, hi, bins + 1, device=field.device, dtype=field.dtype)
    # Contiguous: selecting a channel leaves a strided view, and torch.searchsorted
    # warns and copies anyway when handed one.
    values = field.flatten(1).contiguous()  # (B, C * prod(spatial))
    weights = mask.expand_as(field).flatten(1).to(field.dtype)
    index = torch.bucketize(values, edges[1:-1], right=True)  # 0 .. bins-1, ends absorb out-of-range
    counts = torch.zeros(field.shape[0], bins, device=field.device, dtype=field.dtype)
    counts.scatter_add_(1, index, weights)
    total = counts.sum(dim=1, keepdim=True)
    return counts / total.masked_fill(total == 0, float("nan"))
