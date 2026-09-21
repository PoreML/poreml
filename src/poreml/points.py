"""The point stream: the solver's windows as point clouds on the pore voxels.

The voxel stream (`data.WindowDataset`) hands a model `(B, C, D, H, W)` tensors. Point
and transformer neural operators (Transolver++, AB-UPT, ...) consume a set of positions
with features instead. This module defines that representation once, for every such
model:

- `PointStructure`: the static point set of one run — every pore voxel of the ROI, its
  position, a few local-geometry channels and whether it touches the wall.
- `Points`: one collated batch — positions, features, voxel indices, wall flags and
  `ptr` sample offsets, plus what is needed to scatter values back to the grid.
- `PointWindowDataset`: `WindowDataset` yielding point samples; `encode`/`decode` turn
  grid frames into `Points` and point predictions back into grid frames.

Feature channels follow the voxel convention: static channels first (geometry, then one
constant column per run condition), then the history frames oldest to newest, F channels
each — the most recent frame is the last F channels, so `recent_slice`, persistence and
residual heads need no second convention.
Metrics, rollout and rendering stay on the grid: `as_grid` scatters predictions back,
solid voxels holding each field's fill exactly as in the voxel stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import default_collate

from .data import DEFAULT_FIELDS, ConditionSpec, FieldSpec, RunRef, WindowDataset

DEFAULT_RADII: tuple[int, ...] = (1, 2, 4)


@dataclass(frozen=True)
class PointStructure:
    """The static point set of one run: every pore voxel of the (cropped) grid.

    `index` is the flat voxel index in `ravel` order of `shape`; `pos` is
    `(ijk + 0.5) / max(shape)`, in (0, 1) with the aspect ratio kept and strictly positive
    (sincos / RoPE embeddings assert that); `geo` holds the local solid fraction in a
    `(2r + 1)^3` box per radius (zero-padded at the domain faces); `wall` marks pore voxels
    with a solid voxel among their 26 neighbours.
    """

    index: np.ndarray  # (N,) int64
    pos: np.ndarray  # (N, 3) float32
    geo: np.ndarray  # (N, G) float32
    wall: np.ndarray  # (N,) bool
    shape: tuple[int, int, int]


def structure(solid: np.ndarray, radii: Sequence[int] = DEFAULT_RADII) -> PointStructure:
    solid = np.asarray(solid, dtype=bool)
    if solid.ndim != 3:
        raise ValueError(f"solid must be (D, H, W), got shape {solid.shape}")
    shape = tuple(int(s) for s in solid.shape)
    index = np.flatnonzero(~solid).astype(np.int64)
    ijk = np.stack(np.unravel_index(index, shape), axis=1).astype(np.float32)
    pos = (ijk + 0.5) / max(shape)
    s = torch.from_numpy(solid.astype(np.float32))[None, None]
    # Explicit zero padding: avg_pool3d caps `padding` at half the kernel and refuses inputs
    # smaller than the kernel, both of which small ROIs and large radii hit.
    geo = [F.avg_pool3d(F.pad(s, [r] * 6), 2 * r + 1, stride=1)[0, 0].reshape(-1)[index] for r in radii]
    geo_np = torch.stack(geo, dim=1).numpy() if geo else np.zeros((len(index), 0), dtype=np.float32)
    wall = (F.max_pool3d(F.pad(s, [1] * 6), 3, stride=1)[0, 0].reshape(-1)[index] > 0).numpy()
    return PointStructure(index=index, pos=pos.astype(np.float32), geo=geo_np.astype(np.float32), wall=wall, shape=shape)


def gather(frame: np.ndarray, index: np.ndarray) -> np.ndarray:
    """`(F, D, H, W)` grid values at flat voxel indices -> `(N, F)` float32."""
    flat = np.asarray(frame).reshape(frame.shape[0], -1)
    return np.ascontiguousarray(flat[:, index].T, dtype=np.float32)


@dataclass
class Points:
    """One collated point batch. Sample `b` is rows `ptr[b]:ptr[b+1]`; all samples share `shape`."""

    pos: Tensor  # (N, 3)
    feats: Tensor  # (N, C)
    index: Tensor  # (N,) flat voxel index within its sample's grid
    wall: Tensor  # (N,) bool
    ptr: Tensor  # (B + 1,)
    shape: tuple[int, int, int]
    fill: Tensor  # (F,) per-field fill for `dense`

    def to(self, device) -> Points:
        return Points(
            pos=self.pos.to(device),
            feats=self.feats.to(device),
            index=self.index.to(device),
            wall=self.wall.to(device),
            ptr=self.ptr.to(device),
            shape=self.shape,
            fill=self.fill.to(device),
        )

    @property
    def n_samples(self) -> int:
        return len(self.ptr) - 1

    @property
    def batch(self) -> Tensor:
        """`(N,)` sample index of every point."""
        return torch.repeat_interleave(torch.arange(self.n_samples, device=self.ptr.device), self.ptr.diff())

    def sample(self, b: int) -> Points:
        lo, hi = int(self.ptr[b]), int(self.ptr[b + 1])
        return Points(
            pos=self.pos[lo:hi],
            feats=self.feats[lo:hi],
            index=self.index[lo:hi],
            wall=self.wall[lo:hi],
            ptr=torch.tensor([0, hi - lo], device=self.ptr.device),
            shape=self.shape,
            fill=self.fill,
        )

    def dense(self, values: Tensor, fill: Tensor | float | None = None) -> Tensor:
        """Scatter `(N, C)` point values to `(B, C, D, H, W)`.

        Voxels without a point hold `fill`: a `(C,)` tensor fills per channel, a float
        every channel, `None` zeros (`False` for bool values).
        """
        if values.ndim != 2 or values.shape[0] != len(self.index):
            raise ValueError(f"values must be (N, C) with N={len(self.index)}, got {tuple(values.shape)}")
        n_batch, n_channels, n_voxels = self.n_samples, values.shape[1], prod(self.shape)
        out = torch.zeros((n_batch, n_channels, n_voxels), dtype=values.dtype, device=values.device)
        if fill is not None:
            fill_t = torch.as_tensor(fill, dtype=values.dtype, device=values.device)
            out += fill_t.reshape(1, -1, 1) if fill_t.ndim else fill_t
        out[self.batch, :, self.index] = values
        return out.view(n_batch, n_channels, *self.shape)


def collate(samples: Sequence[tuple[Points, Tensor, Tensor, dict[str, Any]]]):
    """DataLoader collate for point samples: concatenate, build `ptr`, collate `meta` to tensors."""
    points, targets, masks, metas = zip(*samples, strict=True)
    shape = points[0].shape
    if any(p.shape != shape for p in points):
        raise ValueError(f"cannot collate point samples of different grid shape: {[p.shape for p in points]}")
    lengths = torch.tensor([0] + [len(p.index) for p in points])
    batch = Points(
        pos=torch.cat([p.pos for p in points]),
        feats=torch.cat([p.feats for p in points]),
        index=torch.cat([p.index for p in points]),
        wall=torch.cat([p.wall for p in points]),
        ptr=lengths.cumsum(0),
        shape=shape,
        fill=points[0].fill,
    )
    return batch, torch.cat(list(targets)), torch.cat(list(masks)), default_collate(list(metas))


def as_grid(inputs, *values: Tensor) -> tuple[Tensor, ...]:
    """Bring model outputs and targets to the metric layout.

    A tensor input (the voxel stream) already is `(B, C, *spatial)`: `values` come back
    unchanged. A `Points` input scatters each `(N, C)` value to `(B, C, D, H, W)`: bool
    values (masks) fill with False, values with as many channels as fields with the fields'
    fills (solid = fill, as the voxel stream), anything else with 0.
    """
    if not isinstance(inputs, Points):
        return values
    out = []
    for v in values:
        if v.dtype == torch.bool:
            out.append(inputs.dense(v))
        elif v.shape[1] == len(inputs.fill):
            out.append(inputs.dense(v, inputs.fill))
        else:
            out.append(inputs.dense(v))
    return tuple(out)


class PointWindowDataset(WindowDataset):
    """`WindowDataset` yielding `(Points, target (N, F), mask (N, 1), meta)`.

    Same runs, history, fields, stride, ROI and conditions as the voxel dataset; the sample
    is the set of pore voxels with features
    `[geo (G) | conditions (K) | frames oldest..newest (F each)]`.
    `train_points` draws a uniform random subset of that many points per window — a
    memory trade-off for training only; `encode` (used by rollout) and every evaluation
    always take all points.
    """

    def __init__(
        self,
        runs: Sequence[RunRef],
        history: int = 1,
        fields: Sequence[FieldSpec] = DEFAULT_FIELDS,
        stride: int = 1,
        roi: str | None = "rock",
        geometry_radii: Sequence[int] = DEFAULT_RADII,
        train_points: int | None = None,
        conditions: Sequence[ConditionSpec] = (),
        wall_channel: bool = False,
        future: int = 1,
    ) -> None:
        super().__init__(runs, history=history, fields=fields, stride=stride, roi=roi, conditions=conditions, future=future)
        if train_points is not None and train_points < 1:
            raise ValueError(f"train_points must be at least 1, got {train_points}")
        self.wall_channel = bool(wall_channel)
        self.geometry_radii = tuple(int(r) for r in geometry_radii)
        self.train_points = train_points
        self._structures: dict[int, PointStructure] = {}

    collate = staticmethod(collate)

    @property
    def n_geometry_channels(self) -> int:
        # The binary wall flag (when configured), then one solid fraction per radius. The flag
        # defaults off so a checkpoint's saved config rebuilds exactly the inputs it was trained
        # on; shipped configs turn it on (the project default lives in the configs, not here).
        return int(self.wall_channel) + len(self.geometry_radii)

    def structure(self, run_idx: int) -> PointStructure:
        if run_idx not in self._structures:
            self._structures[run_idx] = structure(self.solid(run_idx), self.geometry_radii)
        return self._structures[run_idx]

    def _points(self, run_idx: int, frames: Sequence[np.ndarray], select: np.ndarray | None = None) -> Points:
        st = self.structure(run_idx)
        index, pos, geo, wall = st.index, st.pos, st.geo, st.wall
        if select is not None:
            index, pos, geo, wall = index[select], pos[select], geo[select], wall[select]
        # Static channels first (geometry, then the run's conditions), history frames after —
        # the voxel channel convention, so recent_slice and residual heads are shared. With
        # `wall_channel`, geometry leads with the binary wall flag, mirroring the voxel
        # stream's solid-mask channel 0.
        static = [wall[:, None].astype(np.float32), geo] if self.wall_channel else [geo]
        if len(self.conditions):
            static.append(np.broadcast_to(self.cond_values[run_idx][None, :], (len(index), len(self.conditions))))
        feats = np.concatenate([*static, *[gather(f, index) for f in frames]], axis=1)
        return Points(
            pos=torch.from_numpy(np.ascontiguousarray(pos)),
            feats=torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)),
            index=torch.from_numpy(np.ascontiguousarray(index)),
            wall=torch.from_numpy(np.ascontiguousarray(wall)),
            ptr=torch.tensor([0, len(index)]),
            shape=st.shape,
            fill=torch.from_numpy(self.frame_fill),
        )

    def encode(self, run_idx: int, frames: Sequence[np.ndarray]) -> Points:
        """All pore points of one run with `frames` (oldest first) as features: a batch of one."""
        return self._points(run_idx, frames)

    def decode(self, inputs: Points, pred: Tensor) -> Tensor:
        """`(N, F)` point predictions -> `(1, F, D, H, W)` grid, solid voxels at each field's fill."""
        return inputs.dense(pred, inputs.fill)

    def __getitem__(self, i: int):
        run_idx, t = self.index[i]
        st = self.structure(run_idx)
        select = None
        if self.train_points is not None and self.train_points < len(st.index):
            select = torch.randperm(len(st.index))[: self.train_points].sort().values.numpy()
        past = [self.frame(run_idx, t - k) for k in reversed(range(self.history))]
        pts = self._points(run_idx, past, select)
        index = pts.index.numpy()
        if self.future == 1:
            target = torch.from_numpy(gather(self.frame(run_idx, t + 1), index))
        else:  # (N, future, F): the point axis stays first so `collate` concatenates samples as before
            target = torch.from_numpy(
                np.stack([gather(self.frame(run_idx, t + 1 + k), index) for k in range(self.future)], axis=1)
            )
        mask = torch.ones(len(pts.index), 1, dtype=torch.bool)
        return pts, target, mask, {"run": run_idx, "t": t}

    def __repr__(self) -> str:
        return (
            f"PointWindowDataset(n_runs={len(self.runs)}, history={self.history}, roi={self.roi!r}, "
            f"channels={self.channels}, radii={self.geometry_radii}, train_points={self.train_points}, "
            f"n_windows={len(self)})"
        )
