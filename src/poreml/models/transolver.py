"""Transolver++ on the point stream.

Ported from https://github.com/thuml/Transolver_plus (`models/Transolver_plus.py`),
MIT License, Copyright (c) 2025 THUML @ Tsinghua University. Paper: Luo, Wu, Zhou, Xing,
Di, Wang, Long, "Transolver++: An Accurate Neural Solver for PDEs on Million-Scale
Geometries", ICML 2025 (https://arxiv.org/abs/2502.02414).

The core is Physics-Attention with *eidetic states*: every point is softly assigned to
`slices` physical states by a gumbel-softmax over learned logits with a learned, clamped
temperature; attention runs among the few slice tokens and is broadcast back with the
same weights, so cost is linear in the number of points.

Changes from upstream (this is a modified copy, not the original):
- inputs are `points.Points`: `[pos | feats]` per point, one sample at a time over `ptr`
  (upstream is batch 1 anyway); the output is `(N, out_channels)`;
- `torch.distributed` all-reduces of the slice statistics are dropped (single process);
- gumbel noise is drawn in training only — evaluation uses the noise-free softmax so
  artifacts reproduce (upstream samples noise at inference too);
- `timm.trunc_normal_` -> `torch.nn.init.trunc_normal_`, `einops` -> reshape/permute;
- `unified_pos` (reference-grid distances) and the condition embedding are not ported:
  positions are already a regular grid, and run conditions (M, ca, theta) arrive as
  per-point features through the point stream instead (`task.params.conditions`);
- `residual=True` (default) adds the most recent input frame and zero-initialises the head,
  so an untrained model is exactly persistence — the same start as `unet3d`.

Verified against upstream by copying weights parameter for parameter into
`models.Transolver_plus.Model` and comparing outputs: max|diff| 2.4e-07, relative 6e-07
(`util/archive/docs/sketch/point-based.md` §8).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..points import Points
from ..registry import MODELS
from .baselines import recent_slice


def gumbel_softmax(logits: Tensor, tau: Tensor, noise: bool) -> Tensor:
    if noise:
        u = torch.rand_like(logits)
        logits = logits - torch.log(-torch.log(u + 1e-8) + 1e-8)
    return F.softmax(logits / tau, dim=-1)


class EideticAttention(nn.Module):
    """Physics-Attention with eidetic states (`Physics_Attention_1D_Eidetic` upstream)."""

    def __init__(self, dim: int, heads: int = 8, slices: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads, self.dim_head = heads, dim // heads
        self.bias = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.proj_temperature = nn.Sequential(nn.Linear(self.dim_head, slices), nn.GELU(), nn.Linear(slices, 1), nn.GELU())
        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_slice = nn.Linear(self.dim_head, slices)
        nn.init.orthogonal_(self.in_project_slice.weight)  # upstream: "a principled initialization"
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:  # (B, N, C)
        b, n, c = x.shape
        x_mid = self.in_project_x(x).reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)  # B H N Dh
        temperature = (self.proj_temperature(x_mid) + self.bias).clamp(min=0.01)  # B H N 1
        weights = gumbel_softmax(self.in_project_slice(x_mid), temperature, noise=self.training)  # B H N G
        norm = weights.sum(dim=2)  # B H G
        tokens = torch.einsum("bhnc,bhng->bhgc", x_mid, weights) / (norm + 1e-5)[..., None]
        out = F.scaled_dot_product_attention(self.to_q(tokens), self.to_k(tokens), self.to_v(tokens))  # B H G Dh
        out = torch.einsum("bhgc,bhng->bhnc", out, weights).permute(0, 2, 1, 3).reshape(b, n, c)
        return self.to_out(out)


class Mlp(nn.Module):
    """`Linear -> GELU -> Linear` (upstream `MLP` with `n_layers=0`)."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransolverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, slices: int, mlp_ratio: int, dropout: float, use_checkpoint: bool) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = EideticAttention(dim, heads=heads, slices=slices, dropout=dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * mlp_ratio, dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.use_checkpoint:
            x = checkpoint(self.attn, self.ln_1(x), use_reentrant=False) + x
            x = checkpoint(self.mlp, self.ln_2(x), use_reentrant=False) + x
        else:
            x = self.attn(self.ln_1(x)) + x
            x = self.mlp(self.ln_2(x)) + x
        return x


@MODELS.register("transolver")
class Transolver(nn.Module):
    representation = "points"

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        dim: int = 128,
        depth: int = 5,
        heads: int = 8,
        slices: int = 32,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
        recent_channel: int = -1,
        checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.residual = in_channels, out_channels, residual
        self.recent = recent_slice(in_channels, out_channels, recent_channel)
        self.preprocess = Mlp(3 + in_channels, dim * 2, dim)
        self.placeholder = nn.Parameter(torch.rand(dim) / dim)
        self.blocks = nn.ModuleList([TransolverBlock(dim, heads, slices, mlp_ratio, dropout, checkpoint) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels)
        self.apply(self._init_weights)
        if residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_one(self, pos: Tensor, feats: Tensor) -> Tensor:
        """One sample: `(N, 3)` positions and `(N, C)` features -> `(N, out_channels)`."""
        x = self.preprocess(torch.cat([pos, feats], dim=-1)[None]) + self.placeholder
        for block in self.blocks:
            x = block(x)
        out = self.head(self.norm(x))[0]
        return out + feats[:, self.recent] if self.residual else out

    def forward(self, points: Points) -> Tensor:
        if not isinstance(points, Points):
            raise TypeError(f"{type(self).__name__} consumes points.Points batches, got {type(points).__name__}")
        outs = [self.forward_one(points.pos[lo:hi], points.feats[lo:hi]) for lo, hi in points.ptr.unfold(0, 2, 1).tolist()]
        return torch.cat(outs)
