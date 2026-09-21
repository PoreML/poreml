"""AB-UPT — Anchored-Branched Universal Physics Transformer — on the point stream.

Ported from https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers
(`src/model.py`), licensed by Emmi AI under the Emmi AI Non-Production License (ENPL) —
research, testing and evaluation use only; see `NOTICE`. Paper: Alkin et al., "AB-UPT",
TMLR 2025 (https://arxiv.org/abs/2502.09692). Building blocks live in `upt.py`.

Mapping onto the solver's pore space (this is a modified copy, not the original):
- "surface" tokens are the wall-adjacent pore voxels (`Points.wall`), "volume" tokens the
  bulk pore voxels; both branches predict the same F fields, so the two decoders share the
  output width.
- The geometry branch pools the wall points into `num_supernodes` supernodes (radius in
  voxels, on the grid) and encodes them with `geometry_depth` transformer blocks.
- Anchors: `num_surface_anchors` / `num_volume_anchors` random points per branch, the rest
  are queries that only cross-attend to anchors (linear in N). Random per forward in
  training; drawn from a seeded generator in eval so artifacts reproduce. A branch with
  no more points than anchors has no queries.
- AB-UPT is a positions-only steady-state model; for time stepping the point features
  (`feat_embed`, a linear layer) are added to the sincos position embedding of every token.
- `residual=True` (default) adds the most recent input frame and zero-initialises both
  decoders, so an untrained model is exactly persistence — the same start as `unet3d`.
- Batches are processed one sample at a time over `ptr` (upstream supports batch 1 only).

Verified against upstream by copying weights into `AnchoredBranchedUPT` and comparing outputs:
bit-exact for the sincos embeddings and for everything downstream of the geometry graph, and
6e-08 for supernode pooling, where the port enumerates the radius ball on the voxel grid instead
of calling torch_geometric/torch_scatter (`util/archive/docs/sketch/point-based.md` §8).
"""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..points import Points
from ..registry import MODELS
from .baselines import recent_slice
from .upt import (
    AnchorAttention,
    ContinuousSincosEmbed,
    PerceiverBlock,
    RopeFrequency,
    SharedweightsCrossAttention,
    SharedweightsSplitAttention,
    SupernodePooling,
    TransformerBlock,
)

# Attention backends AB-UPT may use. cuDNN attention is excluded on purpose: it builds an
# execution plan per new (query, key) token count, ~2.4 s each, and every training window
# has a different wall/bulk split, so under bf16 it made a step 5x *slower* than fp32.
# Flash and memory-efficient attention take any length as it comes; math is the CPU fallback.
SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


@MODELS.register("abupt")
class ABUPT(nn.Module):
    representation = "points"
    # Trains in bf16 autocast by default (`models.precision_of`): upstream's recipe, and on an
    # H200 a 0.23 s step against 1.13 s in fp32 with identical loss over 240 windows
    # (2026-09-02). The coordinate embedding and RoPE stay fp32 inside (upt.py).
    precision = "bf16"

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        dim: int = 192,
        num_heads: int = 3,
        geometry_depth: int = 1,
        blocks: str = "pscscs",
        num_surface_blocks: int = 6,
        num_volume_blocks: int = 6,
        num_supernodes: int = 4096,
        supernode_radius: float = 3.0,
        max_degree: int = 32,
        num_surface_anchors: int = 8192,
        num_volume_anchors: int = 16384,
        pos_scale: float = 1000.0,
        residual: bool = True,
        recent_channel: int = -1,
        eval_seed: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.residual = in_channels, out_channels, residual
        self.recent = recent_slice(in_channels, out_channels, recent_channel)
        self.num_supernodes, self.pos_scale, self.eval_seed = num_supernodes, pos_scale, eval_seed
        self.num_surface_anchors, self.num_volume_anchors = num_surface_anchors, num_volume_anchors
        self.rope = RopeFrequency(dim=dim // num_heads, ndim=3)
        self.encoder = SupernodePooling(hidden_dim=dim, ndim=3, radius=supernode_radius, max_degree=max_degree)
        self.geometry_blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(geometry_depth)])
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=3)
        self.surface_bias = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.volume_bias = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.feat_embed = nn.Linear(in_channels, dim)
        ctors = {
            "s": partial(TransformerBlock, attn_ctor=SharedweightsSplitAttention),
            "c": partial(TransformerBlock, attn_ctor=SharedweightsCrossAttention),
            "p": PerceiverBlock,
        }
        if any(b not in ctors for b in blocks):
            raise ValueError(f"blocks must be a string over 'p', 's', 'c'; got {blocks!r}")
        self.blocks = nn.ModuleList([ctors[b](dim=dim, num_heads=num_heads) for b in blocks])
        self.surface_blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, attn_ctor=AnchorAttention) for _ in range(num_surface_blocks)]
        )
        self.volume_blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, attn_ctor=AnchorAttention) for _ in range(num_volume_blocks)]
        )
        self.surface_decoder = nn.Linear(dim, out_channels)
        self.volume_decoder = nn.Linear(dim, out_channels)

        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)
        if residual:
            for dec in (self.surface_decoder, self.volume_decoder):
                nn.init.zeros_(dec.weight)
                nn.init.zeros_(dec.bias)

    def _perm(self, n: int, device) -> Tensor:
        if self.training:
            return torch.randperm(n, device=device)
        g = torch.Generator(device=device).manual_seed(self.eval_seed)
        return torch.randperm(n, device=device, generator=g)

    def forward_one(self, pos: Tensor, feats: Tensor, wall: Tensor, index: Tensor, shape: tuple[int, int, int]) -> Tensor:
        device = pos.device
        wall_idx, bulk_idx = wall.nonzero()[:, 0], (~wall).nonzero()[:, 0]
        n_wall, n_bulk = len(wall_idx), len(bulk_idx)
        if n_wall == 0 or n_bulk == 0:
            raise ValueError(f"abupt needs wall and bulk points; got {n_wall} wall and {n_bulk} bulk")
        pos = pos * self.pos_scale

        # geometry branch: supernodes among the wall points, radius graph on the voxel grid
        sup = self._perm(n_wall, device)[: min(self.num_supernodes, n_wall)]
        wall_flat = index[wall_idx]
        index_grid = torch.full((shape[0] * shape[1] * shape[2],), -1, dtype=torch.long, device=device)
        index_grid[wall_flat] = torch.arange(n_wall, device=device)
        ijk = torch.stack(torch.unravel_index(wall_flat, shape), dim=1)
        geometry = self.encoder(pos[wall_idx], ijk, sup, index_grid.view(shape))
        geometry_rope = self.rope(pos[wall_idx][sup][None])
        for block in self.geometry_blocks:
            geometry = block(geometry, freqs=geometry_rope)

        # anchors and queries per branch; token order = [s_anchor, s_query, v_anchor, v_query]
        s_order, v_order = wall_idx[self._perm(n_wall, device)], bulk_idx[self._perm(n_bulk, device)]
        n_sa, n_va = min(self.num_surface_anchors, n_wall), min(self.num_volume_anchors, n_bulk)
        order = torch.cat([s_order, v_order])
        split = [n_sa, n_wall - n_sa, n_va, n_bulk - n_va]
        freqs = self.rope(pos[order][None])
        x = torch.cat([self.surface_bias(self.pos_embed(pos[s_order])), self.volume_bias(self.pos_embed(pos[v_order]))])
        x = (x + self.feat_embed(feats[order]))[None]
        for block in self.blocks:
            if isinstance(block, PerceiverBlock):
                x = block(x, geometry, q_freqs=freqs, k_freqs=geometry_rope)
            else:
                x = block(x, split_size=split, freqs=freqs)

        x_s, x_v = x.split([n_wall, n_bulk], dim=1)
        f_s, f_v = freqs.split([n_wall, n_bulk], dim=1)
        for block in self.surface_blocks:
            x_s = block(x_s, freqs=f_s, num_anchor_tokens=n_sa if n_sa < n_wall else None)
        for block in self.volume_blocks:
            x_v = block(x_v, freqs=f_v, num_anchor_tokens=n_va if n_va < n_bulk else None)

        out = feats.new_empty(len(feats), self.out_channels)  # the input's dtype, whatever autocast made the decoders emit
        out[s_order] = self.surface_decoder(x_s)[0].to(out.dtype)
        out[v_order] = self.volume_decoder(x_v)[0].to(out.dtype)
        return out + feats[:, self.recent] if self.residual else out

    def forward(self, points: Points) -> Tensor:
        if not isinstance(points, Points):
            raise TypeError(f"{type(self).__name__} consumes points.Points batches, got {type(points).__name__}")
        with sdpa_kernel(SDPA_BACKENDS):
            outs = [
                self.forward_one(points.pos[lo:hi], points.feats[lo:hi], points.wall[lo:hi], points.index[lo:hi], points.shape)
                for lo, hi in points.ptr.unfold(0, 2, 1).tolist()
            ]
        return torch.cat(outs)
