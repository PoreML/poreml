"""Universal Physics Transformer building blocks, from AB-UPT.

Ported from https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers
(`src/modules/**`), licensed by Emmi AI under the Emmi AI Non-Production License (ENPL) —
research, testing and evaluation use only; see `NOTICE` at the repository root. Paper:
Alkin, Bleeker, Kurle, Kronlachner, Sonnleitner, Dorfer, Brandstetter, "AB-UPT: Scaling
Neural CFD Surrogates for High-Fidelity Automotive Aerodynamics Simulations via
Anchored-Branched Universal Physics Transformers", TMLR 2025
(https://arxiv.org/abs/2502.09692).

This is a modified copy: `einops` is replaced by reshape/permute; the supernode radius
graph is built on the voxel grid (`radius_neighbours`: ball offsets + a dense voxel->point
lookup — exact, no torch_geometric) and messages are averaged with `index_add_` (no
torch_scatter); the split/cross attention keep only the generic code paths; kwargs are
passed to attention directly instead of through `attn_kwargs` dicts.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _heads(t: Tensor, num_heads: int) -> Tensor:
    b, s, _ = t.shape
    return t.reshape(b, s, num_heads, -1).transpose(1, 2)  # (B, H, S, Dh)


def _merge(t: Tensor) -> Tensor:
    b, h, s, d = t.shape
    return t.transpose(1, 2).reshape(b, s, h * d)


def _frequencies(dim: int, ndim: int, max_wavelength: float) -> tuple[Tensor, int]:
    """Shared by the sincos and RoPE embeddings: `omega` per axis and the zero padding to reach `dim`."""
    ndim_padding = dim % ndim
    dim_per_ndim = (dim - ndim_padding) // ndim
    sincos_padding = dim_per_ndim % 2
    padding = ndim_padding + sincos_padding * ndim
    effective = (dim - padding) // ndim
    if effective <= 0:
        raise ValueError(f"dim {dim} is too small for ndim {ndim}")
    arange = torch.arange(0, effective, 2, dtype=torch.float32)
    return 1.0 / max_wavelength ** (arange / effective), padding


class ContinuousSincosEmbed(nn.Module):
    """Sine/cosine embedding of continuous coordinates of any dimension (zero-padded to `dim`)."""

    def __init__(self, dim: int, ndim: int, max_wavelength: float = 10000.0) -> None:
        super().__init__()
        self.dim, self.ndim = dim, ndim
        omega, self.padding = _frequencies(dim, ndim, max_wavelength)
        self.register_buffer("omega", omega, persistent=False)

    def forward(self, coords: Tensor) -> Tensor:
        if coords.shape[-1] != self.ndim:
            raise ValueError(f"expected {self.ndim}-D coordinates, got {coords.shape[-1]}")
        with torch.autocast(device_type=coords.device.type, enabled=False):
            out = coords.float().unsqueeze(-1) * self.omega  # (..., ndim, eff/2)
            emb = torch.cat([out.sin(), out.cos()], dim=-1).flatten(-2)  # (..., ndim*eff)
        if self.padding:
            emb = torch.cat([emb, emb.new_zeros(*emb.shape[:-1], self.padding)], dim=-1)
        return emb


class RopeFrequency(nn.Module):
    """Complex RoPE frequencies for continuous coordinates: `(B, S, dim // 2)`, `dim` = head dim."""

    def __init__(self, dim: int, ndim: int, max_wavelength: float = 10000.0) -> None:
        super().__init__()
        self.dim, self.ndim = dim, ndim
        omega, self.padding = _frequencies(dim, ndim, max_wavelength)
        self.register_buffer("omega", omega, persistent=False)

    def forward(self, coords: Tensor) -> Tensor:
        with torch.autocast(device_type=coords.device.type, enabled=False):
            out = (coords.float().unsqueeze(-1) * self.omega).flatten(-2)  # (..., ndim*eff/2)
        if self.padding:
            out = torch.cat([out, out.new_zeros(*out.shape[:-1], self.padding // 2)], dim=-1)
        return torch.polar(torch.ones_like(out), out)


def rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Rotate `(B, H, S, Dh)` by complex `freqs` `(B, S, Dh // 2)` (adapted from llama3)."""
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_ * freqs.unsqueeze(1)).flatten(3)
    return out.type_as(x)


class Mlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DotProductAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim should be divisible by num_heads")
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, s, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        return q, k, v

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        x = F.scaled_dot_product_attention(rope(q, freqs), rope(k, freqs), v)
        return self.proj(_merge(x))


class AnchorAttention(DotProductAttention):
    """Full self-attention among the first `num_anchor_tokens`; the rest (queries) only cross-attend to them."""

    def forward(self, x: Tensor, freqs: Tensor, num_anchor_tokens: int | None = None) -> Tensor:
        if num_anchor_tokens is None or num_anchor_tokens >= x.shape[1]:
            return super().forward(x, freqs)
        anchors, queries = x[:, :num_anchor_tokens], x[:, num_anchor_tokens:]
        q, k, v = self._qkv(anchors)
        bias = None if self.qkv.bias is None else self.qkv.bias[: self.dim]
        q_query = F.linear(queries, self.qkv.weight[: self.dim], bias)
        q = torch.cat([q, _heads(q_query, self.num_heads)], dim=2)
        x = F.scaled_dot_product_attention(rope(q, freqs), rope(k, freqs[:, :num_anchor_tokens]), v)
        return self.proj(_merge(x))


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim should be divisible by num_heads")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q, self.kv, self.proj = nn.Linear(dim, dim), nn.Linear(dim, dim * 2), nn.Linear(dim, dim)

    def forward(self, q: Tensor, kv: Tensor, q_freqs: Tensor, k_freqs: Tensor) -> Tensor:
        b, s, _ = kv.shape
        q = _heads(self.q(q), self.num_heads)
        k, v = self.kv(kv).reshape(b, s, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(rope(q, q_freqs), rope(k, k_freqs), v)
        return self.proj(_merge(x))


class SharedweightsSplitAttention(DotProductAttention):
    """Within-modality attention: surface tokens attend surface anchors, volume tokens volume anchors.

    `split_size` is `[surface_anchors, volume_anchors]` (no queries) or
    `[surface_anchors, surface_queries, volume_anchors, volume_queries]`.
    """

    def forward(self, x: Tensor, split_size: list[int], freqs: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        q, k = rope(q, freqs), rope(k, freqs)
        qs, ks, vs = q.split(split_size, dim=2), k.split(split_size, dim=2), v.split(split_size, dim=2)
        if len(split_size) == 4:
            x1 = F.scaled_dot_product_attention(torch.cat([qs[0], qs[1]], dim=2), ks[0], vs[0])
            x2 = F.scaled_dot_product_attention(torch.cat([qs[2], qs[3]], dim=2), ks[2], vs[2])
        elif len(split_size) == 2:
            x1 = F.scaled_dot_product_attention(qs[0], ks[0], vs[0])
            x2 = F.scaled_dot_product_attention(qs[1], ks[1], vs[1])
        else:
            raise ValueError(f"split_size must have 2 or 4 entries, got {split_size}")
        return self.proj(_merge(torch.cat([x1, x2], dim=2)))


class SharedweightsCrossAttention(DotProductAttention):
    """Cross-modality attention: surface tokens attend volume anchors and vice versa (same `split_size` forms)."""

    def forward(self, x: Tensor, split_size: list[int], freqs: Tensor) -> Tensor:
        q, k, v = self._qkv(x)
        q, k = rope(q, freqs), rope(k, freqs)
        ks, vs = k.split(split_size, dim=2), v.split(split_size, dim=2)
        if len(split_size) == 4:
            qs = q.split([split_size[0] + split_size[1], split_size[2] + split_size[3]], dim=2)
            x1 = F.scaled_dot_product_attention(qs[0], ks[2], vs[2])
            x2 = F.scaled_dot_product_attention(qs[1], ks[0], vs[0])
        elif len(split_size) == 2:
            qs = q.split(split_size, dim=2)
            x1 = F.scaled_dot_product_attention(qs[0], ks[1], vs[1])
            x2 = F.scaled_dot_product_attention(qs[1], ks[0], vs[0])
        else:
            raise ValueError(f"split_size must have 2 or 4 entries, got {split_size}")
        return self.proj(_merge(torch.cat([x1, x2], dim=2)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_ctor: type[nn.Module] = DotProductAttention) -> None:
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), attn_ctor(dim=dim, num_heads=num_heads)
        self.norm2, self.mlp = nn.LayerNorm(dim, eps=1e-6), Mlp(dim)

    def forward(self, x: Tensor, **attn_kwargs) -> Tensor:
        x = x + self.attn(self.norm1(x), **attn_kwargs)
        return x + self.mlp(self.norm2(x))


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1q, self.norm1kv = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = PerceiverAttention(dim=dim, num_heads=num_heads)
        self.norm2, self.mlp = nn.LayerNorm(dim, eps=1e-6), Mlp(dim)

    def forward(self, q: Tensor, kv: Tensor, **attn_kwargs) -> Tensor:
        q = q + self.attn(self.norm1q(q), self.norm1kv(kv), **attn_kwargs)
        return q + self.mlp(self.norm2(q))


def radius_neighbours(index_grid: Tensor, centers: Tensor, radius: float, max_degree: int) -> tuple[Tensor, Tensor]:
    """Neighbours of grid points within `radius` voxels, on the voxel grid.

    `index_grid` is `(D, H, W)` int64: the point id at each voxel, -1 where there is no
    point. `centers` are `(S, 3)` integer voxel coordinates. Returns `(src, dst)`: `src` the
    point ids within the ball around `centers[dst]`, nearest first, at most `max_degree`
    per centre, the centre's own voxel included when it holds a point.
    """
    device = centers.device
    r = int(math.floor(radius))
    rng = torch.arange(-r, r + 1, device=device)
    offsets = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1).reshape(-1, 3)
    dist = offsets.float().norm(dim=1)
    within = dist <= radius + 1e-6
    offsets = offsets[within][dist[within].argsort(stable=True)]  # nearest first, the centre itself at 0
    shape = torch.tensor(index_grid.shape, device=device)
    nb = centers[:, None, :] + offsets[None]  # (S, K, 3)
    inside = ((nb >= 0) & (nb < shape)).all(dim=-1)
    strides = torch.tensor([shape[1] * shape[2], shape[2], 1], device=device)
    flat = (nb.clamp(min=0) * strides).sum(-1).clamp(max=index_grid.numel() - 1)  # in bounds; masked by `inside` below
    ids = torch.where(inside, index_grid.reshape(-1)[flat], torch.full_like(flat, -1))
    hit = ids >= 0
    keep = hit & (hit.cumsum(dim=1) <= max_degree)
    dst = torch.arange(len(centers), device=device)[:, None].expand_as(ids)
    return ids[keep], dst[keep]


class SupernodePooling(nn.Module):
    """Message passing from geometry points to a subset of them (the supernodes), `mode="relpos"`.

    Upstream builds the radius graph with `torch_geometric.nn.pool.radius`; on a voxel
    grid the same neighbourhood is enumerated exactly by `radius_neighbours`.
    """

    def __init__(self, hidden_dim: int, ndim: int = 3, radius: float = 3.0, max_degree: int = 32) -> None:
        super().__init__()
        self.radius, self.max_degree = radius, max_degree
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        self.rel_pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim + 1)
        self.message = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, pos: Tensor, ijk: Tensor, supernode_idx: Tensor, index_grid: Tensor) -> Tensor:
        """`pos` `(N, ndim)` geometry positions, `ijk` their `(N, 3)` voxel coordinates, `supernode_idx`
        `(S,)` indices into them, `index_grid` `(D, H, W)` voxel -> geometry point id (-1 = none).
        Returns `(1, S, hidden_dim)`."""
        src, dst = radius_neighbours(index_grid, ijk[supernode_idx], self.radius, self.max_degree)
        sup_pos = pos[supernode_idx]
        rel = sup_pos[dst] - pos[src]
        msg = self.message(self.rel_pos_embed(torch.cat([rel, rel.norm(dim=1, keepdim=True)], dim=1)))
        agg = torch.zeros(len(supernode_idx), msg.shape[1], dtype=msg.dtype, device=msg.device).index_add_(0, dst, msg)
        counts = torch.bincount(dst, minlength=len(supernode_idx)).clamp(min=1).to(msg.dtype)[:, None]
        x = torch.cat([agg / counts, self.pos_embed(sup_pos)], dim=-1)
        return self.proj(x)[None]
