"""Building blocks of P3D — hybrid CNN / windowed-transformer 3-D backbone.

Ported from https://github.com/tum-pbs/P3D @ ebf216a (`src/p3d_surrogate/models/p3d.py`),
Apache License 2.0 (the repository's `LICENSE`; its `pyproject.toml` and README badge claim MIT —
see `NOTICE`). Paper: Holzschuh, Kohl, Redinger, Thuerey, "P3D: Scalable Neural Surrogates for
High-Resolution 3D Physics Simulations with Global Context", 2025
(https://arxiv.org/abs/2509.10186). Upstream study: `util/archive/docs/sketch/p3d.md`.

Module and parameter names follow upstream so a weight-for-weight comparison stays readable.
This is a modified copy: `einops` becomes reshape/permute, `numpy` becomes `math`, the
`diffusers` mixins (`ModelMixin`, `ConfigMixin`, `Timesteps`, `TimestepEmbedding`,
`LabelEmbedding`) are reimplemented here as plain modules, the 1000-row class-label table
becomes one learned vector (poreml has no class labels; upstream with a fixed label 0 and no
dropout is exactly that row), and the relative-position bias is recomputed every forward instead
of being cached into a registered buffer that upstream then shadows with a plain attribute.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestep_embedding(t: Tensor, dim: int, flip_sin_to_cos: bool = True, shift: float = 1.0) -> Tensor:
    """`diffusers.get_timestep_embedding`, the variant upstream's wrapper conditioning uses."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - shift)
    emb = t[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    return F.pad(emb, (0, 1, 0, 0)) if dim % 2 else emb


class TimestepEmbedding(nn.Module):
    """`diffusers.TimestepEmbedding`: linear, SiLU, linear."""

    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class TimestepEmbedder(nn.Module):
    """Upstream's own sinusoidal embedder (cos before sin, no frequency shift), used inside the backbone."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def frequencies(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1) if dim % 2 else emb

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.frequencies(t, self.frequency_embedding_size))


class Conditioning(nn.Module):
    """Upstream `CombinedTimestepLabelParameterEmbeddings` with the class table reduced to one vector."""

    def __init__(self, embedding_dim: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.timestep_embedder = TimestepEmbedding(frequency_embedding_size, embedding_dim)
        self.parameter_embedder = TimestepEmbedding(frequency_embedding_size, embedding_dim)
        self.class_vector = nn.Parameter(torch.randn(embedding_dim) * 0.02)

    def forward(self, timestep: Tensor, parameter: Tensor) -> Tensor:
        t = self.timestep_embedder(timestep_embedding(timestep, self.frequency_embedding_size))
        p = self.parameter_embedder(timestep_embedding(parameter, self.frequency_embedding_size))
        return t + p + self.class_vector[None]


class PixelShuffle3D(nn.Module):
    """`(B, C r^3, D, H, W) -> (B, C, D r, H r, W r)`."""

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        r = self.upscale_factor
        c, d, h, w = x.shape[-4:]
        if c % r**3:
            raise ValueError(f"channels {c} must be divisible by upscale_factor^3 = {r**3}")
        view = x.contiguous().view(*x.shape[:-4], c // r**3, r, r, r, d, h, w)
        lead = list(range(view.ndim))[:-6]
        out = view.permute(*lead, -3, -6, -2, -5, -1, -4).contiguous()
        return out.view(*x.shape[:-4], c // r**3, d * r, h * r, w * r)


class LayerNorm3d(nn.LayerNorm):
    """LayerNorm over the channel axis of a `(B, C, D, H, W)` volume."""

    def __init__(self, norm_shape: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__(norm_shape, eps=eps, elementwise_affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 4, 1, 2, 3)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


def window_partition(x: Tensor, window: int) -> Tensor:
    """`(B, H, W, D, C) -> (B * windows, window^3, C)`."""
    b, h, w, d, c = x.shape
    x = x.view(b, h // window, window, w // window, window, d // window, window, c)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, window**3, c)


def window_reverse(windows: Tensor, window: int, h: int, w: int, d: int, b: int) -> Tensor:
    x = windows.view(b, h // window, w // window, d // window, window, window, window, -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, h, w, d, -1)


class RelativePositionBias(nn.Module):
    """Upstream `PosEmbMLPSwinv3D`: an MLP on log-spaced relative coordinates, `16 * sigmoid` per head."""

    def __init__(self, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.window_size, self.num_heads = window_size, num_heads
        self.cpb_mlp = nn.Sequential(nn.Linear(3, 512), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))

        span = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        table = torch.stack(torch.meshgrid(span, span, span, indexing="ij")).permute(1, 2, 3, 0).contiguous()
        table = table.unsqueeze(0) / (window_size - 1) * 8  # normalise to [-8, 8]
        table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / math.log2(8)
        self.register_buffer("relative_coords_table", table)

        axis = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous() + (window_size - 1)
        rel[:, :, 0] *= (2 * window_size - 1) ** 2
        rel[:, :, 1] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

    def forward(self, attn: Tensor) -> Tensor:
        n = self.window_size**3
        table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        bias = table[self.relative_position_index.view(-1)].view(n, n, -1).permute(2, 0, 1).contiguous()
        return attn + 16 * torch.sigmoid(bias).unsqueeze(0)


class WindowAttention3D(nn.Module):
    """Cosine ("v2") window attention with a clamped learned logit scale, after FasterViT."""

    def __init__(self, dim: int, num_heads: int, window_size: int, qkv_bias: bool = False) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.pos_emb_funct = RelativePositionBias(window_size, num_heads)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn * torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = self.pos_emb_funct(attn).softmax(dim=-1)
        return (attn @ v).transpose(1, 2).reshape(b, n, c)


class AdaLayerNormZero(nn.Module):
    """Six modulation vectors (shift, scale, gate for attention and MLP) from the conditioning embedding."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def forward(self, emb: Tensor) -> tuple[Tensor, ...]:
        return self.linear(self.silu(emb)).chunk(6, dim=1)


class P3DBlock(nn.Module):
    """DiT-style block over one window: modulated window attention, then a modulated channel MLP."""

    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads=num_heads, window_size=window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.adain_2 = AdaLayerNormZero(dim)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        windows_per_sample = x.shape[0] // emb.shape[0]
        mods = [m.repeat_interleave(windows_per_sample, dim=0)[:, None] for m in self.adain_2(emb)]
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = mods

        h = self.norm1(x) * (1 + msa_scale) + msa_shift
        x = x + self.attn(h) * (1 + msa_gate)
        h = self.norm2(x) * (1 + mlp_scale) + mlp_shift
        return x + self.mlp(h) * (1 + mlp_gate)


class P3DStage(nn.Module):
    """`depth` blocks of window attention over a `(B, C, H, W, D)` volume, padded up to the window."""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([P3DBlock(dim, num_heads, window_size, mlp_ratio) for _ in range(depth)])
        self.window_size = window_size

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        b, _, h, w, d = x.shape
        window = self.window_size
        for block in self.blocks:
            x = x.permute(0, 2, 3, 4, 1)
            pad = [(window - n % window) % window for n in (h, w, d)]
            x = F.pad(x, (0, 0, 0, pad[2], 0, pad[1], 0, pad[0]))
            padded = x.shape[1:4]
            x = window_reverse(block(window_partition(x, window), emb), window, *padded, b)
            x = x[:, :h, :w, :d].contiguous().permute(0, 4, 1, 2, 3)
        return x


class Downsample(nn.Module):
    """Stride-2 convolution that doubles the channels."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat * 2, 3, stride=2, padding=1, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Convolution to `2 C` then a 3-D pixel shuffle: halves the channels, doubles every axis."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False), PixelShuffle3D(2))

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class FinalLayer(nn.Module):
    """Modulated LayerNorm and a 3x3x3 projection; zero-initialised, so the backbone starts silent."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = LayerNorm3d(hidden_size, affine=False, eps=1e-6)
        self.out_proj = nn.Conv3d(hidden_size, out_channels, 3, stride=1, padding=1, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return self.out_proj(x)


class ConditionedConv3DBlock(nn.Module):
    """Residual conv block whose GroupNorm is modulated by the conditioning embedding (upstream's AdaIN)."""

    def __init__(self, in_channels: int, embed_dim: int, num_groups: int = 32) -> None:
        super().__init__()
        self.gn_1 = nn.GroupNorm(num_groups, in_channels)
        self.activation_1 = nn.GELU()
        self.conv_1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.mlp_scale_bias = nn.Linear(embed_dim, 2 * in_channels)
        self.gn_2 = nn.GroupNorm(num_groups, in_channels)
        self.activation_2 = nn.GELU()
        self.conv_2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        scale, shift = self.mlp_scale_bias(embedding).chunk(2, dim=-1)
        h = self.conv_1(self.activation_1(self.gn_1(x)))
        h = self.gn_2(h) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return x + self.conv_2(self.activation_2(h))


class ConditionedEncoder3D(nn.Module):
    """Convolutional stem: embed, then `num_downsampling_layers` stride-2 steps with conditioned blocks."""

    def __init__(
        self,
        in_channels: int,
        feature_embedding_dim: Sequence[int],
        num_downsampling_layers: int,
        embedding_dim: int,
        repetitions: int = 1,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        dims = list(feature_embedding_dim)
        self.repetitions, self.num_downsampling_layers = repetitions, num_downsampling_layers
        self.feature_embed = nn.Conv3d(in_channels, dims[0], 3, 1, 1)
        self.downsampling_layers = nn.ModuleList(
            [nn.Conv3d(dims[i], dims[i + 1], 3, 2, 1) for i in range(num_downsampling_layers)]
        )
        self.blocks = nn.ModuleList(
            [
                ConditionedConv3DBlock(dims[i + 1], embedding_dim, num_groups=num_groups)
                for i in range(num_downsampling_layers - 1)
                for _ in range(repetitions)
            ]
        )

    def forward(self, x: Tensor, embedding: Tensor) -> list[Tensor]:
        x = self.feature_embed(x)
        residuals = [x]
        x = self.downsampling_layers[0](x)
        for i in range(self.num_downsampling_layers - 1):
            for j in range(self.repetitions):
                x = self.blocks[i * self.repetitions + j](x, embedding)
            residuals.append(x)
            x = self.downsampling_layers[i + 1](x)
        residuals.append(x)
        return residuals


class DecoderUpsamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear_conv = nn.Conv3d(in_channels, out_channels * 8, 1)
        self.shuffle = PixelShuffle3D(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.shuffle(self.linear_conv(x))


class ConditionedDecoder3D(nn.Module):
    """Mirror of the encoder: pixel-shuffle upsampling, conditioned blocks, additive encoder skips."""

    def __init__(
        self,
        out_channels: int,
        feature_embedding_dim: Sequence[int],
        num_upsampling_layers: int,
        embedding_dim: int,
        features_first_layer: int,
        repetitions: int = 1,
        num_groups: int = 32,
        skip_connections_active: bool = True,
    ) -> None:
        super().__init__()
        dims = list(feature_embedding_dim)
        self.num_upsampling_layers, self.repetitions = num_upsampling_layers, repetitions
        self.skip_connections_active = skip_connections_active
        self.decompress = nn.Conv3d(dims[-1], out_channels, 3, 1, 1)
        self.blocks = nn.ModuleList(
            [
                ConditionedConv3DBlock(dims[i + 1], embedding_dim, num_groups=num_groups)
                for i in range(num_upsampling_layers - 1)
                for _ in range(repetitions)
            ]
        )
        self.upsampling_layers = nn.ModuleList([DecoderUpsamplingBlock(features_first_layer, dims[1])])
        for i in range(num_upsampling_layers - 1):
            self.upsampling_layers.append(DecoderUpsamplingBlock(dims[i + 1], dims[i + 2]))

    def forward(self, x: Tensor, embedding: Tensor, encoder_outputs: list[Tensor]) -> Tensor:
        skips = encoder_outputs[::-1]
        x = self.upsampling_layers[0](x)
        if self.skip_connections_active:
            x = x + skips[1]
        for i in range(self.num_upsampling_layers - 1):
            for j in range(self.repetitions):
                x = self.blocks[i * self.repetitions + j](x, embedding)
            x = self.upsampling_layers[i + 1](x)
            if self.skip_connections_active:
                x = x + skips[i + 2]
        return self.decompress(x)
