"""FNO 3-D — the Fourier Neural Operator on the voxel stream.

Ported from https://github.com/neuraloperator/neuraloperator @ 00b7d86 (`neuralop/models/fno.py`
and `neuralop/layers/{spectral_convolution,fno_block,channel_mlp,skip_connections,padding,embeddings}.py`),
MIT License, Copyright (c) 2023 NeuralOperator developers. Papers: Kovachki, Li, Liu,
Azizzadenesheli, Bhattacharya, Stuart, Anandkumar, "Neural Operator: Learning Maps Between
Function Spaces", JMLR 24(89) 2023 (https://arxiv.org/abs/2108.08481) — the framework, its
eq. 26 `u = F^-1(R . F(v))` is `SpectralConv3d` — and Li et al., "Fourier Neural Operator for
Parametric Partial Differential Equations", ICLR 2021 (https://arxiv.org/abs/2010.08895) — the
architecture the library implements. Upstream study: `util/archive/docs/sketch/fno.md`.

The kernel integral of a neural operator is a pointwise multiplication in frequency space:
`rfftn` the lifted field, keep a centred block of `modes` Fourier coefficients per axis,
multiply by a dense complex weight `(in, out, m0, m1, m2//2+1)`, scatter back into a zero
spectrum and invert. The weights index *modes*, not voxels, so the same model applies to a rock
of any resolution — the discretisation invariance of the paper.

Changes from upstream (this is a modified copy, not the original):
- the three FFT axes are the three *spatial* axes (D, H, W) of one rock. This is not the paper's
  FNO-3D, which puts time on the third axis and maps initial frames to a whole trajectory; time
  stepping is the task's job here, exactly as for `unet3d`;
- `residual=True` (default) adds the most recent input frame and zero-initialises the last
  projection layer, so an untrained model is exactly persistence — the same start as `unet3d`;
- normalisation stays in the data layer (`FieldSpec` offset/scale); upstream's
  `UnitGaussianNormalizer` over input and output has no counterpart here;
- the dense path only: tensor factorisation (TFNO/`tltorch`), `separable`, `rank` and
  `implementation` are dropped, so the contraction is one `torch.einsum` and no dependency is added;
- dropped as unused: `complex_data`, half/mixed `fno_block_precision`, `stabilizer`,
  `resolution_scaling_factor` and per-block `output_shape`, `max_n_modes` incremental-mode
  training, `preactivation`, `ada_in`/`group_norm`/`instance_norm` (upstream's default is `None`
  anyway), upstream's `BaseModel` metaclass and its checkpoint helpers;
- `domain_padding` is a single fraction, not a per-axis list, and always symmetric;
- `n_modes` larger than the grid raises here instead of being silently clipped to the spectrum;
- `einops`/`opt_einsum`/`tltorch` are not used: `torch.einsum` and reshape/permute only.

Everything else follows upstream: `fft_norm="forward"`, `fftshift` on every axis but the
half-spectrum one, the centred mode block, weights `~ N(0, sqrt(2 / (in + out)))`, a physical-space
bias, the `linear` (pointwise, bias-free) FNO skip, the channel MLP with expansion 0.5 and its
soft-gating skip, GELU everywhere except after the last block, `ChannelMLP` lifting and projection
with ratio 2, the symmetric zero `domain_padding`, and the grid positional embedding on `[0, 1)`.

Verified against upstream by copying weights parameter for parameter into
`neuralop.models.FNO` and comparing outputs: **max|diff| = 0** on random input, with and without
domain padding (`util/archive/docs/sketch/fno.md` §5).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..registry import MODELS
from .baselines import recent_slice

SPATIAL = 3


def grid_embedding(x: Tensor) -> Tensor:
    """Append one channel per axis holding `[0, 1)` coordinates (upstream `GridEmbeddingND`)."""
    axes = [torch.linspace(0.0, 1.0, n + 1, device=x.device, dtype=x.dtype)[:-1] for n in x.shape[2:]]
    grid = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack(grid).unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
    return torch.cat([x, coords], dim=1)


class ChannelMLP(nn.Module):
    """Pointwise MLP over channels: `n_layers` 1x1 convolutions on the flattened spatial axes."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, n_layers: int = 2) -> None:
        super().__init__()
        widths = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
        self.fcs = nn.ModuleList([nn.Conv1d(widths[i], widths[i + 1], 1) for i in range(n_layers)])

    def forward(self, x: Tensor) -> Tensor:
        b, _, *spatial = x.shape
        h = x.reshape(b, x.shape[1], -1)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if i < len(self.fcs) - 1:
                h = F.gelu(h)
        return h.reshape(b, h.shape[1], *spatial)


class SoftGating(nn.Module):
    """`x * w` with one weight per channel (upstream's default channel-MLP skip)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x


class SpectralConv3d(nn.Module):
    """The Fourier layer: `F^-1(R . F(v))` over a centred block of `modes` coefficients per axis."""

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, int, int], bias: bool = True) -> None:
        super().__init__()
        if len(modes) != SPATIAL or any(m < 1 for m in modes):
            raise ValueError(f"modes must be three positive integers, got {modes}")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes = tuple(int(m) for m in modes)
        # the real FFT makes the last axis a half spectrum, so it holds m // 2 + 1 coefficients
        shape = (in_channels, out_channels, self.modes[0], self.modes[1], self.modes[2] // 2 + 1)
        scale = (2 / (in_channels + out_channels)) ** 0.5
        self.weight = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.bias = nn.Parameter(scale * torch.randn(out_channels, 1, 1, 1)) if bias else None

    def _slices(self, sizes: tuple[int, int, int]) -> tuple[slice, ...]:
        """Batch, channel, then the centred mode block of each shifted axis and the half spectrum."""
        out: list[slice] = [slice(None), slice(None)]
        for axis in (0, 1):
            kept, n = self.weight.shape[2 + axis], sizes[axis]
            centre = n // 2  # after fftshift the zero frequency sits here
            out.append(slice(centre - kept // 2, centre + kept // 2 + kept % 2))
        kept_last, half = self.weight.shape[-1], sizes[2] // 2 + 1
        out.append(slice(None, kept_last) if kept_last < half else slice(None))
        return tuple(out)

    def forward(self, x: Tensor) -> Tensor:
        sizes = tuple(x.shape[2:])
        if any(m > n for m, n in zip(self.modes, sizes, strict=True)):
            raise ValueError(f"modes {self.modes} exceed the grid {sizes}; keep every mode below its axis length")
        f = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="forward")
        f = torch.fft.fftshift(f, dim=(-3, -2))  # not the last axis: the real FFT already halved it
        out = torch.zeros(x.shape[0], self.out_channels, *f.shape[2:], device=x.device, dtype=f.dtype)
        block = self._slices(sizes)
        out[block] = torch.einsum("bixyz,ioxyz->boxyz", f[block], self.weight)
        out = torch.fft.ifftshift(out, dim=(-3, -2))
        # irfft needs a Hermitian-symmetric spectrum; cuFFT does not always enforce it and the
        # artefact shows up as lines, so zero the imaginary part where symmetry demands it
        out = torch.fft.ifftn(out, s=sizes[:2], dim=(-3, -2), norm="forward")
        out[..., 0].imag.zero_()
        if sizes[2] % 2 == 0:
            out[..., -1].imag.zero_()
        y = torch.fft.irfft(out, n=sizes[2], dim=-1, norm="forward")
        return y if self.bias is None else y + self.bias


class FNOBlock(nn.Module):
    """`GELU(SpectralConv(x) + Wx)` then `GELU(ChannelMLP(.) + SoftGating(.))`, activations optional."""

    def __init__(self, channels: int, modes: tuple[int, int, int], mlp_expansion: float = 0.5) -> None:
        super().__init__()
        self.conv = SpectralConv3d(channels, channels, modes)
        self.skip = nn.Conv3d(channels, channels, 1, bias=False)  # upstream `fno_skip="linear"`
        self.mlp = ChannelMLP(channels, channels, max(1, round(channels * mlp_expansion)))
        self.gate = SoftGating(channels)

    def forward(self, x: Tensor, activate: bool = True) -> Tensor:
        skip = self.skip(x)
        gated = self.gate(x)
        h = self.conv(x) + skip
        if activate:
            h = F.gelu(h)
        h = self.mlp(h) + gated
        return F.gelu(h) if activate else h


@MODELS.register("fno3d")
class FNO3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        modes: tuple[int, int, int] | list[int] = (16, 16, 16),
        hidden_channels: int = 32,
        n_layers: int = 4,
        lifting_ratio: float = 2.0,
        projection_ratio: float = 2.0,
        mlp_expansion: float = 0.5,
        domain_padding: float = 0.0625,
        positional_embedding: bool = True,
        residual: bool = True,
        recent_channel: int = -1,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers}")
        if not 0.0 <= domain_padding < 0.5:
            raise ValueError(f"domain_padding is a fraction of each axis in [0, 0.5), got {domain_padding}")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes = tuple(int(m) for m in modes)
        self.domain_padding = float(domain_padding)
        self.positional_embedding = positional_embedding
        self.residual = residual
        self.recent = recent_slice(in_channels, out_channels, recent_channel)

        lifted = in_channels + (SPATIAL if positional_embedding else 0)
        self.lifting = ChannelMLP(lifted, hidden_channels, round(lifting_ratio * hidden_channels))
        self.blocks = nn.ModuleList([FNOBlock(hidden_channels, self.modes, mlp_expansion) for _ in range(n_layers)])
        self.projection = ChannelMLP(hidden_channels, out_channels, round(projection_ratio * hidden_channels))
        nn.init.zeros_(self.projection.fcs[-1].weight)
        nn.init.zeros_(self.projection.fcs[-1].bias)

    def _pad(self, x: Tensor) -> tuple[Tensor, tuple[slice, ...]]:
        """Symmetric zero padding of `domain_padding` of each axis: the rock is periodic on none."""
        if not self.domain_padding:
            return x, (Ellipsis,)
        amounts = [round(self.domain_padding * n) for n in x.shape[2:]]
        pad: list[int] = []
        for amount in reversed(amounts):  # F.pad consumes the last axis first
            pad += [amount, amount]
        unpad = (Ellipsis, *(slice(a, -a) if a else slice(None) for a in amounts))
        return F.pad(x, pad, mode="constant"), unpad

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 + SPATIAL:
            raise ValueError(f"expected a (B, C, D, H, W) input, got shape {tuple(x.shape)}")
        h = grid_embedding(x) if self.positional_embedding else x
        h = self.lifting(h)
        h, unpad = self._pad(h)
        for i, block in enumerate(self.blocks):
            h = block(h, activate=i < len(self.blocks) - 1)
        h = h[unpad]
        out = self.projection(h)
        if self.residual:
            out = out + x[:, self.recent]
        return out
