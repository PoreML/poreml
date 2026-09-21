"""P3D — a hybrid CNN / windowed-transformer 3-D surrogate, on the voxel stream.

Ported from https://github.com/tum-pbs/P3D @ ebf216a (`src/p3d_surrogate/models/p3d.py`,
`WrapperModel3D` + `P3DBackbone`), Apache License 2.0 per the repository's `LICENSE` (its
`pyproject.toml` and README badge say MIT — `NOTICE` records both). Paper: Holzschuh, Kohl,
Redinger, Thuerey, "P3D: Scalable Neural Surrogates for High-Resolution 3D Physics Simulations
with Global Context", 2025 (https://arxiv.org/abs/2509.10186). Blocks live in `p3d_blocks.py`;
upstream study in `util/archive/docs/sketch/p3d.md`.

The shape of it: a convolutional stem compresses the volume by 4 and hands it to a U-shaped
transformer whose stages are windowed 3-D self-attention (cosine attention, a learned
relative-position bias from an MLP on log-spaced coordinates, DiT-style modulation from a
conditioning embedding), then a convolutional decoder with additive skips brings it back to full
resolution. Convolutions carry the local structure, attention the long range, and the volume is
only ever attended inside `window_size^3` windows — that is what makes 3-D affordable.

Changes from upstream (this is a modified copy, not the original):
- **deterministic only.** The flow-matching objective, the Euler ODE sampler and the context
  (sequence) model that fuses patches into a global solution are not ported: poreml trains
  one-step deterministic surrogates, and the probabilistic mode would need a diffusion loss and a
  stochastic evaluation protocol in the core, not a model file. The conditioning path is kept
  intact and fed a zero timestep, so wiring a diffusion time or run conditions (M, theta) in later
  is a one-line change;
- **no region partitioning or mending.** `partition_size=1`, the identity case: the solver's 128^3 ROI is
  one region, and the patch-fusion machinery is P3D's answer to 512^3 domains;
- class labels are dropped: the 1000-row embedding table becomes one learned vector per level,
  which is exactly upstream with a fixed label and no class dropout;
- `residual=True` (default) adds the most recent input frame and zero-initialises the decoder's
  final projection, so an untrained model is exactly persistence — the same start as `unet3d`;
- `periodic=False` only (the rock ROI is periodic on no axis), so upstream's `Conv3dVariablePad`
  and its circular padding modes are not needed;
- dropped as unused: `diffusers`' `ModelMixin`/`ConfigMixin` config plumbing and `Output3D`
  dataclasses, `DropPath` (upstream instantiates it at rate 0), the `v1` attention branch,
  `attn_drop`/`drop`/dropout (all 0 upstream), `PlaceholderModel`, `WrapperModel3DPatch`,
  `switch_to_deploy` caching, and `max_hidden_size` clamping (no preset reaches it);
- window shifting is not ported: upstream's presets pass `shift=False` and the paper reports
  removing shifted windows "for improved computational efficiency" with no measurable loss;
- `einops` and `numpy` are not used; the upstream in-place `residuals_wrapper[-1] += state` is
  dropped because nothing reads that tensor afterwards.

Sizes are upstream's presets: S 64/[32,32,64], B 128/[64,128,128], L 256/[128,256,256],
XL 512/[256,512,512] (upstream's XL preset points at a stale module path and cannot be built as
shipped; the numbers are taken from it verbatim).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..registry import MODELS
from .baselines import recent_slice
from .p3d_blocks import (
    ConditionedDecoder3D,
    ConditionedEncoder3D,
    Conditioning,
    Downsample,
    FinalLayer,
    P3DStage,
    TimestepEmbedder,
    Upsample,
)

# hidden_size, num_heads, feature_embedding_dim, num_groups, repetitions — upstream P3D_{S,B,L,XL}
PRESETS = {
    "s": (64, 4, (32, 32, 64), 16, 2),
    "b": (128, 4, (64, 128, 128), 32, 2),
    "l": (256, 8, (128, 256, 256), 32, 2),
    "xl": (512, 8, (256, 512, 512), 64, 4),
}


class P3DBackbone(nn.Module):
    """U-shaped stack of windowed-attention stages, one conditioning embedding per level."""

    def __init__(
        self,
        hidden_size: int,
        depth: tuple[int, ...] = (2, 2, 2, 2, 2),
        num_heads: int = 4,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        skip_connections_active: bool = True,
    ) -> None:
        super().__init__()
        if len(depth) % 2 == 0:
            raise ValueError(f"depth must have an odd length (encoder, latent, decoder), got {depth}")
        self.num_encoder_layers = len(depth) // 2
        self.hidden_size = hidden_size
        self.latent_size = hidden_size * 2**self.num_encoder_layers
        self.skip_connections_active = skip_connections_active
        stage = {"num_heads": num_heads, "window_size": window_size, "mlp_ratio": mlp_ratio}

        for i in range(self.num_encoder_layers + 1):
            dim = hidden_size * 2**i
            setattr(self, f"t_embedder_{i}", TimestepEmbedder(dim))
            setattr(self, f"param_embedder_{i}", TimestepEmbedder(dim))
            setattr(self, f"class_vector_{i}", nn.Parameter(torch.randn(dim) * 0.02))

        for i in range(self.num_encoder_layers):
            dim = hidden_size * 2**i
            setattr(self, f"encoder_level_{i}", P3DStage(dim=dim, depth=depth[i], **stage))
            setattr(self, f"down{i}_{i + 1}", Downsample(dim))

        self.latent = P3DStage(dim=self.latent_size, depth=depth[self.num_encoder_layers], **stage)

        reduce = 1.5 if skip_connections_active else 0.5
        self.up1_0 = Upsample(2 * hidden_size)
        self.reduce_chan_level0 = nn.Conv3d(int(reduce * hidden_size), 2 * hidden_size, 1)
        self.decoder_level_0 = P3DStage(dim=2 * hidden_size, depth=depth[self.num_encoder_layers + 1], **stage)
        for i in range(1, self.num_encoder_layers):
            dim = hidden_size * 2**i
            setattr(self, f"up{i + 1}_{i}", Upsample(2 * dim))
            setattr(self, f"reduce_chan_level{i}", nn.Conv3d(int(reduce * dim), dim, 1))
            setattr(self, f"decoder_level_{i}", P3DStage(dim=dim, depth=depth[self.num_encoder_layers + i + 1], **stage))

        self.final_layer = FinalLayer(2 * hidden_size, hidden_size)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear | nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic)
        for i in range(self.num_encoder_layers):
            for name in (f"t_embedder_{i}", f"param_embedder_{i}"):
                embedder = getattr(self, name)
                nn.init.normal_(embedder.mlp[0].weight, std=0.02)
                nn.init.normal_(embedder.mlp[2].weight, std=0.02)
        for stage in self.stages():
            for block in stage.blocks:
                nn.init.constant_(block.adain_2.linear.weight, 0)
                nn.init.constant_(block.adain_2.linear.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def stages(self) -> list[nn.Module]:
        levels = [getattr(self, f"encoder_level_{i}") for i in range(self.num_encoder_layers)]
        return levels + [self.latent] + [getattr(self, f"decoder_level_{i}") for i in range(self.num_encoder_layers)]

    def embeddings(self, timestep: Tensor, parameter: Tensor) -> list[Tensor]:
        out = []
        for i in range(self.num_encoder_layers + 1):
            t = getattr(self, f"t_embedder_{i}")(timestep)
            p = getattr(self, f"param_embedder_{i}")(parameter)
            out.append(t + p + getattr(self, f"class_vector_{i}")[None])
        return out

    def forward(self, x: Tensor, timestep: Tensor, parameter: Tensor) -> Tensor:
        embeddings = self.embeddings(timestep, parameter)

        residuals = []
        for i, emb in enumerate(embeddings[:-1]):
            x = getattr(self, f"encoder_level_{i}")(x, emb)
            residuals.append(x)
            x = getattr(self, f"down{i}_{i + 1}")(x)
        x = self.latent(x, embeddings[-1])

        for i, (residual, emb) in enumerate(zip(residuals[1:][::-1], embeddings[1:-1][::-1], strict=True)):
            level = self.num_encoder_layers - i - 1
            x = getattr(self, f"up{level + 1}_{level}")(x)
            if self.skip_connections_active:
                x = torch.cat([x, residual], dim=1)
            x = getattr(self, f"reduce_chan_level{level}")(x)
            x = getattr(self, f"decoder_level_{level}")(x, emb)

        x = self.up1_0(x)
        if self.skip_connections_active:
            x = torch.cat([x, residuals[0]], dim=1)
        x = self.reduce_chan_level0(x)
        x = self.decoder_level_0(x, embeddings[1])
        return self.final_layer(x, embeddings[1])


@MODELS.register("p3d")
class P3D(nn.Module):
    representation = "voxel"

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        size: str = "s",
        depth: tuple[int, ...] | list[int] = (2, 2, 2, 2, 2),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        time_embedding_dim: int = 64,
        skip_connections: bool = True,
        residual: bool = True,
        recent_channel: int = -1,
    ) -> None:
        super().__init__()
        if size not in PRESETS:
            raise ValueError(f"unknown size {size!r}; expected one of {sorted(PRESETS)}")
        hidden_size, num_heads, feature_dims, num_groups, repetitions = PRESETS[size]
        self.size = size
        self.residual = residual
        self.recent = recent_slice(in_channels, out_channels, recent_channel)
        self.num_downsampling_layers = len(feature_dims) - 1
        self.backbone_levels = len(depth) // 2
        self.factor = 2 ** (self.num_downsampling_layers + self.backbone_levels)

        self.class_embedding = Conditioning(time_embedding_dim)
        self.encoder = ConditionedEncoder3D(
            in_channels=in_channels,
            feature_embedding_dim=feature_dims,
            num_downsampling_layers=self.num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            repetitions=repetitions,
            num_groups=num_groups,
        )
        self.backbone = P3DBackbone(
            hidden_size=hidden_size,
            depth=tuple(depth),
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            skip_connections_active=skip_connections,
        )
        if feature_dims[-1] != hidden_size:
            raise ValueError(f"preset {size!r} feeds {feature_dims[-1]} channels into a {hidden_size}-wide backbone")
        self.decoder = ConditionedDecoder3D(
            out_channels=out_channels,
            feature_embedding_dim=feature_dims[::-1],
            num_upsampling_layers=self.num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            features_first_layer=feature_dims[-1],
            repetitions=repetitions,
            num_groups=num_groups,
            skip_connections_active=skip_connections,
        )
        nn.init.zeros_(self.decoder.decompress.weight)
        nn.init.zeros_(self.decoder.decompress.bias)

    def forward(self, x: Tensor, timestep: Tensor | None = None, parameter: Tensor | None = None) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected a (B, C, D, H, W) input, got shape {tuple(x.shape)}")
        # A shape the pyramid cannot halve (underfill's 26x482x476 divides by no preset's
        # factor) is replicate-padded up to the next multiple and cropped back after —
        # a no-op for divisible domains, like fno's domain_padding.
        spatial = x.shape[2:]
        pads = [(-n) % self.factor for n in spatial]
        if any(pads):
            x = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]), mode="replicate")
        zeros = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        timestep = zeros if timestep is None else timestep
        parameter = zeros if parameter is None else parameter

        emb = self.class_embedding(timestep, parameter)
        residuals = self.encoder(x, emb)
        state = self.backbone(residuals[-1], timestep, parameter)
        out = self.decoder(state, emb, residuals)
        out = out + x[:, self.recent] if self.residual else out
        return out[:, :, : spatial[0], : spatial[1], : spatial[2]] if any(pads) else out
