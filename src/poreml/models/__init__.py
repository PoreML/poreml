"""Models: the zoo behind `MODELS`.

A model is any `nn.Module` mapping `(B, in_channels, *spatial)` to `(B, out_channels,
*spatial)`, constructed as `cls(in_channels=..., out_channels=..., **params)`.
Registering one is a decorator in its own file plus an import here; nothing in
training, evaluation, rollout or rendering changes — that is the whole point of the
case procedure.

A model class may declare `representation = "points"` to be fed `points.Points` batches
(positions + features on the pore voxels, see `poreml.points`) instead of
`(B, in_channels, *spatial)` tensors; the task builds the matching dataset
(`representation_of` is how train/eval find out).
"""

from torch import nn

from ..config import ModelConfig
from ..registry import MODELS
from . import abupt as abupt  # noqa: F401
from . import baselines as baselines  # noqa: F401  (registration side effect)
from . import fno as fno  # noqa: F401
from . import p3d as p3d  # noqa: F401
from . import transolver as transolver  # noqa: F401
from . import unet as unet  # noqa: F401
from .abupt import ABUPT
from .baselines import Persistence
from .fno import FNO3D
from .p3d import P3D
from .transolver import Transolver
from .unet import UNet3D

__all__ = ["ABUPT", "FNO3D", "MODELS", "P3D", "Persistence", "Transolver", "UNet3D", "build_model", "representation_of"]


def build_model(cfg: ModelConfig, in_channels: int, out_channels: int = 1) -> nn.Module:
    return MODELS.get(cfg.name)(in_channels=in_channels, out_channels=out_channels, **cfg.params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def representation_of(cfg: ModelConfig) -> str:
    """Which data stream the configured model consumes: "voxel" (default) or "points"."""
    return getattr(MODELS.get(cfg.name), "representation", "voxel")


def precision_of(cfg: ModelConfig) -> str:
    """The precision a model trains *and infers* in: `model.precision` when the config states
    it, else the model class's `precision` attribute, else the project default `tf32`
    (2026-09-10: fp32 storage, TF32 tensor-core matmuls — 5x faster inference and 2-4x faster
    training steps for Transolver at a 2e-5 deviation on phi; the convolutional models already
    ran TF32 convolutions under torch's cuDNN default). AB-UPT declares bf16 (its upstream
    trains in mixed precision; here it is 5x faster with no loss change, and 7x at inference
    over every pore voxel). `precision.py` applies it; `fp32` reproduces a pre-2026-09-10 run."""
    if cfg.precision is not None:
        return cfg.precision
    return getattr(MODELS.get(cfg.name), "precision", "tf32")
