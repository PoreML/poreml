"""Precision is a model property.

`models.precision_of(cfg.model)` says what a model runs in — `model.precision` when the config
states it, else the class's `precision` attribute, else the project default — and `configure`
applies it to the process:

- `tf32` (project default, 2026-09-10): fp32 weights, activations, loss and metrics; matmuls on
  the tensor cores with 10-bit-mantissa inputs and fp32 accumulation
  (`torch.set_float32_matmul_precision("high")`). Measured on an H200 with the drainage
  Transolver checkpoint over 494k pore voxels: 0.313 -> 0.058 s per forward, one-step mae@phi
  equal to five digits, |fp32 - tf32| mean 2e-5 (max 4e-3) on phi; a training step 0.87 -> 0.49 s
  with activation checkpointing, 0.23 s without. cuDNN convolutions have used TF32 under torch's
  own default all along, so the voxel models change nothing here; FNO's spectral products and
  the point models' linear layers do.
- `bf16` (AB-UPT): the forward runs under bf16 autocast, weights, optimiser and loss stay fp32,
  and `inference_model` hands fp32 to the metrics. Measured 2026-09-08 with the gdl AB-UPT
  checkpoint over 592k pore voxels: 7.95 -> 1.08 s per forward, |fp32 - bf16| mean 2e-4 on phi.
  Transolver under bf16 measured 0.086 s per forward with a 2e-4 deviation — slower and ten
  times further from fp32 than tf32, which is why tf32 is the default and bf16 AB-UPT's own.
- `fp32`: torch's defaults (full-precision matmuls) — what every run before 2026-09-10 got.
  State it in a config to reproduce one.

Training autocasts its forward pass under the precision; since 2026-09-08 so do eval, rollout
and the per-epoch selection pass, through `inference_model`. A benchmark number therefore
depends on the *model's* precision, recorded as `precision` in every results and rollout JSON,
in `run_meta.json` under `model.precision`, and per segment under `run.segments`.
"""

from __future__ import annotations

import torch
from torch import nn

PRECISIONS = ("fp32", "bf16", "tf32")


def configure(precision: str) -> None:
    """Apply `precision` to this process: TF32 tensor-core matmuls for tf32, torch's full-precision
    default otherwise (bf16 models autocast their forward and keep the rest at full precision)."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")


def autocast(precision: str, device: str | torch.device) -> torch.autocast:
    """The autocast context a forward pass under `precision` runs in (disabled unless bf16)."""
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=precision == "bf16")


class AutocastInference(nn.Module):
    """`model` whose forward runs under bf16 autocast and returns fp32 — what the metrics,
    the rollout feedback and the frame store expect."""

    def __init__(self, model: nn.Module, precision: str, device: str | torch.device):
        super().__init__()
        self.model = model
        self.precision = precision
        self.device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def forward(self, x):
        with autocast(self.precision, self.device_type):
            out = self.model(x)
        return out.float()


def inference_model(model: nn.Module, precision: str, device: str | torch.device) -> nn.Module:
    """`model` itself for fp32 and tf32 (the process is configured for it); wrapped for bf16 so
    every forward runs under autocast and returns fp32."""
    configure(precision)
    if precision != "bf16":
        return model
    return AutocastInference(model, precision, device)
