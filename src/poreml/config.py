"""Run configuration.

Validation is strict: unknown keys are an error rather than a shrug, so a typo in a
config can never silently train something other than what was written down.
"""

from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .metrics.spec import MetricSpec


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    root: Path = Field(description="Directory holding campaign folders, e.g. data/case")
    campaign: str = Field(description="Campaign name: drainage, GDL, trapping, underfill")
    split: Path = Field(description="Path to the frozen split YAML")
    require_finished: bool = Field(
        default=True,
        description="Refuse a split naming a run the solver has not finished; its frame count can still change",
    )


class TaskConfig(_Strict):
    name: str
    history: int = Field(default=1, ge=1, description="Number of past frames fed to the model")
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    precision: Literal["fp32", "bf16", "tf32"] | None = Field(
        default=None,
        description="Precision the model trains and infers in: tf32 (project default since 2026-09-10) keeps fp32 "
        "storage and runs matmuls on the tensor cores; bf16 autocasts the forward pass (weights, optimiser, loss "
        "and every evaluation stay fp32); fp32 is torch's full-precision default, what runs before 2026-09-10 got. "
        "None: the model class's own default (`models.precision_of`) — bf16 for abupt, tf32 for every other model",
    )


class EvalConfig(_Strict):
    """Periodic full evaluation during training: the val split scored with *every* metric
    (including `eval_only` ones), rolled out, rendered, logged to `eval/metrics.csv` and
    plotted. Reporting only — the best checkpoint is still chosen by the per-epoch pass."""

    every: int = Field(ge=1, description="Epochs between evaluations; the last epoch is always evaluated")
    stride: int = Field(default=1, ge=1, description="Score every stride-th val window (curvature is ~1.5 s per frame)")
    rollout: bool = Field(default=True, description="Roll every val run out with the `rollout:` protocol")
    rollout_stride: int = Field(
        default=64,
        ge=1,
        description="Score every stride-th rollout step (plus each run's final step, so the at-horizon value always "
        "exists); unscored steps are null. The default 64 scores the at-horizon step only; 1 restores per-step "
        "scoring. Periodic eval only — `poreml rollout` always scores every step",
    )
    render: bool = Field(default=True, description="Truth | prediction GIF of the first val run (needs the viz group)")


class PushForwardConfig(_Strict):
    """The push-forward trick (Brandstetter et al. 2022, as trained in BubbleML): on a pushed
    batch the model steps `steps` times on its own output without gradients, then takes one
    graded step scored against the true frame `steps + 1` ahead — it learns to correct the
    errors it makes during a rollout. The other batches get Gaussian noise on their history
    channels (pore voxels only) instead. Whether a batch is pushed is drawn per step with a
    probability that ramps linearly from `prob_start` to `prob_end` over the run."""

    steps: int = Field(default=1, ge=1, description="Ungraded steps on the model's own output before the graded one")
    prob_start: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability of pushing at the first step")
    prob_end: float = Field(default=1.0, ge=0.0, le=1.0, description="Probability of pushing at the last step")
    noise: float = Field(
        default=0.01, ge=0.0, description="Std of the Gaussian noise on the history channels of a batch not pushed; 0 = none"
    )


class TrainConfig(_Strict):
    epochs: int = Field(default=1, ge=0)
    batch_size: int = Field(default=1, ge=1)
    val_stride: int | None = Field(
        default=8,
        ge=1,
        description="Score every stride-th val window in the per-epoch selection pass (a fixed, evenly spaced subset "
        "of each run's timeline; a full AB-UPT pass on a b99 val split costs ~5 h per epoch). None: the task's own "
        "stride — the full pass every run before the knob got",
    )
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(
        default=0.01,
        ge=0,
        description="AdamW decoupled weight decay; the default is torch's, which every run before the knob got",
    )
    grad_clip: float | None = Field(default=None, gt=0, description="Clip the global gradient norm before each step")
    lr_schedule: Literal["constant", "onecycle", "steplr"] = Field(
        default="constant",
        description="onecycle: OneCycleLR(max_lr=lr) over epochs x batches per epoch, stepped per batch; "
        "steplr: multiply the lr by lr_gamma every lr_step_size epochs (neuraloperator's FNO recipe), stepped per epoch",
    )
    lr_step_size: int = Field(default=100, ge=1, description="steplr only: epochs between lr decays")
    lr_gamma: float = Field(default=0.5, gt=0, le=1, description="steplr only: lr decay factor")
    loss: Literal["mse", "h1"] = Field(
        default="mse",
        description="Training loss: mse = masked MSE (all runs before the knob); h1 = masked relative H1 (Sobolev) "
        "norm with finite-difference gradients, upstream neuraloperator's FNO training loss — voxel stream only",
    )
    device: str = "auto"
    num_workers: int = Field(default=0, ge=0)
    max_batches: int | None = Field(default=None, ge=1, description="Cap batches per epoch; None means all")
    log_every: int = Field(default=100, ge=1, description="Batches between train_log.csv rows and log lines")
    checkpoint_every: int | None = Field(
        default=1800,
        ge=0,
        description="Seconds between mid-epoch writes of ckpts/resume.pt (model, optimiser, scheduler, position, "
        "RNG); it is also written at every epoch end and on SIGTERM. None: epoch ends and SIGTERM only",
    )
    eval: EvalConfig | None = Field(default=None, description="Periodic full evaluation; None means never")
    init_from: Path | None = Field(
        default=None,
        description="Checkpoint (a run's ckpts/best.pt or last.pt) whose weights the model starts from — a fine-tune. "
        "The optimiser, scheduler and epoch count start fresh; ignored when resuming a run",
    )
    push_forward: PushForwardConfig | None = Field(
        default=None, description="Push-forward training (see PushForwardConfig); None trains one-step, teacher-forced"
    )


class RolloutConfig(_Strict):
    horizon: int = Field(default=64, ge=1, description="Autonomous rollout length in saved frames")
    start: int | None = Field(
        default=None, ge=0, description="Time index of the last true input frame; None means `start_fraction` of the run"
    )
    start_fraction: float = Field(
        default=0.25,
        ge=0.0,
        lt=1.0,
        description="When start is None: start at this fraction of each run's frames (never before history - 1)",
    )
    keyframes: int | None = Field(
        default=None,
        ge=2,
        description=(
            "Rollout steps whose predictions are stored and scored with the eval_only metrics: the first step, the last "
            "and keyframes - 2 evenly spaced between. None = the scored steps (every rollout_stride-th plus the last)."
        ),
    )


class Config(_Strict):
    name: str
    data: DataConfig
    task: TaskConfig
    model: ModelConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    metrics: list[MetricSpec] | None = Field(
        default=None, min_length=1, description="None means use the task's declared defaults"
    )
    out_dir: Path = Path("runs")
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        payload = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(payload)

    def to_yaml(self, path: str | Path) -> None:
        payload = self.model_dump(mode="json")
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))

    def updated(self, **sections: Any) -> "Config":
        """A copy with named top-level sections shallowly updated and re-validated.

        `cfg.updated(data={"campaign": "GDL", "split": "splits/shift_GDL.yaml"})` changes only
        the named keys; task, model, metrics and rollout survive byte-identical — which is what
        scoring a checkpoint under a different split needs. Strictness is preserved: an unknown
        section or key raises, silently feeding a model something it was not trained on does not.
        """
        payload = self.model_dump(mode="json")
        for section, changes in sections.items():
            if section not in payload:
                raise KeyError(f"unknown config section {section!r}; sections are {sorted(payload)}")
            payload[section] = {**payload[section], **changes} if isinstance(changes, dict) else changes
        return type(self).model_validate(payload)


def resolve_device(spec: str) -> str:
    """Map the device spec to a concrete device. "auto" prefers CUDA when present."""
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"
