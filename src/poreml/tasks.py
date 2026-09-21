"""Benchmark tasks.

A task owns three decisions: what a sample looks like, how many input channels a model
must accept, and which criteria it is scored on. Adding a task is one class plus one
decorator — nothing in the core changes.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import Dataset

from .config import TaskConfig
from .data import ConditionSpec, FieldSpec, RunRef, WindowDataset
from .metrics.spec import MetricSpec
from .points import DEFAULT_RADII, PointWindowDataset
from .registry import TASKS

REPRESENTATIONS = ("voxel", "points")


class PointsParams(BaseModel):
    """`task.params.points`: the point stream's knobs. Unknown keys raise."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    geometry_radii: tuple[int, ...] = Field(default=DEFAULT_RADII, min_length=0)
    train_points: int | None = Field(default=None, ge=1, description="Random points per training window; None = all")
    wall_channel: bool = Field(
        default=False,
        description="Feed the binary touches-solid flag as the first feature channel (the voxel stream's rock-mask "
        "channel 0, mirrored). Project default: every shipped point config sets it true — the default is false so a "
        "checkpoint's saved config keeps rebuilding exactly the inputs it was trained on",
    )


def parse_fields(params: dict) -> tuple[FieldSpec, ...]:
    """`fields: [{name, offset, scale, fill}, ...]`; without it the task predicts `phi` alone."""
    if "field" in params:
        raise ValueError(f"task params: `field` was retired; write `fields: [{{name: {params['field']}}}]`")
    if "fields" not in params:
        return (FieldSpec(name="phi"),)
    fields = tuple(FieldSpec.model_validate(f) for f in params["fields"])
    if not fields:
        raise ValueError("task params: `fields` must name at least one field")
    return fields


@runtime_checkable
class Task(Protocol):
    """What a sample is, how many channels a model must accept, and how it is scored.

    Channel convention: a task's dataset stacks input channels as static geometry first,
    then the `history` frames oldest to newest — each history frame contributes
    `out_channels` channels, so the most recent frame is the last `out_channels` channels
    — and the target is the frame after it. Models are allowed to rely on this —
    `persistence` returns the last channels as its prediction. A task that departs from
    the layout must document it and configure the models that care (for persistence,
    `params: {recent_channel: ...}` in the model config); otherwise the baseline goes on
    returning channels that are no longer a frame, with no error to notice it by.
    """

    name: str

    def dataset(self, runs: Sequence[RunRef], stride: int | None = None, future: int = 1) -> Dataset:
        """Build the sample set for these runs; `future > 1` targets that many frames ahead (push-forward)."""
        ...

    @property
    def in_channels(self) -> int:
        """Channel count a model must accept."""
        ...

    @property
    def out_channels(self) -> int:
        """Channels a model must produce: one per target channel."""
        ...

    @property
    def target_channels(self) -> tuple[str, ...]:
        """Names of the target's channels, in order; metric specs address channels by these."""
        ...

    @property
    def metrics(self) -> tuple[MetricSpec, ...]:
        """Criteria this task is scored on by default; the first selects the best checkpoint."""
        ...


@TASKS.register("next_frame")
class NextFrameTask:
    """Predict the phase field one saved frame ahead.

    A placeholder that proves the wiring end to end and serves as the template real task
    definitions are written against.
    """

    name = "next_frame"

    def __init__(self, cfg: TaskConfig, representation: str = "voxel") -> None:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unknown representation {representation!r}; expected one of {REPRESENTATIONS}")
        self.cfg = cfg
        self.representation = representation
        self.history = cfg.history
        self.fields = parse_fields(cfg.params)
        self.stride = cfg.params.get("stride", 1)
        # Score the rock, not the solver's inlet reservoir / porous plate / outlet buffer.
        self.roi = cfg.params.get("roi", "rock")
        if "points" in cfg.params and representation != "points":
            raise ValueError("task params `points` were given but the model consumes voxels; drop them or use a point model")
        self.points = PointsParams.model_validate(cfg.params.get("points", {}))
        self.conditions = tuple(ConditionSpec.model_validate(c) for c in cfg.params.get("conditions", []))

    def dataset(
        self, runs: Sequence[RunRef], stride: int | None = None, training: bool = False, future: int = 1
    ) -> WindowDataset:
        """`stride` overrides the configured window stride (the periodic evaluation subsamples with
        it); `training=True` enables the point stream's `train_points` subsampling; `future` is
        how many frames ahead the target stacks (push-forward training needs `steps + 1`)."""
        kwargs = dict(
            history=self.history,
            fields=self.fields,
            stride=self.stride if stride is None else stride,
            roi=self.roi,
            conditions=self.conditions,
            future=future,
        )
        if self.representation == "points":
            return PointWindowDataset(
                runs,
                **kwargs,
                geometry_radii=self.points.geometry_radii,
                train_points=self.points.train_points if training else None,
                wall_channel=self.points.wall_channel,
            )
        return WindowDataset(runs, **kwargs)

    @property
    def target_channels(self) -> tuple[str, ...]:
        return tuple(c for f in self.fields for c in f.channels)

    @property
    def out_channels(self) -> int:
        return len(self.target_channels)

    @property
    def in_channels(self) -> int:
        static = (
            int(self.points.wall_channel) + len(self.points.geometry_radii) if self.representation == "points" else 1
        )  # points: optional wall flag + fractions; voxels: the rock mask
        # Static channels (geometry, then one constant channel per condition) plus F per history frame.
        return static + len(self.conditions) + self.history * self.out_channels

    @property
    def metrics(self) -> tuple[MetricSpec, ...]:
        return (MetricSpec(error="mae"), MetricSpec(error="iou"), MetricSpec(descriptor="saturation", error="abs_err"))

    def __repr__(self) -> str:
        return (
            f"NextFrameTask(history={self.history}, channels={self.target_channels!r}, roi={self.roi!r}, "
            f"representation={self.representation!r})"
        )


def build_task(cfg: TaskConfig, representation: str = "voxel") -> Task:
    """`representation` is what the configured model consumes (`models.representation_of`)."""
    return TASKS.get(cfg.name)(cfg, representation=representation)
