"""Metric configuration entry.

A metric is `error(descriptor(pred), descriptor(target))`. The entry names both halves
plus the knobs the descriptor needs. It is deliberately a small closed schema rather than
a free-form params dict so a typo is a validation error, not a silently ignored key.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VOXELS = "voxels"


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error: str = Field(description="Registered error, e.g. mae, abs_err, rel_err, iou, w1, pred, target")
    descriptor: str = Field(default=VOXELS, description="Registered descriptor; 'voxels' scores the field itself")
    channel: str | int | None = Field(default=None, description="Target channel name or index; None means all channels")
    phase: Literal["nw", "w"] = Field(default="nw", description="Fluid the descriptor measures")
    bins: int | None = Field(default=None, ge=1, description="Distributions (hist, curvature_hist): number of bins")
    range: tuple[float, float] | None = Field(default=None, description="Distributions: (lo, hi) of the fixed bin edges")
    name: str | None = Field(default=None, description="Override for the results key")
    eval_only: bool = Field(
        default=False,
        description="Skip in per-epoch validation (too slow to score every epoch); `eval` and `rollout` still score it",
    )

    @property
    def descriptor_key(self) -> str:
        """`<descriptor>@<channel>@w`: what the descriptor half measures, shared by every error on it."""
        key = self.descriptor
        if self.channel is not None:
            key += f"@{self.channel}"
        if self.phase == "w":
            key += "@w"
        return key

    @property
    def key(self) -> str:
        """Results key: canonical `<descriptor>/<error>@<channel>@w`, or `name` if set."""
        if self.name is not None:
            return self.name
        key = self.error if self.descriptor == VOXELS else f"{self.descriptor}/{self.error}"
        if self.channel is not None:
            key += f"@{self.channel}"
        if self.phase == "w":
            key += "@w"
        return key
