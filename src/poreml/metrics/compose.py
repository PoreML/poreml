"""Turn metric specs into validated callables and evaluate them per sample.

`resolve` runs before a run directory exists: every way a spec can be wrong — a typo,
a channel the task does not produce, an error that cannot read what the descriptor
returns — is raised here rather than at the end of epoch 0.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from ..registry import DESCRIPTORS, ERRORS
from . import distributions as distributions  # noqa: F401  (registration side effect)
from . import errors as errors  # noqa: F401
from . import trace
from .descriptors import VOXELS_KIND, check_field, shared_meshes
from .spec import MetricSpec

_PARAM_FIELDS = ("phase", "bins", "range")
# metrics.jsonl rows carry these alongside the metrics, so a metric may not claim them.
_RESERVED_NAMES = ("epoch", "train_loss", "val_loss", "loss")

Channel = int | tuple[int, int] | None  # one index, a half-open range, or every channel


@dataclass(frozen=True)
class ResolvedMetric:
    name: str
    spec: MetricSpec
    descriptor: Callable
    error: Callable
    channel: Channel
    params: dict[str, Any]

    @property
    def kind(self) -> str:
        return self.descriptor.kind

    @property
    def higher_is_better(self) -> bool:
        return self.error.higher_is_better


def _resolve_channel(spec: MetricSpec, target_channels: Sequence[str]) -> Channel:
    if spec.channel is None:
        return None
    if isinstance(spec.channel, int):
        if not 0 <= spec.channel < len(target_channels):
            raise ValueError(
                f"{spec.key}: channel index {spec.channel} out of range for target channels {tuple(target_channels)}"
            )
        return spec.channel
    names = list(target_channels)
    if spec.channel in names:
        return names.index(spec.channel)
    # A vector field: `u` names the contiguous components ux, uy, uz (data.FieldSpec.channels).
    components = [f"{spec.channel}{axis}" for axis in "xyz"]
    if all(c in names for c in components):
        first = names.index(components[0])
        if names[first : first + 3] == components:
            return (first, first + 3)
    raise ValueError(f"{spec.key}: channel {spec.channel!r} not among target channels {tuple(target_channels)}")


def _resolve_params(spec: MetricSpec, descriptor: Callable) -> dict[str, Any]:
    given = {
        name: getattr(spec, name) for name in _PARAM_FIELDS if getattr(spec, name) != MetricSpec.model_fields[name].default
    }
    unexpected = set(given) - descriptor.params
    if unexpected:
        raise ValueError(
            f"{spec.key}: descriptor {spec.descriptor!r} does not take {sorted(unexpected)}; "
            f"it takes {sorted(descriptor.params)}"
        )
    params = {name: getattr(spec, name) for name in descriptor.params}
    missing = [name for name, value in params.items() if value is None]
    if missing:
        raise ValueError(f"{spec.key}: descriptor {spec.descriptor!r} requires {missing}")
    return params


def resolve(specs: Sequence[MetricSpec], target_channels: Sequence[str]) -> list[ResolvedMetric]:
    """Validate every spec against the registries and the task's target channels."""
    if not specs:
        raise ValueError("at least one metric is required; the first selects the best checkpoint")
    resolved: list[ResolvedMetric] = []
    seen: set[str] = set()
    for i, spec in enumerate(specs):
        descriptor = DESCRIPTORS.get(spec.descriptor)
        error = ERRORS.get(spec.error)
        if descriptor.kind not in error.accepts:
            raise ValueError(
                f"{spec.key}: error {spec.error!r} accepts {sorted(error.accepts)} but descriptor "
                f"{spec.descriptor!r} produces {descriptor.kind!r}"
            )
        if i == 0 and not error.rankable:
            raise ValueError(f"{spec.key}: the first metric selects the best checkpoint, but {spec.error!r} is not rankable")
        if i == 0 and spec.eval_only:
            raise ValueError(f"{spec.key}: the first metric selects the best checkpoint, so it cannot be eval_only")
        if not error.params <= descriptor.params:
            raise ValueError(
                f"{spec.key}: error {spec.error!r} needs {sorted(error.params)} but descriptor "
                f"{spec.descriptor!r} takes {sorted(descriptor.params)}"
            )
        if spec.key in _RESERVED_NAMES:
            raise ValueError(f"metric name {spec.key!r} is reserved by the metrics.jsonl row; set `name` to something else")
        if spec.key in seen:
            raise ValueError(f"duplicate metric name {spec.key!r}; set `name` to disambiguate")
        seen.add(spec.key)
        channel = _resolve_channel(spec, target_channels)
        # A scalar descriptor takes one field, so on a multi-channel target it would raise
        # inside `compute` at the end of epoch 0. That is a config error: say so here.
        if descriptor.kind != VOXELS_KIND and (isinstance(channel, tuple) or (channel is None and len(target_channels) != 1)):
            raise ValueError(
                f"{spec.key}: descriptor {spec.descriptor!r} needs a single channel; "
                f"set `channel` to one field (target channels {tuple(target_channels)})"
            )
        resolved.append(
            ResolvedMetric(
                name=spec.key,
                spec=spec,
                descriptor=descriptor,
                error=error,
                channel=channel,
                params=_resolve_params(spec, descriptor),
            )
        )
    return resolved


def _select(x: Tensor, channel: Channel) -> Tensor:
    if channel is None:
        return x
    if isinstance(channel, tuple):
        return x[:, channel[0] : channel[1]]
    return x[:, channel : channel + 1]


def validation_metrics(metrics: Sequence[ResolvedMetric]) -> list[ResolvedMetric]:
    """The metrics scored every epoch: everything not marked `eval_only`."""
    return [m for m in metrics if not m.spec.eval_only]


def compute(metrics: Sequence[ResolvedMetric], pred: Tensor, target: Tensor, mask: Tensor) -> dict[str, Tensor]:
    """Evaluate every resolved metric; values are per-sample tensors `(B,)` (or `(B, K)`).

    A descriptor is evaluated once per side however many errors read it — four entries on
    `curvature_hist` mesh the interface once, not four times — and the interface meshes
    are shared across descriptors for the duration of the call (`shared_meshes`), so the
    curvature family meshes each side once in total. Under an open `trace.tracing()`,
    every descriptor value is recorded as `<descriptor_key>/pred|target` and the
    descriptor's own intermediates beneath `<descriptor_key>/<side>/`.
    """
    check_field(pred, mask)
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ in shape")
    out: dict[str, Tensor] = {}
    cache: dict[tuple, tuple[Tensor, Tensor]] = {}
    with shared_meshes():
        for m in metrics:
            p, t = _select(pred, m.channel), _select(target, m.channel)
            if m.kind == VOXELS_KIND:
                out[m.name] = m.error(p, t, mask)
                continue
            key = (m.spec.descriptor, m.channel, tuple(sorted(m.params.items())))
            if key not in cache:
                dkey = m.spec.descriptor_key
                sides = []
                for side, field in (("pred", p), ("target", t)):
                    with trace.scoped(f"{dkey}/{side}/"):
                        value = m.descriptor(field, mask, **m.params)
                    trace.record(f"{dkey}/{side}", value)
                    sides.append(value)
                cache[key] = (sides[0], sides[1])
            dp, dt = cache[key]
            out[m.name] = m.error(dp, dt, **{name: m.params[name] for name in m.error.params})
    return out


def is_better(metric: ResolvedMetric, new: float, old: float) -> bool:
    """Whether `new` improves on `old` for this metric. Any value beats NaN."""
    if math.isnan(new):
        return False
    if math.isnan(old):
        return True
    return new > old if metric.higher_is_better else new < old
