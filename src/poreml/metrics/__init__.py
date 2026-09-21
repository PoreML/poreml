"""Evaluation criteria: descriptors × errors, composed by config.

A metric is `error(descriptor(pred), descriptor(target))`, evaluated per sample. Layer 1
(`descriptors.py`) turns one field into a physical quantity — volume, saturation,
interfacial area, contact area, trapped volume, a histogram. Layer 2 (`errors.py`)
compares the two sides — abs/rel error, MAE, RMSE, IoU, Wasserstein — or reports one raw.
`voxels` is the identity descriptor, so a plain MAE of the field is the same machinery.

Every metric returns a per-sample tensor `(B,)`; the evaluator concatenates across the
loader and reports the mean, so a number never depends on batch size.
"""

import math
from typing import Any

from . import descriptors as descriptors  # noqa: F401  (registration side effect)
from . import distributions as distributions  # noqa: F401
from . import errors as errors  # noqa: F401
from .compose import ResolvedMetric, compute, is_better, resolve, validation_metrics
from .descriptors import PHASE_THRESHOLD
from .spec import VOXELS, MetricSpec

__all__ = [
    "METRICS_VERSION",
    "PHASE_THRESHOLD",
    "VOXELS",
    "MetricSpec",
    "ResolvedMetric",
    "compute",
    "is_better",
    "json_safe",
    "resolve",
    "validation_metrics",
]

# Bump whenever a descriptor or error changes what it computes. Two result files with
# different versions are not comparable even under identical specs; a metric bug fix is
# a new evaluator version, not a silent change to published numbers.
# 5: mesh components under 30 vertices (speckle shells) dropped before every mesh descriptor —
#    they segfaulted libigl's quadric fit on 3 % of the study predictions (83 % at the horizon
#    for the noisiest) and made the area-normalised curvature integral blow up
# 4: open rims held on the box face while smoothing (changes curvature on ROI-cut phases); mesh areas
#    and curvature integrals added
# 3: meshes deduplicated before the quadric fit
# 2: w1 in field units (was bins); curvature descriptors
METRICS_VERSION = "5"


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with None so a payload is strict JSON.

    Metrics are NaN whenever nothing was scored — an all-solid mask, an empty loader —
    and Python's json module writes that as a bare `NaN`, which jq, JSON.parse, Go and R
    all reject. Every artifact writer in this repo runs its payload through here, so a
    published result file is always parseable by something other than Python.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value
