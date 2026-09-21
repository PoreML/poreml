"""A side channel for everything a metric computes on the way to its number.

Descriptors return one value per sample; the intermediates behind it — the pore volume
under a saturation, the scored area under an integral curvature — are just as measured
and would otherwise be thrown away. A `tracing()` context, opened by the scoring loop
around `compute`, collects whatever `record` is called with while it is active; outside
one, `record` costs a dictionary lookup and does nothing, so descriptors record freely.

Keys are `<descriptor>@<channel>[@w]/<side>[/<intermediate>]`: `compute` records each
descriptor's value under `<key>/pred` and `<key>/target`, and runs the descriptor under
`scoped("<key>/<side>/")` so its own records land beneath it. Every recorded tensor is
per sample, `(B,)` or `(B, K)`, like the metrics themselves.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from torch import Tensor


class Trace:
    """Ordered `{key: per-sample tensor}` collected while the context is open."""

    def __init__(self) -> None:
        self.values: dict[str, Tensor] = {}
        self.prefix = ""


_ACTIVE: ContextVar[Trace | None] = ContextVar("poreml_metrics_trace", default=None)


def active() -> Trace | None:
    return _ACTIVE.get()


@contextmanager
def tracing() -> Iterator[Trace]:
    """Collect records into a fresh `Trace`; nested traces are independent."""
    trace = Trace()
    token = _ACTIVE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE.reset(token)


@contextmanager
def scoped(prefix: str) -> Iterator[None]:
    """Prefix every record made inside with `prefix`; a no-op without an active trace."""
    trace = _ACTIVE.get()
    if trace is None:
        yield
        return
    previous = trace.prefix
    trace.prefix = previous + prefix
    try:
        yield
    finally:
        trace.prefix = previous


def record(name: str, values: Tensor) -> None:
    """Keep `values` (per sample) under the current scope's `name`; nothing outside a trace."""
    trace = _ACTIVE.get()
    if trace is not None:
        trace.values[trace.prefix + name] = values.detach()
