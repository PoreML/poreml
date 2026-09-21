"""Autonomous rollout: feed the model its own predictions and score every horizon.

One-step accuracy is not rollout quality — rankings and errors change with horizon, and
a surrogate that is finite for 64 frames may be physically wrong after 8. So the model is
started from a true window and advanced on its own output, and every horizon step is
scored against the true frame with the same composed metrics as single-step evaluation.

The dataset owns the sample convention (crop, history, field, channel order) and the
stream (`encode`/`decode`), so rollout composes on `WindowDataset` — voxel or point — and
works for every model in the zoo unchanged.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import nn

from .data import WindowDataset
from .frames import FrameStore
from .metrics import ResolvedMetric, compute
from .metrics import trace as tracing

Curve = list[float] | list[list[float]]
Stage = Literal["full", "inference"]


class Interrupted(RuntimeError):
    """Raised by `infer_windows` after a stop request: every finished window is on disk, the rest is not started."""


@dataclass
class RolloutResult:
    """One run rolled out from `t0` for `horizon` steps: per-metric values per step.

    `values` holds the metrics first, in config order, then every descriptor value and
    intermediate the metrics computed on the way (`metrics.trace` keys), all as
    per-step lists with None at unscored steps.
    """

    run_id: str
    t0: int
    horizon: int
    values: dict[str, Curve]
    keyframes: list[int] = field(default_factory=list)  # steps stored / scored with the eval_only metrics
    frames: list[np.ndarray] | None = None  # (F, D, H, W) predictions at t0+1 .. t0+horizon when kept


@dataclass
class RolloutSummary:
    """Rollouts of every run in a dataset, aggregated per horizon step.

    `curves[metric][run_id]` is that run's per-step values; `summary[metric][h-1]` the
    mean over the runs that reach step `h` (`n_runs[metric][h-1]` says how many did — a
    short run is scored to its last frame and then leaves the average, never padded).
    `at_horizon` is the last summary value per metric.
    """

    horizon: int
    runs: dict[str, dict[str, int]] = field(default_factory=dict)  # run_id -> {"t0", "horizon"}
    keyframes: dict[str, list[int]] = field(default_factory=dict)  # run_id -> stored / eval_only-scored steps
    curves: dict[str, dict[str, Curve]] = field(default_factory=dict)
    summary: dict[str, Curve] = field(default_factory=dict)
    n_runs: dict[str, list[int]] = field(default_factory=dict)
    at_horizon: dict[str, float | list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "rollouts": self.runs,
            "keyframes": self.keyframes,
            "summary": self.summary,
            "at_horizon": self.at_horizon,
            "n_runs_per_step": self.n_runs,
            "curves": self.curves,
        }


def _as_value(x: torch.Tensor) -> float | list[float]:
    x = x.detach().cpu()
    return x.item() if x.ndim == 0 else x.tolist()


def scored_steps(horizon: int, eval_stride: int) -> list[int]:
    """The steps a rollout scores: `eval_stride, 2*eval_stride, ...` plus the final step."""
    return sorted({h for h in range(1, horizon + 1) if h % eval_stride == 0} | {horizon})


def keyframe_steps(horizon: int, keyframes: int | None, eval_stride: int = 1) -> list[int]:
    """The steps whose predictions are stored and scored with the eval_only metrics.

    `keyframes=None` means the scored steps (the rule before keyframes existed). An integer k
    picks the first step, the last step and k - 2 evenly spaced between them; a run whose
    horizon is shorter than k yields every step. The last keyframe is always `horizon`, so
    the at-horizon value exists for every metric.
    """
    if keyframes is None:
        return scored_steps(horizon, eval_stride)
    if horizon <= keyframes:
        return list(range(1, horizon + 1))
    return sorted({int(round(x)) for x in np.linspace(1, horizon, keyframes)})


def start_index(dataset: WindowDataset, run_idx: int, start: int | None = None, start_fraction: float = 0.25) -> int:
    """Where a rollout begins: `start` if given, else `start_fraction` of the way into the run.

    The very first frames of a drainage run are the invading phase entering the inlet
    face — every model looks alike there. A quarter of the way in the front is inside
    the rock and the rollout tests real evolution. Runs differ in length, so the
    fraction resolves per run; it is clamped to leave the history window before it and
    at least one true frame after it.
    """
    n_steps = dataset.n_steps(run_idx)
    if start is not None:
        return start
    t0 = round(start_fraction * (n_steps - 1))
    return min(max(t0, dataset.history - 1), n_steps - 2)


def _score(metrics: Sequence[ResolvedMetric], pred, target, mask) -> dict[str, float | list[float]]:
    with tracing.tracing() as trace:
        step = {name: _as_value(value[0]) for name, value in compute(metrics, pred, target, mask).items()}
    step.update({name: _as_value(value[0]) for name, value in trace.values.items()})
    return step


def window_starts(n_steps: int, history: int, horizon: int, stride: int) -> list[int]:
    """Where the study's rollout windows begin: `history - 1`, then every `stride` frames.

    A window feeds the true frames up to `t0` and predicts `t0 + 1 .. t0 + horizon`; it is kept
    only if that fits in the run (`t0 + horizon <= n_steps - 1`). Every window is therefore
    exactly `horizon` steps long, a trailing partial window is discarded, and a run shorter than
    `horizon + history` frames yields none — the caller reports it, nothing is dropped silently.
    """
    return [t0 for t0 in range(history - 1, n_steps, stride) if t0 + horizon <= n_steps - 1]


def score_frame(
    dataset: WindowDataset, metrics: Sequence[ResolvedMetric], run_idx: int, t0: int, h: int, pred: np.ndarray, device="cpu"
) -> dict[str, float | list[float]]:
    """Score one stored prediction `pred` `(F, D, H, W)` of step `h` against the true frame at `t0 + h`.

    The metric stage's unit of work — the same call the full stage makes at a scored step, so the
    two agree bit for bit."""
    mask = torch.from_numpy(~dataset.solid(run_idx))[None, None].to(device)
    target = torch.from_numpy(dataset.frame(run_idx, t0 + h))[None].to(device)
    return _score(metrics, torch.from_numpy(np.asarray(pred, dtype=np.float32))[None].to(device), target, mask)


def rollout(
    model: nn.Module,
    dataset: WindowDataset,
    run_idx: int,
    t0: int,
    horizon: int,
    metrics: Sequence[ResolvedMetric],
    device: str,
    keep_frames: bool = False,
    eval_stride: int = 1,
    keyframes: int | None = None,
    store: FrameStore | None = None,
    stage: Stage = "full",
    checkpoint_sha256: str | None = None,
) -> RolloutResult:
    """Roll `model` out from the true window ending at `t0` for `horizon` steps.

    `full` scores every `eval_stride`-th step plus the last with the cheap metrics and the
    keyframes (`keyframe_steps`) with the `eval_only` ones too. `inference` scores nothing: it
    writes the keyframe predictions to `store` under (run, t0) with `horizon`, `steps` and
    `checkpoint_sha256` as the window's meta, and every curve stays None. The horizon is
    capped at the last true frame and the actual value recorded (the study passes windows that
    fit, `window_starts`); a run too short for one step raises.

    The dataset owns the stream: `encode` builds the model input from grid frames (a voxel
    tensor or `points.Points`), `decode` returns a grid frame with solid voxels forced to
    each field's fill (`dataset.frame_fill`) — the model is never trusted on voxels that are
    not fluid, and each channel is forced to its own fill rather than a single scalar since
    a frame may hold several fields.
    """
    if not isinstance(dataset, WindowDataset):
        raise TypeError(f"rollout composes on WindowDataset; got {type(dataset).__name__}")
    if stage == "inference" and store is None:
        raise ValueError("stage='inference' needs a frame store to write the keyframes to")
    run_id = dataset.runs[run_idx].run_id
    history = dataset.history
    n_steps = dataset.n_steps(run_idx)
    if t0 < history - 1:
        raise ValueError(f"t0={t0} leaves no room for history={history}; the earliest start is t0={history - 1}")
    if t0 + 1 >= n_steps:
        raise ValueError(f"t0={t0} has no frame after it in a run of {n_steps} frames")
    horizon = min(horizon, n_steps - 1 - t0)
    grid = set(scored_steps(horizon, eval_stride))
    keys = keyframe_steps(horizon, keyframes, eval_stride)
    cheap = [m for m in metrics if not m.spec.eval_only]
    expensive = [m for m in metrics if m.spec.eval_only]
    if stage == "inference":
        store.open(run_id, t0, horizon=horizon, steps=keys, checkpoint_sha256=checkpoint_sha256)

    solid = dataset.solid(run_idx)
    mask = torch.from_numpy(~solid)[None, None].to(device)
    window = [dataset.frame(run_idx, t0 - k) for k in reversed(range(history))]
    scored: dict[int, dict[str, float | list[float]]] = {}
    frames: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for h in range(1, horizon + 1):
            inputs = dataset.encode(run_idx, window).to(device)
            pred = dataset.decode(inputs, model(inputs))  # (1, F, D, H, W), solid at each field's fill
            if stage == "full":
                todo = (cheap if h in grid else []) + (expensive if h in keys else [])
                if todo:
                    target = torch.from_numpy(dataset.frame(run_idx, t0 + h))[None].to(device)
                    scored[h] = _score(todo, pred, target, mask)
            frame = pred[0].cpu().numpy()
            if stage == "inference" and h in keys:
                store.write(run_id, t0, h, frame)
            window = window[1:] + [frame]
            if keep_frames:
                frames.append(frame)
    if stage == "inference":
        store.finish(run_id, t0)
    names = list(dict.fromkeys([m.name for m in metrics] + [name for step in scored.values() for name in step]))
    values = {name: [scored.get(h, {}).get(name) for h in range(1, horizon + 1)] for name in names}
    return RolloutResult(
        run_id=run_id, t0=t0, horizon=horizon, values=values, keyframes=keys, frames=frames if keep_frames else None
    )


def infer_windows(
    model: nn.Module,
    dataset: WindowDataset,
    horizon: int,
    stride: int,
    keyframes: int | None,
    store: FrameStore,
    device: str,
    checkpoint_sha256: str | None = None,
    stop: Callable[[], bool] | None = None,
    log: logging.Logger | None = None,
    shared: bool = False,
) -> tuple[dict[str, list[int]], list[str]]:
    """The inference stage: every `window_starts` window of every run into `store`.

    Returns `(windows, short)`: the window starts per run id, and the ids of runs too short
    for one window. A window already in the store for the same checkpoint *and* the same
    keyframe steps is skipped, so a rerun resumes at the first missing one; a rerun with a
    different `keyframes` setting rewrites every window instead of trusting a stale, mismatched
    set of stored steps. `stop()` is polled between windows; when it returns True the function
    raises `Interrupted` — the window in hand is always finished.

    `shared`: several workers walk the same list against one store. Each missing window is
    `store.claim`ed before it is computed and skipped when another worker holds it, so the
    workers hand the windows out among themselves as they go (a worker that starts late takes
    what is left; none is assigned a fixed share). The returned `windows` still lists every
    window — the caller checks the store to know whether the stage is complete.
    """
    keys = keyframe_steps(horizon, keyframes)
    windows: dict[str, list[int]] = {}
    short: list[str] = []
    for i, ref in enumerate(dataset.runs):
        starts = window_starts(dataset.n_steps(i), dataset.history, horizon, stride)
        windows[ref.run_id] = starts
        if not starts:
            short.append(ref.run_id)
            continue
        for t0 in starts:
            if store.has(ref.run_id, t0):
                meta = store.meta(ref.run_id, t0)
                if meta.get("checkpoint_sha256") == checkpoint_sha256 and list(meta.get("steps", [])) == keys:
                    continue
            if shared and not store.claim(ref.run_id, t0):
                if log:
                    log.info("inference %s t0=%d held by another worker; skipped", ref.run_id, t0)
                continue
            if log:
                log.info("inference %s t0=%d (%d/%d windows)", ref.run_id, t0, starts.index(t0) + 1, len(starts))
            rollout(
                model,
                dataset,
                i,
                t0,
                horizon,
                [],
                device,
                keyframes=keyframes,
                store=store,
                stage="inference",
                checkpoint_sha256=checkpoint_sha256,
            )
            if stop is not None and stop():
                raise Interrupted(f"stopped after {ref.run_id} t0={t0}; finished windows are stored, rerun to resume")
    return windows, sorted(short)


def summarise(results: Sequence[RolloutResult], horizon: int, names: Sequence[str]) -> RolloutSummary:
    """Aggregate per-run rollouts per horizon step (see `RolloutSummary`)."""
    out = RolloutSummary(horizon=horizon)
    for r in results:
        out.runs[r.run_id] = {"t0": r.t0, "horizon": r.horizon}
        out.keyframes[r.run_id] = list(r.keyframes)
    for name in names:
        out.curves[name] = {r.run_id: r.values.get(name, [None] * r.horizon) for r in results}
        per_h: list[torch.Tensor] = []
        n_at: list[int] = []
        for h in range(horizon):
            at_h = [
                torch.tensor(out.curves[name][r.run_id][h], dtype=torch.float64)
                for r in results
                if h < r.horizon and out.curves[name][r.run_id][h] is not None
            ]
            n_at.append(len(at_h))
            per_h.append(torch.nanmean(torch.stack(at_h), dim=0) if at_h else torch.tensor(float("nan")))
        out.summary[name] = [_as_value(v) for v in per_h]
        out.n_runs[name] = n_at
        out.at_horizon[name] = out.summary[name][-1]
    return out


def evaluate_rollout(
    model: nn.Module,
    dataset: WindowDataset,
    metrics: Sequence[ResolvedMetric],
    horizon: int,
    device: str,
    start: int | None = None,
    start_fraction: float = 0.25,
    eval_stride: int = 1,
    keyframes: int | None = None,
) -> RolloutSummary:
    """Roll out every run in `dataset` and aggregate per horizon step.

    `start` is the time index of the last true input frame; None means
    `start_fraction` of the way into each run (see `start_index`). `eval_stride`
    (see `rollout`) leaves unscored steps as None in the curves and NaN in the
    summary (null in the JSON); `n_runs` counts the runs *scored* at each step.
    `keyframes` is passed through to `rollout`. Full stage only.
    """
    results = [
        rollout(
            model,
            dataset,
            i,
            start_index(dataset, i, start, start_fraction),
            horizon,
            metrics,
            device,
            eval_stride=eval_stride,
            keyframes=keyframes,
        )
        for i in range(len(dataset.runs))
    ]
    owned = list(metrics)
    names = list(dict.fromkeys([m.name for m in owned] + [name for r in results for name in r.values]))
    return summarise(results, horizon, names)
