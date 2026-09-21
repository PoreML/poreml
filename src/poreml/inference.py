"""Timed in-distribution rollouts of a finished training: every 64-frame window of every validation run.

The stage behind `util/inference/inference.py` and `poreml inference`. It applies the studies' window
protocol to a training's own validation split: windows start at frame 0 (`history - 1`) and every
`stride` frames after it (`rollout.window_starts`, full windows only — a run under `horizon + 1`
frames yields none and is listed under `runs_without_windows`), and every window is rolled out
`horizon` steps. At every step it

- times the model call alone between two device synchronisations (`forward_s`) and the whole
  step — encode, forward, decode, host copy — as `step_s`;
- scores `mae` and `rel_mae` on each requested field the task targets (default `phi` and `p`),
  pore voxels only, through the same registry and scorer as every other stage — every step,
  not only the stored ones;
- stores the `keyframes` predictions (first and last step always; 12 unless the config or the
  caller says otherwise) under `<out_dir>/inference/frames/<run>/w<t0>.h5`, the studies' format.

Artifacts: `<out_dir>/inference.csv` — one row per window and step with the studies' keys
(`run, family, group, t0, h, t`) plus `forward_s, step_s, stored` and the metrics, appended per
window; `<out_dir>/inference/inference.json` — header, protocol, windows, the per-step summary
**per run first, then over runs** (`metric.aggregate`: finite values only, `n_runs_per_step`,
`n_windows_per_step`, the same per gen / uCT group), `at_horizon`, the timing statistics —
written last, the completion sentinel. SIGTERM stops the stage after the window in hand
(`rollout.Interrupted`; the CLI and the driver exit 75); a rerun for the same checkpoint keeps
every complete window (frame file for this checkpoint + all its rows) and resumes at the first
missing one. `force`, or a different checkpoint, redoes the stage from scratch.

`metric_inference` is the CPU twin (2026-09-19): the studies' `metric` stage on the keyframes this
stage stored — the run's **whole** metric list (descriptors, meshes, curvature) on every stored
frame, written as `<out_dir>/inference/metrics_<split>.csv` / `.json` in the studies' format, so
a training's in-distribution descriptors are scored under the same window protocol as the
transfer tests. `inference.csv` and `inference.json` are not touched.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import signal
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import provenance
from .config import Config
from .data import WindowDataset
from .evaluate import FRAMES_DIR, _header, _load, _specs, _write_json, metric_split
from .frames import FrameStore, H5FrameStore
from .metric import aggregate, group_of
from .metrics import MetricSpec, ResolvedMetric, json_safe, resolve
from .rollout import Interrupted, _score, keyframe_steps, window_starts
from .tables import append_rows

log = logging.getLogger(__name__)

CSV_NAME = "inference.csv"
DIR_NAME = "inference"
JSON_NAME = "inference.json"
DEFAULT_FIELDS = ("phi", "p")
DEFAULT_KEYFRAMES = 12  # the study protocol; the training configs predate the knob and hold None
ERRORS = ("mae", "rel_mae")
ROW_KEYS = ("run", "family", "group", "t0", "h", "t")  # the studies' keys, in their order
TIMING_KEYS = ("forward_s", "step_s", "stored")


def inference_metrics(fields: Sequence[str], target_channels: Sequence[str]) -> tuple[list[ResolvedMetric], list[str]]:
    """`mae` and `rel_mae` on every requested field the task targets, and the fields it does not."""
    present = [f for f in fields if f in target_channels]
    missing = [f for f in fields if f not in target_channels]
    if not present:
        raise ValueError(f"none of the fields {list(fields)} is a target of this task (targets: {list(target_channels)})")
    specs = [MetricSpec(error=e, channel=f) for f in present for e in ERRORS]
    return resolve(specs, target_channels), missing


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def rollout_window(
    model: nn.Module,
    dataset: WindowDataset,
    run_idx: int,
    t0: int,
    horizon: int,
    metrics: Sequence[ResolvedMetric],
    device: str,
    keyframes: int | None,
    store: FrameStore,
    checkpoint_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Roll one window out from `t0`, timing every forward: one row per step.

    The loop is `rollout.rollout`'s (encode → forward → decode with solid at each field's fill →
    feed back) with the model call bracketed by device synchronisations so `forward_s` is the
    kernel time, not the launch time. Scoring happens after the step's clock stops. The caller
    passes windows that fit (`window_starts`), so the horizon is never capped here.
    """
    ref = dataset.runs[run_idx]
    run_id = ref.run_id
    history = dataset.history
    n_steps = dataset.n_steps(run_idx)
    if t0 < history - 1:
        raise ValueError(f"t0={t0} leaves no room for history={history}; the earliest start is t0={history - 1}")
    if t0 + horizon > n_steps - 1:
        raise ValueError(f"t0={t0} + horizon={horizon} runs past the last frame {n_steps - 1} of {run_id}")
    keys = keyframe_steps(horizon, keyframes)
    store.open(run_id, t0, horizon=horizon, steps=keys, checkpoint_sha256=checkpoint_sha256)

    family = ref.geometry.get("family")
    solid = dataset.solid(run_idx)
    mask = torch.from_numpy(~solid)[None, None].to(device)
    window = [dataset.frame(run_idx, t0 - k) for k in reversed(range(history))]
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for h in range(1, horizon + 1):
            step_started = time.perf_counter()
            inputs = dataset.encode(run_idx, window).to(device)
            _sync(device)
            started = time.perf_counter()
            out = model(inputs)
            _sync(device)
            forward_s = time.perf_counter() - started
            pred = dataset.decode(inputs, out)  # (1, F, D, H, W), solid at each field's fill
            frame = pred[0].cpu().numpy()
            step_s = time.perf_counter() - step_started

            target = torch.from_numpy(dataset.frame(run_idx, t0 + h))[None].to(device)
            scores = _score(metrics, pred, target, mask)
            if h in keys:
                store.write(run_id, t0, h, frame)
            window = window[1:] + [frame]
            rows.append(
                {
                    "run": run_id,
                    "family": family,
                    "group": group_of(family),
                    "t0": t0,
                    "h": h,
                    "t": t0 + h,
                    "forward_s": forward_s,
                    "step_s": step_s,
                    "stored": h in keys,
                    **scores,
                }
            )
    store.finish(run_id, t0)
    return rows


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    values = [v for v in values if _finite(v)]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def _checkpoint_epoch(ckpt: Path | str | None) -> int | None:
    if ckpt is None:
        return None
    state = torch.load(Path(ckpt), map_location="cpu", weights_only=True)
    epoch = state.get("epoch") if isinstance(state, dict) else None
    return int(epoch) if epoch is not None else None


def _stored(sentinel: Path, sha: str | None, split: str) -> dict | None:
    """The finished stage's payload when its sentinel names this checkpoint and split, else None."""
    if not sentinel.is_file():
        return None
    try:
        payload = json.loads(sentinel.read_text())
    except (OSError, ValueError):
        return None
    if payload.get("checkpoint_sha256") == sha and payload.get("split") == split:
        return payload
    return None


def _parse(value: str, key: str) -> Any:
    """A CSV cell back to the type `rollout_window` wrote."""
    if key in ("run", "family"):
        return value
    if key == "group":
        return value or None
    if key in ("t0", "h", "t"):
        return int(value)
    if key == "stored":
        return value == "True"
    if value in ("", "nan", "NaN", "None"):
        return float("nan")
    return float(value)


def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    """The stage's CSV back as typed rows (an empty list when there is none)."""
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="") as f:
        return [{k: _parse(v, k) for k, v in row.items()} for row in csv.DictReader(f)]


def complete_windows(
    rows: Sequence[dict[str, Any]], store: H5FrameStore, windows: dict[str, list[int]], sha: str | None, keys: list[int]
) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    """The windows a rerun may keep: frame file for this checkpoint with these steps, every step's row present."""
    by_window: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        by_window.setdefault((r["run"], r["t0"]), []).append(r)
    kept: set[tuple[str, int]] = set()
    kept_rows: list[dict[str, Any]] = []
    for run_id, starts in windows.items():
        for t0 in starts:
            if not store.has(run_id, t0):
                continue
            meta = store.meta(run_id, t0)
            if meta.get("checkpoint_sha256") != sha or list(meta.get("steps", [])) != keys:
                continue
            got = by_window.get((run_id, t0), [])
            if sorted(r["h"] for r in got) != list(range(1, int(meta["horizon"]) + 1)):
                continue
            kept.add((run_id, t0))
            kept_rows.extend(sorted(got, key=lambda r: r["h"]))
    return kept, kept_rows


def rollout_inference(
    cfg: Config,
    ckpt: Path | str | None = None,
    split: str = "val",
    out_dir: Path | str | None = None,
    horizon: int | None = None,
    stride: int | None = None,
    keyframes: int | None = None,
    fields: Sequence[str] = DEFAULT_FIELDS,
    force: bool = False,
    stop: Callable[[], bool] | None = None,
) -> dict:
    """Roll `ckpt` out over every window of every run of `split`; write `<out_dir>/inference.csv` and `<out_dir>/inference/`.

    `out_dir` defaults to the checkpoint's run directory (the parent of its `ckpts/`), else
    `cfg.out_dir`. `horizon` and `keyframes` override the config's rollout section (a config
    without `keyframes` — every training config — stores `DEFAULT_KEYFRAMES` steps, never every
    step); `stride` is the frames between window starts, the horizon when None (windows tile the
    run). Returns the JSON payload; a finished stage for the same checkpoint and split is
    returned as stored unless `force`. `stop()` is polled between windows (SIGTERM sets it in
    the main thread); when it returns True the stage raises `Interrupted` with every finished
    window on disk, and the rerun resumes at the first missing one.
    """
    out_dir = stage_out_dir(cfg, ckpt, out_dir)
    stage_dir = out_dir / DIR_NAME
    sentinel = stage_dir / JSON_NAME
    csv_path = out_dir / CSV_NAME
    sha = provenance.sha256_of(ckpt) if ckpt else None
    if not force and (stored := _stored(sentinel, sha, split)) is not None:
        log.info("inference: %s already holds this checkpoint's %s rollouts; skipping (force to redo)", out_dir, split)
        return stored

    device, runs, task, _, model = _load(cfg, ckpt, split)
    dataset = task.dataset(runs[split])
    if not isinstance(dataset, WindowDataset):
        raise TypeError(f"inference needs a WindowDataset; task {cfg.task.name!r} builds {type(dataset).__name__}")
    metrics, missing = inference_metrics(fields, task.target_channels)
    if missing:
        log.warning("inference: the task targets none of %s; scoring %s", missing, [m.name for m in metrics])

    horizon = cfg.rollout.horizon if horizon is None else horizon
    stride = horizon if stride is None else stride
    keyframes = cfg.rollout.keyframes if keyframes is None else keyframes
    if keyframes is None:
        keyframes = DEFAULT_KEYFRAMES
    keys = keyframe_steps(horizon, keyframes)
    windows = {
        ref.run_id: window_starts(dataset.n_steps(i), dataset.history, horizon, stride) for i, ref in enumerate(dataset.runs)
    }
    short = sorted(run_id for run_id, starts in windows.items() if not starts)

    # A stale sentinel (another checkpoint's finished stage) must not read as complete once this
    # one starts; `force` or another checkpoint's frames mean a fresh start, otherwise every
    # complete window of this checkpoint is kept and the rest resumed.
    sentinel.unlink(missing_ok=True)
    store = H5FrameStore(stage_dir / FRAMES_DIR, fields=dataset.channels)
    kept: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    if not force and stage_dir.is_dir():
        kept, rows = complete_windows(read_rows(csv_path), store, windows, sha, keys)
    if not kept:
        if stage_dir.is_dir():
            shutil.rmtree(stage_dir)
        csv_path.unlink(missing_ok=True)
        stage_dir.mkdir(parents=True)
    else:
        log.info("inference: resuming %s — %d windows already complete", out_dir, len(kept))
        csv_path.unlink()
        append_rows(csv_path, rows)  # the CSV holds exactly the kept windows again; incomplete ones are redone

    stop_requested = False

    def on_sigterm(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        log.warning("SIGTERM received; stopping after the window in hand")

    previous = signal.signal(signal.SIGTERM, on_sigterm) if threading.current_thread() is threading.main_thread() else None
    started = time.perf_counter()
    n_total = sum(len(v) for v in windows.values())
    done = len(kept)
    try:
        for i, ref in enumerate(dataset.runs):
            for t0 in windows[ref.run_id]:
                if (ref.run_id, t0) in kept:
                    continue
                window_rows = rollout_window(model, dataset, i, t0, horizon, metrics, device, keyframes, store, sha)
                append_rows(csv_path, window_rows)
                rows.extend(window_rows)
                done += 1
                last = window_rows[-1][metrics[0].name]
                log.info(
                    "inference %s t0=%d (%d/%d windows)  forward %.3f s/step  %s at horizon %s",
                    ref.run_id,
                    t0,
                    done,
                    n_total,
                    statistics.fmean(r["forward_s"] for r in window_rows),
                    metrics[0].name,
                    f"{last:.5g}" if _finite(last) else "-",
                )
                if stop_requested or (stop is not None and stop()):
                    raise Interrupted(f"stopped after {ref.run_id} t0={t0}; finished windows are stored, rerun to resume")
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)

    names = list(dict.fromkeys([m.name for m in metrics] + [k for r in rows for k in r if k not in ROW_KEYS + TIMING_KEYS]))
    groups: dict[str, list[str]] = {}
    for ref in dataset.runs:
        g = group_of(ref.geometry.get("family"))
        if g is not None and windows.get(ref.run_id):
            groups.setdefault(g, []).append(ref.run_id)
    curves = aggregate(rows, names, list(range(1, horizon + 1)), groups)
    fresh = [r for r in rows if (r["run"], r["t0"]) not in kept]
    payload = {
        **_header(cfg, split, ckpt, runs[split], device),
        "checkpoint_epoch": _checkpoint_epoch(ckpt),
        "protocol": {"horizon": horizon, "stride": stride, "keyframes": keyframes, "steps": keys},
        "scored_steps": list(range(1, horizon + 1)),
        "fields": [f for f in fields if f not in missing],
        "fields_missing": missing,
        "metrics_spec": _specs(metrics),
        "csv": CSV_NAME,
        "frames_dir": FRAMES_DIR,
        "windows": windows,
        "runs_without_windows": short,
        "n_windows": n_total,
        "n_frames": len(rows),
        **curves,
        "at_horizon": {name: curves["summary"][name][-1] for name in names},
        "timing": {
            "forward_s": _stats([r["forward_s"] for r in rows]),
            "step_s": _stats([r["step_s"] for r in rows]),
            "seconds": time.perf_counter() - started,
            "windows_timed_this_segment": n_total - len(kept),
            "forward_s_this_segment": _stats([r["forward_s"] for r in fresh]),
        },
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(sentinel, payload)
    return json.loads(json.dumps(json_safe(payload)))


def stage_out_dir(cfg: Config, ckpt: Path | str | None, out_dir: Path | str | None) -> Path:
    """Where the stage lives: `out_dir`, else the checkpoint's run directory, else `cfg.out_dir`."""
    if out_dir is not None:
        return Path(out_dir)
    return Path(ckpt).parent.parent if ckpt is not None and Path(ckpt).parent.name == "ckpts" else Path(cfg.out_dir)


def metric_inference(
    cfg: Config, ckpt: Path | str | None = None, split: str = "val", out_dir: Path | str | None = None, workers: int = 1
) -> dict:
    """Score the keyframes `rollout_inference` stored with the config's whole metric list, on CPU.

    The studies' `metric_split` pointed at `<out_dir>/inference/` with `inference.json` as its
    sentinel: same refusals (another checkpoint, a missing window, other stored steps), same
    artifacts — `inference/metrics_<split>.csv` (one row per run, window and stored step: every
    metric and traced value) and `inference/metrics_<split>.json` (per step, per run first then
    over runs; `at_horizon`; a `metric` block with `METRICS_VERSION` and the checkpoint sha).
    Both are replaced on rescoring; nothing the GPU stage wrote changes.
    """
    stage_dir = stage_out_dir(cfg, ckpt, out_dir) / DIR_NAME
    return metric_split(cfg, ckpt=ckpt, split=split, out_dir=stage_dir, workers=workers, sentinel_name=JSON_NAME)
