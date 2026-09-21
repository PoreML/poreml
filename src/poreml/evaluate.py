"""Evaluate a checkpoint against a split.

Separate from training so scoring can be rerun at any time — a different split, a
different metric list, a checkpoint from an earlier run — without retraining. A
checkpoint carries only its model config, so scoring one still needs the matching
`Config` supplied alongside it; the run directory's `config.yaml` is that config for
anything this repo produced.
"""

import csv
import json
import logging
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import provenance
from .config import Config, resolve_device
from .data import WindowDataset
from .frames import H5FrameStore
from .losses import loss_function
from .metrics import METRICS_VERSION, json_safe, resolve, validation_metrics
from .models import build_model, precision_of, representation_of
from .precision import inference_model
from .rollout import (
    RolloutSummary,
    evaluate_rollout,
    infer_windows,
    keyframe_steps,
    rollout,
    start_index,
)
from .scoring import Evaluation, evaluate_loader, load_runs
from .tables import append_rows
from .tasks import build_task

FRAMES_DIR = "frames"  # under the target dir: the inference stage's stored keyframes

log = logging.getLogger(__name__)


def _load(cfg: Config, ckpt: Path | str | None, split: str, model: bool = True):
    """Shared preamble: runs, task, resolved metrics, and the model with weights loaded.

    `model=False` (the CPU metric stage) skips the model and pins the device to CPU."""
    device = resolve_device(cfg.train.device) if model else "cpu"
    runs = load_runs(cfg)
    task = build_task(cfg.task, representation_of(cfg.model))
    metrics = resolve(cfg.metrics if cfg.metrics is not None else task.metrics, task.target_channels)
    if not model:
        return device, runs, task, metrics, None
    net = build_model(cfg.model, task.in_channels, task.out_channels).to(device)
    if ckpt is not None:
        state = torch.load(Path(ckpt), map_location=device, weights_only=True)
        net.load_state_dict(state["model"])
    # Inference runs in the model's own precision (bf16 for AB-UPT): the wrapper autocasts
    # the forward and hands fp32 to the metrics; fp32 models come back untouched.
    return device, runs, task, metrics, inference_model(net, precision_of(cfg.model), device)


def _write_json(path: Path, payload: dict) -> None:
    """Strict JSON to a sibling, then rename: a kill mid-write never leaves a torn artifact."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2))
    tmp.replace(path)


def _target_dir(cfg: Config, ckpt: Path | str | None, out_dir: Path | str | None) -> Path:
    """Results land next to the checkpoint unless `out_dir` says otherwise."""
    target = Path(out_dir) if out_dir is not None else (Path(ckpt).parent if ckpt else Path(cfg.out_dir))
    target.mkdir(parents=True, exist_ok=True)
    return target


def _header(cfg: Config, split: str, ckpt: Path | str | None, runs, device: str) -> dict:
    return {
        "name": cfg.name,
        "split": split,
        "task": cfg.task.name,
        "model": cfg.model.name,
        "checkpoint": str(ckpt) if ckpt else None,
        "checkpoint_sha256": provenance.sha256_of(ckpt) if ckpt else None,
        "precision": precision_of(cfg.model),
        "n_runs": len(runs),
        "runs": [r.summary() for r in runs],
        "provenance": provenance.collect(cfg, device),
    }


def evaluate(
    cfg: Config,
    ckpt: Path | str | None = None,
    split: str = "test",
    out_dir: Path | str | None = None,
    stride: int | None = None,
) -> Evaluation:
    """Score a checkpoint on one split and write results_<split>.json and samples_<split>.csv.

    Results land next to the checkpoint unless `out_dir` says otherwise, and the split is
    part of the filename: scoring val after test on one checkpoint must not silently
    destroy the first result. The file carries the headline numbers, the per-run values
    behind them, validity counts, one record per scored run, and provenance — enough
    to stratify, re-aggregate, or trace any number without rerunning. The CSV beside it
    is the log behind those numbers: one row per scored sample with every metric, the
    configured training loss, every descriptor value on both sides and every intermediate
    the descriptors computed. `stride` scores every stride-th window (the periodic
    evaluation during training uses it); None means the task's own stride.
    """
    device, runs, task, metrics, model = _load(cfg, ckpt, split)
    dataset = task.dataset(runs[split], stride=stride)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, collate_fn=dataset.collate)
    run_ids = [r.run_id for r in runs[split]]
    results = evaluate_loader(model, loader, metrics, device, run_ids, loss_fn=loss_function(cfg.train.loss))

    payload = {
        **_header(cfg, split, ckpt, runs[split], device),
        "batch_size": cfg.train.batch_size,
        "stride": stride,
        "n_samples": len(dataset),
        "metrics_spec": _specs(metrics),
        **results.as_dict(),
    }
    target_dir = _target_dir(cfg, ckpt, out_dir)
    (target_dir / f"results_{split}.json").write_text(json.dumps(json_safe(payload), indent=2))
    write_distributions_csv(payload, target_dir / f"distributions_{split}.csv")
    samples = target_dir / f"samples_{split}.csv"
    samples.unlink(missing_ok=True)  # a rescoring replaces the log, as it replaces the JSON
    append_rows(samples, results.samples)
    return results


def _specs(metrics) -> list[dict]:
    return [{"name": m.name, **m.spec.model_dump(mode="json", exclude={"name"})} for m in metrics]


def write_distributions_csv(payload: dict, path: Path) -> bool:
    """Write every distribution-valued summary metric of a results payload as a long CSV.

    A `curvature_hist/target` entry is a K-vector in the JSON — fine for a program, not
    for a plot. Rows are `metric, bin, lo, hi, model` with the bin edges recovered from
    the metric's spec, so the file plots straight from pandas. Nothing is written (and
    False returned) when no metric is a distribution.
    """
    specs = {spec["name"]: spec for spec in payload.get("metrics_spec", [])}
    rows = []
    for name, values in payload["metrics"].items():
        if not isinstance(values, list):
            continue
        spec = specs[name]
        lo, hi, bins = spec["range"][0], spec["range"][1], spec["bins"]
        width = (hi - lo) / bins
        for k, value in enumerate(values):
            rows.append(
                {
                    "metric": name,
                    "bin": k,
                    "lo": lo + k * width,
                    "hi": lo + (k + 1) * width,
                    "model": value,
                }
            )
    if not rows:
        return False
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return True


def rollout_split(
    cfg: Config,
    ckpt: Path | str | None = None,
    split: str = "test",
    out_dir: Path | str | None = None,
    horizon: int | None = None,
    eval_stride: int = 1,
    eval_only: bool = True,
) -> RolloutSummary:
    """Roll the model out once per run from `cfg.rollout.start`/`start_fraction` and write rollout_<split>.json.

    Training's periodic eval and `poreml rollout` without flags. Same runs, metrics, provenance
    and placement rules as `evaluate`; `eval_stride` scores every stride-th step plus each run's
    final step. `eval_only=False` drops the `eval_only` metrics (the periodic eval during
    training: no meshing there). The studies use `inference_split` + `metric_split` instead.
    """
    device, runs, task, metrics, model = _load(cfg, ckpt, split)
    if not eval_only:
        metrics = validation_metrics(metrics)
    dataset = task.dataset(runs[split])
    if not isinstance(dataset, WindowDataset):
        raise TypeError(f"rollout needs a WindowDataset; task {cfg.task.name!r} builds {type(dataset).__name__}")
    horizon = cfg.rollout.horizon if horizon is None else horizon
    start, fraction = cfg.rollout.start, cfg.rollout.start_fraction
    summary = evaluate_rollout(
        model,
        dataset,
        metrics,
        horizon,
        device,
        start=start,
        start_fraction=fraction,
        eval_stride=eval_stride,
        keyframes=cfg.rollout.keyframes,
    )
    payload = {
        **_header(cfg, split, ckpt, runs[split], device),
        "start": start,
        "start_fraction": fraction,
        "rollout_stride": eval_stride,
        "metrics_spec": _specs(metrics),
        **summary.as_dict(),
    }
    target_dir = _target_dir(cfg, ckpt, out_dir)
    _write_json(target_dir / f"rollout_{split}.json", payload)
    return summary


def inference_split(
    cfg: Config,
    ckpt: Path | str | None = None,
    split: str = "test",
    out_dir: Path | str | None = None,
    stride: int | None = None,
    shared: bool = False,
) -> dict:
    """The study's GPU stage: store the keyframes of every 64-step window of every run, then write inference_<split>.json.

    Windows start every `stride` frames (`rollout.window_starts`; None = the horizon, so they
    tile the run) and store `cfg.rollout.keyframes` steps each under `<target_dir>/frames/`.
    Nothing is scored. The JSON is written last and is the completion sentinel: a run
    interrupted by SIGTERM keeps its finished windows and raises `Interrupted` (the CLI exits
    75); the rerun skips them. A run too short for one window is listed under
    `runs_without_windows`.

    `shared`: this process is one of several workers on the same cell (`infer_windows(shared=True)`:
    windows are claimed through the store, a held window is skipped). The sentinel is then
    written only by a worker that finds every window stored when its own pass ends — the last
    one — and the payload says `complete: False` (with `n_missing`) for the others.
    """
    device, runs, task, metrics, model = _load(cfg, ckpt, split)
    dataset = task.dataset(runs[split])
    if not isinstance(dataset, WindowDataset):
        raise TypeError(f"inference needs a WindowDataset; task {cfg.task.name!r} builds {type(dataset).__name__}")
    horizon = cfg.rollout.horizon
    stride = horizon if stride is None else stride
    sha = provenance.sha256_of(ckpt) if ckpt else None
    target_dir = _target_dir(cfg, ckpt, out_dir)
    (target_dir / f"inference_{split}.json").unlink(missing_ok=True)  # a stale sentinel from an interrupted rerun for
    # another checkpoint must not read as "complete" once this one starts
    store = H5FrameStore(target_dir / FRAMES_DIR, fields=dataset.channels)

    stop_requested = False

    def on_sigterm(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        log.warning("SIGTERM received; stopping after the window in hand")

    previous = signal.signal(signal.SIGTERM, on_sigterm) if threading.current_thread() is threading.main_thread() else None
    started = time.perf_counter()
    try:
        windows, short = infer_windows(
            model,
            dataset,
            horizon,
            stride,
            cfg.rollout.keyframes,
            store,
            device,
            checkpoint_sha256=sha,
            stop=lambda: stop_requested,
            log=log,
            shared=shared,
        )
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    steps = keyframe_steps(horizon, cfg.rollout.keyframes)
    n_windows = sum(len(v) for v in windows.values())
    missing = missing_windows(store, windows, sha, steps)
    if missing and not shared:
        raise RuntimeError(f"inference ended with {len(missing)} window(s) missing from {store.directory}: {missing[:3]}")
    payload = {
        **_header(cfg, split, ckpt, runs[split], device),
        "protocol": {"horizon": horizon, "stride": stride, "keyframes": cfg.rollout.keyframes, "steps": steps},
        "frames_dir": FRAMES_DIR,
        "windows": windows,
        "runs_without_windows": short,
        "n_windows": n_windows,
        "n_frames": n_windows * len(steps),
        "seconds": round(time.perf_counter() - started, 1),
        "shared": shared,
        "complete": not missing,
        "n_missing": len(missing),
    }
    if missing:
        log.info("shared inference: %d window(s) still held or missing; no sentinel written", len(missing))
        return payload
    _write_json(target_dir / f"inference_{split}.json", payload)
    return payload


def missing_windows(
    store: H5FrameStore, windows: dict[str, list[int]], sha: str | None, steps: list[int]
) -> list[tuple[str, int]]:
    """The (run, t0) windows not stored for this checkpoint and keyframe set — a shared worker checks before the sentinel."""
    missing = []
    for run_id, starts in windows.items():
        for t0 in starts:
            if not store.has(run_id, t0):
                missing.append((run_id, t0))
                continue
            meta = store.meta(run_id, t0)
            if meta.get("checkpoint_sha256") != sha or list(meta.get("steps", [])) != steps:
                missing.append((run_id, t0))
    return missing


def render_rollout(
    cfg: Config,
    ckpt: Path | str | None,
    run_id: str,
    out_dir: Path | str,
    split: str = "test",
    horizon: int | None = None,
    fps: int = 6,
) -> Path:
    """Roll one run out and render truth | prediction per step, plus a GIF, in the solver's style.

    Frames go to `<out_dir>/frames/frame_<step>.png` (named by the solver's step number, as in a
    run's own `frames/`), the animation to `<out_dir>/rollout_<run_id>.gif`, and the
    per-step metric values to `<out_dir>/rollout_<run_id>.json`. Returns `out_dir`.
    """
    from . import viz  # optional dependency group; imported here so training never needs VTK

    device, runs, task, metrics, model = _load(cfg, ckpt, split)
    dataset = task.dataset(runs[split])
    if not isinstance(dataset, WindowDataset):
        raise TypeError(f"render needs a WindowDataset; task {cfg.task.name!r} builds {type(dataset).__name__}")
    ids = [r.run_id for r in dataset.runs]
    if run_id not in ids:
        raise KeyError(f"run {run_id!r} is not in split {split!r}; it holds {ids}")
    run_idx = ids.index(run_id)
    if "phi" not in dataset.channels:
        raise ValueError(f"render needs a phi channel; the task's channels are {dataset.channels}")
    phi = dataset.channels.index("phi")
    t0 = start_index(dataset, run_idx, cfg.rollout.start, cfg.rollout.start_fraction)
    horizon = cfg.rollout.horizon if horizon is None else horizon
    result = rollout(model, dataset, run_idx, t0, horizon, metrics, device, keep_frames=True)

    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.png"):  # a previous render with another start/horizon must not linger
        stale.unlink()
    solid = dataset.solid(run_idx)
    steps = dataset.trajectories[run_idx].steps
    shown = [m.name for m in metrics if m.name.split("@")[0] in ("iou", "saturation/abs_err")] or [metrics[0].name]
    jobs = []
    for h, pred in enumerate(result.frames, start=1):
        step = steps[t0 + h]
        truth = dataset.frame(run_idx, t0 + h)[phi]
        stats = "  ".join(
            f"{name} {result.values[name][h - 1]:.3f}" for name in shown if isinstance(result.values[name][h - 1], float)
        )
        jobs.append(
            viz.RenderJob(
                path=frames_dir / f"frame_{int(step):08d}.png",
                panels=[
                    viz.Panel(red=truth > 0, label=f"truth  {run_id}  step {int(step)}"),
                    viz.Panel(red=pred[phi] > 0, label=f"{cfg.model.name}  h={h}  {stats}"),
                ],
            )
        )
    # `jobs` now holds only the bool red masks it needs; the multi-field float32 frame
    # stack (e.g. 64 x (5, 128, 128, 128) ~= 2.7 GB for 5 fields) has no further use and
    # must not sit in memory through rendering.
    result.frames = None
    viz.render_jobs(jobs, solid)
    viz.write_gif([j.path for j in jobs], out_dir / f"rollout_{run_id}.gif", fps=fps)
    payload = {
        "run_id": run_id,
        "model": cfg.model.name,
        "checkpoint": str(ckpt) if ckpt else None,
        "t0": result.t0,
        "horizon": result.horizon,
        "steps": [steps[t0 + h] for h in range(1, result.horizon + 1)],
        "values": result.values,
    }
    (out_dir / f"rollout_{run_id}.json").write_text(json.dumps(json_safe(payload), indent=2))
    return out_dir


_ROW_KEYS = ("run", "family", "group", "t0", "h", "t")


def metric_split(
    cfg: Config,
    ckpt: Path | str | None = None,
    split: str = "test",
    out_dir: Path | str | None = None,
    workers: int = 1,
    sentinel_name: str | None = None,
) -> dict:
    """The study's CPU stage: score every stored frame with the whole metric list and aggregate.

    Needs the `inference_<split>.json` an inference wrote beside `frames/` (`sentinel_name` names
    another file with the same keys: a training's `inference/inference.json`); refuses a checkpoint
    other than the one it names and a listed window whose file is missing or belongs to another
    checkpoint. Writes `metrics_<split>.csv` — one row per (run, window, step) with every metric
    and every traced value, the raw record behind every aggregate — and `metrics_<split>.json`
    (per-step means, per run first then over runs, overall and per run group). Both replaced on
    rescoring. No GPU: `workers` spawned processes score on CPU. The JSON's top-level `n_runs`
    and `n_windows` are the header's int counts; `n_runs_per_step` and `n_windows_per_step`
    (and the same pair inside each `groups` entry) are per-metric, per-step dicts.
    """
    from .metric import group_of, score_frames  # lazy: metric imports this module's _load

    target_dir = _target_dir(cfg, ckpt, out_dir)
    sentinel = target_dir / (sentinel_name or f"inference_{split}.json")
    if not sentinel.is_file():
        raise FileNotFoundError(f"{sentinel}: the metric stage needs the {sentinel.name} an inference wrote")
    inference = json.loads(sentinel.read_text())
    sha = provenance.sha256_of(ckpt) if ckpt else None
    if inference.get("checkpoint_sha256") != sha:
        raise ValueError(f"{sentinel} names checkpoint {inference.get('checkpoint_sha256')}, scoring {sha}")
    store_dir = target_dir / inference["frames_dir"]
    store = H5FrameStore(store_dir)
    windows: dict[str, list[int]] = {run_id: [int(t) for t in starts] for run_id, starts in inference["windows"].items()}
    for run_id, starts in windows.items():
        for t0 in starts:
            if not store.has(run_id, t0):
                raise ValueError(f"{store.path(run_id, t0)}: window listed in {sentinel.name} is missing; rerun the inference")
            stored_sha = store.meta(run_id, t0).get("checkpoint_sha256")
            if stored_sha != sha:
                raise ValueError(f"{store.path(run_id, t0)}: stored for checkpoint {stored_sha}, scoring {sha}")
            stored_steps = store.steps(run_id, t0)
            if stored_steps != list(inference["protocol"]["steps"]):
                raise ValueError(
                    f"{store.path(run_id, t0)}: stored steps {stored_steps} != the sentinel's "
                    f"{list(inference['protocol']['steps'])}; rerun the inference"
                )

    device, runs, task, metrics, _ = _load(cfg, ckpt, split, model=False)
    started = time.perf_counter()
    rows = score_frames(cfg, ckpt, split, store_dir, windows, workers)
    names = list(dict.fromkeys([m.name for m in metrics] + [k for r in rows for k in r if k not in _ROW_KEYS]))
    groups: dict[str, list[str]] = {}
    for ref in runs[split]:
        g = group_of(ref.geometry.get("family"))
        if g is not None and windows.get(ref.run_id):
            groups.setdefault(g, []).append(ref.run_id)
    metric = {
        "workers": workers,
        "seconds": round(time.perf_counter() - started, 1),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "poreml_commit": provenance.collect(cfg, device)["poreml_commit"],
        "metrics_version": METRICS_VERSION,
        "checkpoint_sha256": sha,
    }
    payload = metrics_payload(
        _header(cfg, split, ckpt, runs[split], device), inference, _specs(metrics), rows, names, groups, metric
    )
    write_metrics(target_dir, split, rows, payload)
    return payload


def metrics_payload(
    header: dict,
    inference: dict,
    metrics_spec: list[dict],
    rows: list[dict],
    names: list[str],
    groups: dict[str, list[str]],
    metric: dict,
) -> dict:
    """The `metrics_<split>.json` body from scored rows.

    `header` is `_header(...)`, `inference` the sentinel payload the rows were scored against,
    `names` the metric and traced keys to aggregate, `groups` `{gen|uCT: [run ids with windows]}`,
    `metric` the stage's own block (workers, seconds, checkpoint sha, ...). `metric_split` and a
    synthetic study (`case_dummy/make.py`) both build their files through this function and
    `write_metrics`, so the two can never disagree on keys or columns.
    """
    from .metric import aggregate  # lazy: metric imports this module's _load

    steps = list(inference["protocol"]["steps"])
    curves = aggregate(rows, names, steps, groups)
    return {
        **header,
        "protocol": inference["protocol"],
        "metrics_spec": metrics_spec,
        "steps": steps,
        "windows": inference["windows"],
        "runs_without_windows": inference["runs_without_windows"],
        "n_windows": inference["n_windows"],  # the int; `n_windows_per_step` in `curves` is per metric and step
        "n_frames": len(rows),
        **curves,
        "at_horizon": {name: curves["summary"][name][-1] for name in names},
        "metric": metric,
    }


def write_metrics(target_dir: Path, split: str, rows: list[dict], payload: dict) -> None:
    """`metrics_<split>.csv` (one row per stored frame, replaced) and `metrics_<split>.json` (atomic)."""
    csv_path = Path(target_dir) / f"metrics_{split}.csv"
    csv_path.unlink(missing_ok=True)
    append_rows(csv_path, rows)
    _write_json(Path(target_dir) / f"metrics_{split}.json", payload)
