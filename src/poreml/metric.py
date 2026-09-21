"""The CPU metric stage: score every stored rollout frame on a process pool, then aggregate.

One unit of work is one (run, window, stored step): the worker reads the stored prediction,
the true frame and the solid mask and calls `rollout.score_frame` with the whole metric list
under a trace — the same call the in-process rollout makes at a scored step, so the numbers
are identical. Workers are spawned (a parent may hold a CUDA context) and build the dataset
and metrics once each; `workers == 1` runs in the calling process.

`aggregate` turns the rows into the study's curves: per stored step, the mean over a run's
windows first, then the mean over runs (every trajectory counts once, however many windows
it has), with `n_runs_per_step` and `n_windows_per_step` recorded; the same per run group (gen / uCT).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from .config import Config
from .frames import H5FrameStore
from .rollout import score_frame
from .study import SOURCES

_STATE: dict = {}


def default_workers() -> int:
    """`$SLURM_CPUS_PER_TASK` inside a job, else the machine's CPU count."""
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1))


def group_of(family: str | None) -> str | None:
    """`gen` or `uCT` for a rock family (`study.SOURCES`), None for anything else."""
    for name, members in SOURCES.items():
        if family in members:
            return name
    return None


def _init(cfg: Config, ckpt, split: str, store_dir: str) -> None:
    from .evaluate import _load  # evaluate imports this module lazily; importing here avoids a cycle at load

    _, runs, task, metrics, _ = _load(cfg, ckpt, split, model=False)
    _STATE["dataset"] = task.dataset(runs[split])
    _STATE["metrics"] = list(metrics)
    _STATE["store"] = H5FrameStore(store_dir)


def _one(job: tuple[int, str, int, int]) -> tuple[int, int, int, dict]:
    run_idx, run_id, t0, h = job
    assert _STATE["dataset"].runs[run_idx].run_id == run_id, (run_idx, run_id)
    pred = _STATE["store"].read(run_id, t0, h)
    return run_idx, t0, h, score_frame(_STATE["dataset"], _STATE["metrics"], run_idx, t0, h, pred, "cpu")


def _pool_map(fn: Callable, jobs: Iterable, workers: int, initializer: Callable | None, initargs: tuple) -> Iterator:
    """`fn` over `jobs` on `workers` spawned processes, results in job order.

    A worker that dies — the kernel's OOM killer, a C++ crash — raises `BrokenProcessPool` at once.
    `multiprocessing.Pool` instead replaces the worker and waits for its lost job for ever: two
    drainage cells idled to their 2 h limit that way (2026-09-19, 16 workers under `--mem=48G`).
    """
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=initializer, initargs=initargs) as pool:
        yield from pool.map(fn, jobs, chunksize=1)


def score_frames(cfg: Config, ckpt, split: str, store_dir: Path, windows: dict[str, list[int]], workers: int) -> list[dict]:
    """One row per stored frame of every window in `windows`, ordered by run, t0, step."""
    from .evaluate import _load

    _, runs, task, _, _ = _load(cfg, ckpt, split, model=False)
    dataset = task.dataset(runs[split])
    store = H5FrameStore(store_dir)
    refs = {ref.run_id: (i, ref) for i, ref in enumerate(dataset.runs)}
    jobs = [
        (refs[run_id][0], run_id, t0, h) for run_id, starts in windows.items() for t0 in starts for h in store.steps(run_id, t0)
    ]
    scored: dict[tuple[int, int, int], dict] = {}
    if workers <= 1:
        _init(cfg, ckpt, split, str(store_dir))
        for job in jobs:
            run_idx, t0, h, step = _one(job)
            scored[(run_idx, t0, h)] = step
    else:
        for run_idx, t0, h, step in _pool_map(_one, jobs, workers, _init, (cfg, ckpt, split, str(store_dir))):
            scored[(run_idx, t0, h)] = step
    rows = []
    for run_idx, run_id, t0, h in jobs:
        family = dataset.runs[run_idx].geometry.get("family")
        rows.append(
            {
                "run": run_id,
                "family": family,
                "group": group_of(family),
                "t0": t0,
                "h": h,
                "t": t0 + h,
                **scored[(run_idx, t0, h)],
            }
        )
    return rows


def _nanmean(values: list) -> float | list[float]:
    stacked = torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in values])
    mean = torch.nanmean(stacked, dim=0)
    return mean.item() if mean.ndim == 0 else mean.tolist()


def _curves(rows: list[dict], names: Sequence[str], steps: Sequence[int], run_ids: Sequence[str]) -> dict:
    summary: dict[str, list] = {}
    n_runs: dict[str, list[int]] = {}
    n_windows: dict[str, list[int]] = {}
    by_key: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if row["run"] in run_ids:
            by_key.setdefault((row["run"], row["h"]), []).append(row)
    for name in names:
        means, runs_at, windows_at = [], [], []
        for h in steps:
            run_means = []
            n_win = 0
            for run_id in run_ids:
                values = [
                    r[name] for r in by_key.get((run_id, h), []) if name in r and r[name] is not None and _finite(r[name])
                ]
                if values:
                    run_means.append(_nanmean(values))
                    n_win += len(values)
            means.append(_nanmean(run_means) if run_means else float("nan"))
            runs_at.append(len(run_means))
            windows_at.append(n_win)
        summary[name], n_runs[name], n_windows[name] = means, runs_at, windows_at
    return {"summary": summary, "n_runs_per_step": n_runs, "n_windows_per_step": n_windows}


def _finite(value) -> bool:
    t = torch.as_tensor(value, dtype=torch.float64)
    return bool(torch.isfinite(t).any())


def aggregate(rows: list[dict], names: Sequence[str], steps: Sequence[int], groups: dict[str, list[str]]) -> dict:
    """The study curves from per-frame rows: per run first, then over runs; overall and per group.

    A row whose value is None or entirely non-finite does not count; `n_windows_per_step` is the
    number of rows that did, `n_runs_per_step` the number of runs with at least one. List-valued
    metrics (histograms) average element-wise. A step with no sample is NaN (null in the JSON).
    """
    all_runs = sorted({r for ids in groups.values() for r in ids} | {row["run"] for row in rows})
    out = _curves(rows, names, steps, all_runs)
    out["groups"] = {g: {**_curves(rows, names, steps, sorted(ids)), "runs": sorted(ids)} for g, ids in groups.items() if ids}
    return out
