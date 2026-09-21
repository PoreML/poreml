"""Scoring a loader: per-sample metrics aggregated per run, then over runs.

Lives apart from `train` so `evaluate` (which the training loop calls for its periodic
evaluation) and `train` can both import it without a cycle.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .config import Config
from .data import RunRef, discover, load_split
from .data import resolve as resolve_runs
from .losses import per_sample_loss
from .metrics import ResolvedMetric, compute
from .metrics import trace as tracing
from .points import as_grid

Value = float | list[float]


def load_runs(cfg: Config) -> dict[str, list[RunRef]]:
    available = discover(cfg.data.root, cfg.data.campaign)
    return resolve_runs(load_split(cfg.data.split), available, require_finished=cfg.data.require_finished)


@dataclass
class Evaluation:
    """What scoring a loader produces.

    `summary` is the headline number per metric: the mean of per-run means, so every
    trajectory counts once however many frames it has (runs span tens to hundreds of
    frames; a mean over windows would be a mean over the longest runs). `per_run` keeps
    the per-run means so results can be stratified by geometry, M or theta offline, and
    `counts` records how many samples were scored and how many came back non-finite: a
    failed sample is counted, never silently dropped. `samples` is the log behind all of
    it: one row per scored sample — `run`, `t`, every metric, the loss when one was
    given, and everything the descriptors computed on the way (`metrics.trace`) — in
    loader order, so nothing computed is thrown away.
    """

    summary: dict[str, Value] = field(default_factory=dict)
    per_run: dict[str, dict[str, Value]] = field(default_factory=dict)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"metrics": self.summary, "per_run": self.per_run, "counts": self.counts}


def _as_value(x: Tensor) -> Value:
    return x.item() if x.ndim == 0 else x.tolist()


def evaluate_loader(
    model, loader, metrics: Sequence[ResolvedMetric], device, run_ids: Sequence[str], loss_fn=None
) -> Evaluation:
    """Score every sample in the loader and aggregate per run, then over runs.

    `loss_fn`, when given, is scored per sample as `loss` and aggregated like a metric —
    the validation loss of a training run, or the training loss on a test split. It never
    selects anything; `metrics[0]` does.

    Metrics are per-sample, so nothing here depends on batch size. `run_ids` maps each
    sample's `meta["run"]` back to its run ID; callers pass the IDs of the dataset's
    runs in order. A `(B, K)` raw distribution aggregates bin-wise and reports a list.
    NaN samples (nothing scored, zero target, a non-finite prediction) are skipped by
    `nanmean` and counted in `n_invalid`. Point-stream batches are scattered back to the
    grid by `as_grid` before scoring, so metrics see one layout.

    An empty loader reports the scalar `float("nan")` for every metric with zero counts,
    including the distribution ones that are lists otherwise: with no batch there is no
    `K` to shape a list of NaNs with.
    """
    model.eval()
    names = [m.name for m in metrics] + (["loss"] if loss_fn is not None else [])
    values: dict[str, dict[int, list[Tensor]]] = {name: defaultdict(list) for name in names}
    out = Evaluation()
    with torch.no_grad():
        for inputs, target, mask, meta in loader:
            inputs, target, mask = inputs.to(device), target.to(device), mask.to(device)
            pred, target, mask = as_grid(inputs, model(inputs), target, mask)
            with tracing.tracing() as trace:
                scored = compute(metrics, pred, target, mask)
            if loss_fn is not None:
                scored["loss"] = per_sample_loss(loss_fn, pred.float(), target, mask)
            scored = {name: value.detach().cpu() for name, value in scored.items()}
            extras = {name: value.cpu() for name, value in trace.values.items()}
            for name, value in scored.items():
                for run_idx, sample in zip(meta["run"].tolist(), value, strict=True):
                    values[name][run_idx].append(sample)
            for i, (run_idx, t) in enumerate(zip(meta["run"].tolist(), meta["t"].tolist(), strict=True)):
                row: dict[str, Any] = {"run": run_ids[run_idx], "t": t}
                row.update({name: _as_value(value[i]) for name, value in scored.items()})
                row.update({name: _as_value(value[i]) for name, value in extras.items()})
                out.samples.append(row)

    for name, by_run in values.items():
        per_run: dict[str, Value] = {}
        run_means: list[Tensor] = []
        n_samples = n_invalid = 0
        for run_idx in sorted(by_run):
            samples = torch.stack(by_run[run_idx])
            n_samples += samples.shape[0]
            n_invalid += int((~torch.isfinite(samples).reshape(samples.shape[0], -1).all(dim=1)).sum())
            mean = torch.nanmean(samples, dim=0)
            per_run[run_ids[run_idx]] = _as_value(mean)
            run_means.append(mean)
        summary = torch.nanmean(torch.stack(run_means), dim=0) if run_means else torch.tensor(float("nan"))
        out.summary[name] = _as_value(summary)
        out.per_run[name] = per_run
        out.counts[name] = {"n_runs": len(by_run), "n_samples": n_samples, "n_invalid": n_invalid}
    return out
