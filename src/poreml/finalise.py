"""The rollout-selected checkpoint of a training run (2026-09-10).

A run selects `best.pt` by the one-step primary metric every epoch, and rolls the epoch out
in its periodic eval. After training, the epoch whose *rollout* was best is what a
benchmark number should be reported on — push-forward fine-tunes trade one-step accuracy
for rollout accuracy, so the two choices disagree — and this module makes that choice:

- `select` reads `eval/metrics.csv` and picks the evaluated epoch with the lowest (or
  highest, per the metric) `rollout/<primary>` among the epochs whose weights are still on
  disk (`candidate_checkpoints`: `ckpts/epoch_<k>.pt`, else `best.pt` for the one-step
  best epoch, else `last.pt` for the last one — a run trained before per-epoch
  checkpoints existed still has those two);
- `mark` copies the winner to `ckpts/best_rollout.pt` (a copy, not a link: the studies
  hash the file);
- `finalise` does both, then runs the full evaluation of the marked checkpoint under
  `<run_dir>/final/`: the val pass with every metric (the mesh family included) at the
  config's `eval.stride`, the rollout with every step scored, the render when the config
  asks for it, and `final/metrics.json`; `run_meta.json` records the choice under
  `progress.best_rollout` and the stage under `progress.final`.

`train` calls `finalise` when it ends; `poreml finalise <run_dir>` applies it to a run that
finished under the old protocol or whose final stage failed. The stage is reporting: it
never changes the run's status, and an error is recorded rather than raised by `train`.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any

from .config import Config
from .metrics import json_safe

log = logging.getLogger("poreml")

FINAL_DIR = "final"
MARK = "best_rollout.pt"
_EPOCH_FILE = re.compile(r"epoch_(\d+)\.pt$")


def candidate_checkpoints(run_dir: Path | str) -> dict[int, Path]:
    """Epoch -> checkpoint file for every epoch whose weights the run still holds."""
    run_dir = Path(run_dir)
    ckpts = run_dir / "ckpts"
    out: dict[int, Path] = {}
    meta_path = run_dir / "run_meta.json"
    progress = json.loads(meta_path.read_text()).get("progress", {}) if meta_path.is_file() else {}
    best, last = progress.get("best") or {}, progress.get("last") or {}
    if (ckpts / "last.pt").is_file():
        epoch = last.get("epoch", (progress.get("epochs_done") or 0) - 1)
        if isinstance(epoch, int) and epoch >= 0:
            out[epoch] = ckpts / "last.pt"
    if (ckpts / "best.pt").is_file() and isinstance(best.get("epoch"), int):
        out[best["epoch"]] = ckpts / "best.pt"
    for path in ckpts.glob("epoch_*.pt"):  # the per-epoch file is the canonical copy of its epoch
        match = _EPOCH_FILE.search(path.name)
        if match:
            out[int(match.group(1))] = path
    return dict(sorted(out.items()))


def _numeric(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return None if math.isnan(value) else value


def select(run_dir: Path | str, primary: str, higher_is_better: bool = False) -> dict[str, Any] | None:
    """The rollout-best evaluated epoch that still has weights, or None when the run has no
    periodic rollout rows (or none of the evaluated epochs has a checkpoint left).

    Returns `{"epoch", "rollout/<primary>", "checkpoint", "candidates"}`; ties go to the
    earlier epoch, so a plateau does not favour the most-trained checkpoint by accident.
    """
    run_dir = Path(run_dir)
    csv_path = run_dir / "eval" / "metrics.csv"
    if not csv_path.is_file():
        return None
    column = f"rollout/{primary}"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    candidates = candidate_checkpoints(run_dir)
    scored = []
    for row in rows:
        epoch, value = _numeric(row.get("epoch")), _numeric(row.get(column))
        if epoch is None or value is None or int(epoch) not in candidates:
            continue
        scored.append((int(epoch), value))
    if not scored:
        return None
    epoch, value = min(scored, key=lambda ev: ((-ev[1] if higher_is_better else ev[1]), ev[0]))
    return {"epoch": epoch, column: value, "checkpoint": str(candidates[epoch]), "candidates": len(scored)}


def mark(run_dir: Path | str, checkpoint: Path | str) -> Path:
    """Copy `checkpoint` to `<run_dir>/ckpts/best_rollout.pt` (sibling + rename: never torn)."""
    target = Path(run_dir) / "ckpts" / MARK
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(checkpoint, tmp)
    tmp.replace(target)
    return target


def progress_metric(names: list[str] | tuple[str, ...], primary: str) -> str:
    """The metric the progress plot follows: `rel_mae@phi` when the config scores it, else
    the first relative error, else the primary metric."""
    if "rel_mae@phi" in names:
        return "rel_mae@phi"
    return next((n for n in names if n.startswith("rel_mae@")), primary)


def _read_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run_meta.json").read_text())


def _write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    tmp = run_dir / "run_meta.json.tmp"
    tmp.write_text(json.dumps(json_safe(meta), indent=2))
    tmp.replace(run_dir / "run_meta.json")


def finalise(run_dir: Path | str, cfg: Config | None = None, render: bool | None = None) -> dict[str, Any] | None:
    """Select, mark and fully evaluate the rollout-best checkpoint of `run_dir`.

    `cfg` defaults to the run's saved `config.yaml`; `render` overrides `cfg.train.eval.render`.
    Returns the `progress.final` record (None when there was nothing to select on). The
    selection and the mark are written before the evaluation starts, so a failed or
    interrupted evaluation leaves the choice on disk and `progress.final.error` says why.
    """
    from .evaluate import evaluate, render_rollout, rollout_split  # lazy, as train.py imports them
    from .metrics import resolve
    from .models import representation_of
    from .scoring import load_runs
    from .tasks import build_task

    run_dir = Path(run_dir)
    cfg = Config.from_yaml(run_dir / "config.yaml") if cfg is None else cfg
    task = build_task(cfg.task, representation_of(cfg.model))
    metrics = resolve(cfg.metrics if cfg.metrics is not None else task.metrics, task.target_channels)
    primary = metrics[0]
    meta = _read_meta(run_dir)
    progress = meta.setdefault("progress", {})
    chosen = select(run_dir, primary.name, primary.higher_is_better)
    progress["best_rollout"] = chosen
    if chosen is None:
        progress["final"] = None
        _write_meta(run_dir, meta)
        log.info("%s: no periodic rollout to select a checkpoint on; final stage skipped", run_dir)
        return None
    marked = mark(run_dir, chosen["checkpoint"])
    out = run_dir / FINAL_DIR
    record: dict[str, Any] = {"epoch": chosen["epoch"], "dir": str(out), "checkpoint": str(marked), "error": None}
    progress["final"] = record
    _write_meta(run_dir, meta)
    log.info(
        "%s: rollout-best epoch %d (%s = %.6g of %d candidates) marked as %s",
        run_dir,
        chosen["epoch"],
        f"rollout/{primary.name}",
        chosen[f"rollout/{primary.name}"],
        chosen["candidates"],
        marked.name,
    )

    ev = cfg.train.eval
    do_render = ev.render if (render is None and ev is not None) else bool(render)
    try:
        stride = ev.stride if ev is not None else None
        results = evaluate(cfg, ckpt=marked, split="val", out_dir=out, stride=stride)
        summary = rollout_split(cfg, ckpt=marked, split="val", out_dir=out)
        if do_render:
            runs = load_runs(cfg)
            render_rollout(cfg, ckpt=marked, run_id=runs["val"][0].run_id, out_dir=out, split="val")
        payload = {
            "epoch": chosen["epoch"],
            "checkpoint": str(marked),
            "selection": chosen,
            "stride": stride,
            "horizon": summary.horizon,
            "one_step": results.summary,
            "rollout_at_horizon": summary.at_horizon,
            "rendered": do_render,
        }
        tmp = out / "metrics.json.tmp"
        tmp.write_text(json.dumps(json_safe(payload), indent=2))
        tmp.replace(out / "metrics.json")
    except Exception as e:  # reporting only: the run is trained; say what failed and leave the mark
        import traceback

        record["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
        log.exception("%s: final stage failed", run_dir)
    _write_meta(run_dir, meta)
    return record
