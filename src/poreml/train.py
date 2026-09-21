"""Training loop.

Deliberately one readable loop: no distributed, no experiment tracker. Each of those
has an obvious slot when a real task asks for it.
"""

import json
import logging
import math
import random
import signal
import threading
import time
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import finalise, provenance, pushforward
from .config import Config, resolve_device
from .losses import loss_function, masked_h1, masked_mse  # noqa: F401  (re-exported: tests and users import them here)
from .metrics import is_better, json_safe, resolve, validation_metrics
from .models import build_model, count_parameters, precision_of, representation_of
from .precision import autocast as precision_autocast
from .precision import configure as precision_configure
from .precision import inference_model
from .scoring import (  # noqa: F401  (re-exported: tests and users import them here)
    Evaluation,
    Value,
    evaluate_loader,
    load_runs,
)
from .tables import append_csv, append_rows  # noqa: F401  (re-exported)
from .tasks import build_task

log = logging.getLogger("poreml")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_run_dir(cfg: Config, now: str | None = None) -> Path:
    """Create a fresh run directory, never an existing one.

    Two runs of the same name started in the same second — a scripted sweep over a small
    dataset — would otherwise share a directory, interleaving their epoch lines in one
    metrics.jsonl and overwriting each other's checkpoints. A colliding name gets a
    counter suffix instead.
    """
    stamp = now or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = Path(cfg.out_dir) / f"{cfg.name}_{stamp}"
    path, collisions = base, 0
    while True:
        try:
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            collisions += 1
            path = base.with_name(f"{base.name}-{collisions}")
    (path / "ckpts").mkdir()
    return path


class Preempted(RuntimeError):
    """Raised by `train` after SIGTERM: the resume state is on disk and the run says `preempted`."""


def save_atomic(obj: Any, path: Path) -> None:
    """torch.save to a sibling, then rename: a kill mid-write never leaves a torn checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def epoch_permutation(n: int, seed: int, epoch: int, skip: int = 0) -> list[int]:
    """The shuffle order of one epoch, a pure function of (seed, epoch).

    Independent of the global RNG, so a resumed epoch replays exactly the order the
    interrupted one was walking and `skip` drops the windows it had already trained on.
    """
    gen = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(n, generator=gen).tolist()[skip:]


# `eval` is here too: the periodic eval scores, rolls out and renders under a forked RNG and
# changes nothing about what is trained, so a stopped run may continue with a cheaper one.
OPERATIONAL_KNOBS = ("num_workers", "device", "log_every", "checkpoint_every", "eval")


def training_identity(cfg: Config) -> dict[str, Any]:
    """The config minus the knobs that change nothing about what is trained.

    Loader count, device, logging/checkpoint cadence and the periodic eval may differ
    between a run's segments — lowering the workers when the filesystem chokes must not
    strand every stopped run — everything else must match exactly for a resume to be honest.
    """
    payload = cfg.model_dump(mode="json")
    for knob in OPERATIONAL_KNOBS:
        payload["train"].pop(knob, None)
    # A config that leaves `model.precision` unset trains in the model's default: a saved
    # config.yaml written before the project default was stated in the configs (2026-09-10)
    # names the same training as one that states it.
    payload["model"]["precision"] = precision_of(cfg.model)
    return payload


def find_resumable(cfg: Config) -> Path | None:
    """The newest run under `cfg.out_dir` that stopped short and was trained under this
    exact config, or None. Runs of a different config are skipped: resuming them would
    silently continue something else."""
    out_dir = Path(cfg.out_dir)
    if not out_dir.is_dir():
        return None
    wanted = training_identity(cfg)
    for meta_path in sorted(out_dir.glob("*/run_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        run_dir = meta_path.parent
        if not (run_dir / "ckpts" / "resume.pt").is_file():
            continue
        if json.loads(meta_path.read_text()).get("status") == "finished":
            continue
        if training_identity(Config.from_yaml(run_dir / "config.yaml")) != wanted:
            log.warning("%s stopped short but was trained under a different config; not resuming it", run_dir)
            continue
        return run_dir
    return None


def load_init(model: torch.nn.Module, path: Path, device: str) -> dict[str, Any]:
    """Start `model` from the weights of a checkpoint (a run's `ckpts/best.pt` or `last.pt`):
    a fine-tune. The load is strict, so a checkpoint of another architecture fails here, and
    what was loaded is returned for run_meta.json."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return {"path": str(path), "epoch": ckpt.get("epoch"), "sha256": provenance.sha256_of(path)}


def _rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())  # resume.pt is loaded with map_location=device; generators want CPU ByteTensors
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    """Rewrite run_meta.json atomically (write to a sibling, then replace)."""
    tmp = run_dir / "run_meta.json.tmp"
    tmp.write_text(json.dumps(json_safe(meta), indent=2))
    tmp.replace(run_dir / "run_meta.json")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in values.items()}


def _plot_progress(run_dir: Path, metric: str) -> None:
    """`<run_dir>/progress.svg`: the one-step `metric` per epoch and its rollout twin per periodic
    eval. Reporting only — a missing viz group or an odd row must never touch training."""
    try:
        from . import viz

        viz.plot_progress(run_dir / "metrics.csv", run_dir / "eval" / "metrics.csv", run_dir / "progress.svg", metric)
    except ImportError as e:
        log.warning("progress.svg skipped: %s", e)
    except Exception:
        log.exception("progress.svg skipped")


def _periodic_eval(
    cfg: Config, run_dir: Path, epoch: int, step: int, ckpt: Path, val_run_ids: Sequence[str], val: dict[str, Any] | None = None
) -> dict:
    """The periodic evaluation of `ckpt`: the val split rolled out under `<run_dir>/eval/epoch_<e>/`.

    Since 2026-09-10 this is the rollout alone (`rollout_split`, scored with the validation
    metrics — nothing `eval_only`, so no meshing during training — at `rollout_stride`): the
    one-step columns of the `eval/metrics.csv` row are the epoch's own selection-pass metrics
    (`val`, the same samples at `train.val_stride`), and the full val pass, the `eval_only`
    metrics and the render belong to the final stage on the rollout-selected checkpoint
    (`finalise.py`). `eval/curves.svg` is redrawn from the CSV.
    """
    from .evaluate import rollout_split  # evaluate imports scoring, not train: no cycle

    ev = cfg.train.eval
    out = run_dir / "eval" / f"epoch_{epoch:03d}"
    device = resolve_device(cfg.train.device)
    row: dict[str, Any] = {"epoch": epoch, "step": step, **(val or {})}
    # This is reporting only: any dropout/init inside a rolled-out model draws from the
    # global torch RNG, so without forking it a run with
    # `eval:` set would train on a different shuffle order from epoch 2 on than the same
    # config with `eval: null` — a "reporting only" step must not change what is trained.
    # `devices=[]` on CPU avoids fork_rng's "no CUDA devices" warning; on CUDA, torch's
    # default (`devices=None`) forks the RNG state of *every visible* CUDA device and
    # warns when there is more than one, so we pin it to just the process's current device.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []):
        if ev.rollout:
            summary = rollout_split(cfg, ckpt=ckpt, split="val", out_dir=out, eval_stride=ev.rollout_stride, eval_only=False)
            row.update(_prefixed("rollout/", summary.at_horizon))
    (run_dir / "eval").mkdir(exist_ok=True)
    with (run_dir / "eval" / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(json_safe(row)) + "\n")
    append_csv(run_dir / "eval" / "metrics.csv", row)
    try:
        from . import viz

        viz.plot_eval_curves(run_dir / "eval" / "metrics.csv", run_dir / "eval" / "curves.svg")
    except ImportError as e:
        log.warning("eval/curves.svg skipped: %s", e)
    log.info("epoch %d  periodic eval written to %s", epoch, out)
    return {"epoch": epoch, "dir": str(out)}


def train(cfg: Config, resume: Path | None = None) -> Path:
    """Run training and return the run directory holding all its artifacts.

    `resume` names a run directory that stopped short (preempted, timed out, crashed):
    training continues inside it from `ckpts/resume.pt` — same model, optimiser,
    scheduler, epoch/batch position, running loss and RNG — and its metrics files keep
    growing. The run's saved `config.yaml` must equal `cfg`.
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.train.device)

    # Metrics are resolved before a run directory exists. A metric typo has to fail here
    # rather than inside `compute` at the end of epoch 0 — hours into a real run — and
    # a config that cannot run must not leave an empty run directory behind. Same reason
    # for checking the viz group here when periodic rendering is on: without this, a
    # missing pyvista/matplotlib/PIL only surfaces after epoch `every` finishes training
    # (potentially hours of GPU time and a full rollout), inside `render_rollout`.
    if cfg.train.eval is not None and cfg.train.eval.render:
        from . import viz

        viz.require_viz()

    # Same fail-fast rule as the viz check: an impossible config must die before a run
    # directory exists, not at the first batch.
    if cfg.train.loss == "h1" and representation_of(cfg.model) != "voxel":
        raise ValueError("train.loss: h1 needs the voxel stream — finite-difference gradients are undefined on a point cloud")
    if cfg.train.init_from is not None and not Path(cfg.train.init_from).is_file():
        raise FileNotFoundError(f"train.init_from: no checkpoint at {cfg.train.init_from}")

    runs = load_runs(cfg)
    task = build_task(cfg.task, representation_of(cfg.model))
    metrics = resolve(cfg.metrics if cfg.metrics is not None else task.metrics, task.target_channels)
    primary = metrics[0]
    progress_metric_name = finalise.progress_metric([m.name for m in metrics], primary.name)

    state: dict[str, Any] | None = None
    if resume is not None:
        run_dir = Path(resume)
        saved = Config.from_yaml(run_dir / "config.yaml")
        if training_identity(saved) != training_identity(cfg):
            raise ValueError(f"cannot resume {run_dir}: its saved config.yaml trains something else than the config given")
        state = torch.load(run_dir / "ckpts" / "resume.pt", map_location=device, weights_only=False)
    else:
        run_dir = build_run_dir(cfg)
        cfg.to_yaml(run_dir / "config.yaml")

    # Push-forward needs the true frames `steps` further ahead as targets (`pushforward.py`);
    # the default `future=1` keeps every window exactly as the benchmark runs saw it.
    push = cfg.train.push_forward
    train_dataset = task.dataset(runs["train"], training=True, future=1 + (push.steps if push is not None else 0))
    n_fields = task.out_channels
    n_static = task.in_channels - task.history * n_fields  # solid mask or geometry, then the conditions
    batch_size = cfg.train.batch_size
    batches_per_epoch = math.ceil(len(train_dataset) / batch_size)
    if cfg.train.max_batches is not None:
        batches_per_epoch = min(batches_per_epoch, cfg.train.max_batches)
    total_steps = max(1, cfg.train.epochs * batches_per_epoch)

    def epoch_loader(epoch: int, skip_batches: int) -> DataLoader:
        # A fixed permutation per epoch, truncated to the batches this epoch still owes:
        # the loader itself is what makes a resumed epoch identical to an uninterrupted one.
        indices = epoch_permutation(len(train_dataset), cfg.seed, epoch, skip=skip_batches * batch_size)
        indices = indices[: (batches_per_epoch - skip_batches) * batch_size]
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=indices,
            num_workers=cfg.train.num_workers,
            collate_fn=train_dataset.collate,
        )

    val_dataset = task.dataset(runs["val"], stride=cfg.train.val_stride)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=cfg.train.num_workers, collate_fn=val_dataset.collate
    )
    val_run_ids = [r.run_id for r in runs["val"]]

    model = build_model(cfg.model, task.in_channels, task.out_channels).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # bf16 autocasts the forward pass: the loss is taken on an fp32 copy of the output and
    # weights and optimiser states are fp32 master copies. The per-epoch selection pass,
    # periodic eval, rollout and `poreml eval` run in the same precision (`precision.py`):
    # precision is a model property, and a benchmark number is recorded with it.
    precision = precision_of(cfg.model)
    precision_configure(precision)
    autocast = precision_autocast(precision, device)
    val_model = inference_model(model, precision, device)
    # A parameter-free baseline (persistence) is still worth "training": it establishes
    # the floor on exactly the same loader and metrics as everything else.
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay) if params else None
    loss_fn = loss_function(cfg.train.loss)
    scheduler, scheduler_per_batch = None, False
    if optimizer is not None and cfg.train.epochs > 0:
        if cfg.train.lr_schedule == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.train.lr, total_steps=total_steps)
            scheduler_per_batch = True
        elif cfg.train.lr_schedule == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.lr_step_size, gamma=cfg.train.lr_gamma)

    # run_meta.json — what this run is, in one place, like the solver's: config, data, model,
    # provenance, and progress that is refreshed every epoch so a half-finished run is
    # still fully described. `status` is running / finished / preempted / failed;
    # `run.segments` lists every (re)start and where it picked up.
    start_epoch, batches_done, step, best, total_loss, elapsed, pushed = 0, 0, 0, float("nan"), 0.0, 0.0, 0
    if state is not None:
        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch, batches_done, step = state["epoch"], state["batches_done"], state["step"]
        best, total_loss, elapsed = state["best"], state["total_loss"], state["elapsed"]
        pushed = state.get("pushed", 0)
        _set_rng_state(state["rng"])
        meta: dict[str, Any] = json.loads((run_dir / "run_meta.json").read_text())
        meta["status"] = "running"
        meta["run"].update(end_time=None, error=None)
        log.info("resuming %s at epoch %d, batch %d (step %d)", run_dir, start_epoch, batches_done, step)
    else:
        init = load_init(model, Path(cfg.train.init_from), device) if cfg.train.init_from is not None else None
        prov = provenance.collect(cfg, device)
        meta = {
            "name": cfg.name,
            "status": "running",
            "run": {"start_time": _now(), "end_time": None, "run_dir": str(run_dir), "error": None, "segments": []},
            "config": cfg.model_dump(mode="json"),
            "data": {
                "root": prov["data_root"],
                "campaign": cfg.data.campaign,
                "split": prov["split"],
                "runs": {section: [r.summary() for r in refs] for section, refs in runs.items()},
                "n_windows": {"train": len(train_dataset), "val": len(val_dataset)},
            },
            "task": {
                "name": cfg.task.name,
                "history": cfg.task.history,
                "params": cfg.task.params,
                "representation": task.representation,
                "in_channels": task.in_channels,
                "out_channels": task.out_channels,
            },
            "model": {
                "name": cfg.model.name,
                "params": cfg.model.params,
                "in_channels": task.in_channels,
                "out_channels": task.out_channels,
                "n_parameters": count_parameters(model),
                "precision": precision,
                **({"init_from": init} if init is not None else {}),
            },
            "metrics": [{"name": m.name, **m.spec.model_dump(mode="json", exclude={"name"})} for m in metrics],
            "primary_metric": metrics[0].name,
            "provenance": prov,
            "progress": {
                "epochs_done": 0,
                "epochs_total": cfg.train.epochs,
                "steps": 0,
                "best": None,
                "last": None,
                "evals": [],
                "last_eval": None,
                "best_rollout": None,
                "final": None,
            },
        }
    meta["run"]["segments"].append(
        {"start_time": _now(), "epoch": start_epoch, "batches_done": batches_done, "precision": precision}
    )
    write_meta(run_dir, meta)

    # SIGTERM is what SLURM sends on preemption and, with --signal, before the time limit.
    # The handler only raises a flag; the loop saves the resume state at the next batch
    # boundary so the checkpoint is always a consistent post-step snapshot.
    stop_requested = False

    def on_sigterm(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        log.warning("SIGTERM received; saving the resume state at the next batch boundary")

    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.signal(signal.SIGTERM, on_sigterm)

    def save_resume(epoch: int, batches_done: int, total_loss: float, elapsed: float, pushed: int = 0) -> None:
        save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "batches_done": batches_done,
                "step": step,
                "best": best,
                "total_loss": total_loss,
                "elapsed": elapsed,
                "pushed": pushed,
                "rng": _rng_state(),
            },
            run_dir / "ckpts" / "resume.pt",
        )

    def preempt(epoch: int, batches_done: int, total_loss: float, elapsed: float, pushed: int = 0) -> None:
        save_resume(epoch, batches_done, total_loss, elapsed, pushed)
        meta["status"] = "preempted"
        meta["run"]["end_time"] = _now()
        write_meta(run_dir, meta)
        raise Preempted(f"{run_dir} preempted at epoch {epoch}, batch {batches_done}; resume state saved")

    resume_from = (batches_done, total_loss, elapsed, pushed)
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            batches_done, total_loss, elapsed, pushed = resume_from
            resume_from = (0, 0.0, 0.0, 0)
            started = time.perf_counter() - elapsed
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            n_batches = batches_done
            last_save = time.perf_counter()
            for inputs, target, mask, _ in epoch_loader(epoch, batches_done):
                inputs, target, mask = inputs.to(device), target.to(device), mask.to(device)
                if push is not None:
                    # `n` ungraded steps on the model's own output (0: a noised, teacher-forced
                    # batch), then the graded step below against the true frame n + 1 ahead.
                    n = pushforward.n_pushes(push, cfg.seed, step, total_steps)
                    if n == 0:
                        inputs = pushforward.perturb(
                            inputs, push.noise, n_static, pushforward.generator(cfg.seed, step, device)
                        )
                    else:
                        inputs = pushforward.unroll(model, train_dataset, inputs, n, n_static, n_fields, autocast)
                        pushed += 1
                    target = target[:, n]
                with autocast:
                    pred = model(inputs)
                loss = loss_fn(pred.float(), target, mask)
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    if cfg.train.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                    optimizer.step()
                    if scheduler is not None and scheduler_per_batch:
                        scheduler.step()
                total_loss += loss.item()
                n_batches += 1
                step += 1
                last_batch = n_batches == batches_per_epoch
                if n_batches % cfg.train.log_every == 0 or last_batch:
                    elapsed = time.perf_counter() - started
                    append_csv(
                        run_dir / "train_log.csv",
                        {
                            "step": step,
                            "epoch": epoch,
                            "batch": n_batches,
                            "loss": loss.item(),
                            "mean_loss": total_loss / n_batches,
                            "elapsed_seconds": elapsed,
                        },
                    )
                    if not last_batch:
                        log.info("epoch %d  batch %d  loss %.5f  %.0fs", epoch, n_batches, total_loss / n_batches, elapsed)
                if stop_requested:
                    preempt(epoch, n_batches, total_loss, time.perf_counter() - started, pushed)
                every = cfg.train.checkpoint_every
                if every is not None and not last_batch and time.perf_counter() - last_save >= every:
                    save_resume(epoch, n_batches, total_loss, time.perf_counter() - started, pushed)
                    last_save = time.perf_counter()

            # The lr this epoch actually trained at, read before an epoch-stepped
            # scheduler moves it — metrics.csv's lr column reports what was used.
            epoch_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else cfg.train.lr
            if scheduler is not None and not scheduler_per_batch:
                scheduler.step()

            validation = evaluate_loader(
                val_model, val_loader, validation_metrics(metrics), device, val_run_ids, loss_fn=loss_fn
            )
            val = dict(validation.summary)
            val_loss = val.pop("loss")
            peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.startswith("cuda") else None
            record = {
                "epoch": epoch,
                "step": step,
                "train_loss": total_loss / max(n_batches, 1),
                "val_loss": val_loss,
                "epoch_seconds": time.perf_counter() - started,
                "peak_memory_mb": peak_mb,
                "lr": epoch_lr,
                **({"push_frac": pushed / max(n_batches, 1)} if push is not None else {}),
                **val,
            }
            with (run_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(json_safe(record)) + "\n")
            append_csv(run_dir / "metrics.csv", record)
            # The log behind the row: every validation sample with every value computed for it.
            append_rows(run_dir / "val_samples.csv", [{"epoch": epoch, **row} for row in validation.samples])
            log.info(
                "epoch %d done  train_loss %.5f  val %s %s  %.0fs  peak %s MB",
                epoch,
                record["train_loss"],
                primary.name,
                f"{val[primary.name]:.5f}" if isinstance(val[primary.name], float) else "-",
                record["epoch_seconds"],
                "-" if peak_mb is None else f"{peak_mb:.0f}",
            )

            ckpt = {
                "model": model.state_dict(),
                "config": cfg.model.model_dump(mode="json"),
                "epoch": epoch,
                "metrics": val,
            }
            save_atomic(ckpt, run_dir / "ckpts" / "last.pt")
            # Every epoch keeps its weights (7-45 MB): the rollout-best epoch is chosen only
            # after training (`finalise.py`) and must still exist then.
            save_atomic(ckpt, run_dir / "ckpts" / f"epoch_{epoch:03d}.pt")
            if is_better(primary, val[primary.name], best) or not (run_dir / "ckpts" / "best.pt").exists():
                best = val[primary.name]
                save_atomic(ckpt, run_dir / "ckpts" / "best.pt")
                meta["progress"]["best"] = {"epoch": epoch, primary.name: best}
            meta["progress"].update(epochs_done=epoch + 1, steps=step, last={"epoch": epoch, **record})
            # The epoch is complete: the resume state now points at the next epoch's start,
            # written before the (long, reporting-only) periodic eval so a kill during it
            # costs nothing that was trained.
            save_resume(epoch + 1, 0, 0.0, 0.0)
            write_meta(run_dir, meta)
            if stop_requested:
                preempt(epoch + 1, 0, 0.0, 0.0)
            _plot_progress(run_dir, progress_metric_name)

            ev = cfg.train.eval
            if ev is not None and ((epoch + 1) % ev.every == 0 or epoch + 1 == cfg.train.epochs):
                # Periodic eval is reporting, not training: a bug in a metric, renderer or
                # rollout must never turn into a failed multi-hour run. Record the error
                # in progress and keep training; the fail-fast `viz.require_viz()` check
                # above (at the start of `train()`) is the deliberate early exception.
                try:
                    entry = _periodic_eval(cfg, run_dir, epoch, step, run_dir / "ckpts" / "last.pt", val_run_ids, val)
                except Exception as e:
                    log.exception("periodic eval at epoch %d failed; training continues", epoch)
                    out_dir = run_dir / "eval" / f"epoch_{epoch:03d}"
                    entry = {
                        "epoch": epoch,
                        "dir": str(out_dir),
                        "error": "".join(traceback.format_exception_only(type(e), e)).strip(),
                    }
                meta["progress"]["evals"].append(entry)
                meta["progress"]["last_eval"] = entry
                write_meta(run_dir, meta)
                _plot_progress(run_dir, progress_metric_name)
    except Preempted:
        raise
    except BaseException as e:
        meta["status"] = "failed"
        meta["run"]["end_time"] = _now()
        meta["run"]["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
        write_meta(run_dir, meta)
        raise
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)

    meta["status"] = "finished"
    meta["run"]["end_time"] = _now()
    write_meta(run_dir, meta)
    # The rollout-selected checkpoint and its full evaluation (`finalise.py`): after the
    # status, so a kill during this reporting stage leaves a finished run that
    # `poreml finalise <run_dir>` completes; a failure is recorded there, never raised.
    if cfg.train.eval is not None and cfg.train.eval.rollout:
        try:
            finalise.finalise(run_dir, cfg)
        except Exception:
            log.exception("final stage failed for %s; rerun it with `poreml finalise %s`", run_dir, run_dir)
    return run_dir
