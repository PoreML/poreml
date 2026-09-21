"""Roll the best checkpoint of every finished training out on its own validation split.

    uv run python util/inference/inference.py [TARGET ...] [--which auto|best|best_rollout|last]
                                         [--split val] [--stride 64] [--device cuda] [--dry-run] [--force]

A TARGET is a phase root (`case/train`, `case/train_push`), a campaign folder, a job folder
(`case/train/drainage/abupt_all`) or one run directory (`.../ckpts/<run>`); the default is
`case/train`. Every job resolves to its newest **finished** run (a run directory named directly
must be finished too); folders whose name starts with `_` (`_archive`, `_logs`) are never
searched. The checkpoint is `ckpts/best.pt` for a run under `case/train` and
`ckpts/best_rollout.pt` for one under `case/train_push` (`--which` pins one); a run whose
checkpoint is not there yet is `waiting`.

Each run is rolled out with its **own saved `config.yaml`** — never the current `configs/` —
by `poreml.inference.rollout_inference` (the modelling code lives there, not here): the studies'
window protocol on the validation split — every 64-step window from frame 0 at stride 64, full
windows only — with `forward_s` per step, `mae`/`rel_mae` on `phi` and `p` at every step of every
window in `<run_dir>/inference.csv`, the 12 keyframes per window under `<run_dir>/inference/frames/`,
`<run_dir>/inference/inference.json` as the sentinel (per run first, then over runs). States:
`waiting` (no checkpoint), `pending` (nothing stored), `resume` (windows stored, no sentinel),
`stale` (stored for another checkpoint — redone), `finished` (skipped unless `--force`). SIGTERM
stops after the window in hand and the script exits 75; rerunning resumes. Needs a GPU for
anything but the fixture:

    srun --gres=gpu:1 --cpus-per-task=8 uv run python util/inference/inference.py case/train/drainage

`--metric [--workers N]` is the CPU twin (`poreml.inference.metric_inference`, no GPU): the run's
**whole** metric list — the Minkowski and curvature descriptors the GPU stage leaves out — on the
keyframes stored above, written as `<run_dir>/inference/metrics_<split>.csv` / `.json` in the
studies' format, so the in-distribution descriptors follow the transfer tests' window protocol.
States there: `waiting` (no checkpoint), `no-frames` (the GPU stage has not finished for this
checkpoint), `metric` (to score), `scored` (a metrics JSON for this checkpoint under the current
`METRICS_VERSION`; skipped unless `--force`). `util/inference/submit.py --metric` runs it as a CPU array.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from poreml.provenance import sha256_of  # noqa: E402

DEFAULT_TARGET = REPO / "case" / "train"
WHICH = ("auto", "best", "best_rollout", "last")
PUSH_DIR = "train_push"
STAGE_DIR, SENTINEL = "inference", "inference.json"


def _status(run_dir: Path) -> str | None:
    try:
        return json.loads((run_dir / "run_meta.json").read_text()).get("status")
    except (OSError, ValueError):
        return None


def _is_run_dir(path: Path) -> bool:
    return (path / "run_meta.json").is_file() and (path / "config.yaml").is_file()


def _run_dirs_under(root: Path) -> list[Path]:
    """Every run directory below `root`, skipping `_`-prefixed folders (archives, logs, manifests)."""
    if _is_run_dir(root):
        return [root]
    out = []
    for meta in root.rglob("run_meta.json"):
        rel = meta.relative_to(root).parts
        if any(p.startswith("_") for p in rel):
            continue
        if _is_run_dir(meta.parent):
            out.append(meta.parent)
    return sorted(out)


def _recency(run_dir: Path) -> tuple[float, str]:
    """Newest first by run_meta.json mtime; the directory name (its `_YYYYMMDD-HHMMSS` suffix) breaks ties."""
    return (run_dir / "run_meta.json").stat().st_mtime, run_dir.name


def newest_finished(job_dir: Path) -> Path | None:
    """The newest finished run under `<job_dir>/ckpts/`, by `_recency` (as the push configs pick their base)."""
    runs = [r for r in _run_dirs_under(job_dir) if _status(r) == "finished"]
    if not runs:
        return None
    return max(runs, key=_recency)


def find_runs(targets: list[Path]) -> list[Path]:
    """The run directories to process: the newest finished run of every job under each target.

    A run directory named directly is taken as it is when finished (an older run can be
    rolled out on purpose); anything else is grouped by job (`<job>/ckpts/<run>`) and the
    newest finished run per job wins.
    """
    chosen: dict[Path, Path] = {}
    for target in targets:
        target = Path(target)
        if _is_run_dir(target):
            if _status(target) == "finished":
                chosen[target] = target
            continue
        for run in _run_dirs_under(target):
            if _status(run) != "finished":
                continue
            job = run.parent.parent if run.parent.name == "ckpts" else run.parent
            current = chosen.get(job)
            if current is None or _recency(run) > _recency(current):
                chosen[job] = run
    return sorted(set(chosen.values()))


def choose_checkpoint(run_dir: Path, which: str = "auto") -> Path | None:
    """`best.pt` under `case/train`, `best_rollout.pt` under `case/train_push` (`auto`), or the named one; None if absent."""
    if which not in WHICH:
        raise ValueError(f"--which must be one of {WHICH}, got {which!r}")
    if which == "auto":
        which = "best_rollout" if PUSH_DIR in run_dir.resolve().parts else "best"
    path = run_dir / "ckpts" / f"{which}.pt"
    return path if path.is_file() else None


def state(run_dir: Path, ckpt: Path | None) -> str:
    """`waiting` (no checkpoint), `finished` (sentinel for this checkpoint), `stale` (for another),
    `resume` (windows stored, no sentinel), else `pending`."""
    if ckpt is None:
        return "waiting"
    sentinel = run_dir / STAGE_DIR / SENTINEL
    if not sentinel.is_file():
        frames = run_dir / STAGE_DIR / "frames"
        return "resume" if frames.is_dir() and any(frames.glob("*/w*.h5")) else "pending"
    try:
        stored = json.loads(sentinel.read_text()).get("checkpoint_sha256")
    except (OSError, ValueError):
        return "pending"
    return "finished" if stored == sha256_of(ckpt) else "stale"


def metric_state(run_dir: Path, ckpt: Path | None, split: str = "val") -> str:
    """`waiting` (no checkpoint), `no-frames` (GPU stage not finished for it), `scored` (metrics JSON for this
    checkpoint under the current `METRICS_VERSION`), else `metric`."""
    from poreml.metrics import METRICS_VERSION

    gpu = state(run_dir, ckpt)
    if gpu != "finished":
        return "waiting" if gpu == "waiting" else "no-frames"
    try:
        block = json.loads((run_dir / STAGE_DIR / f"metrics_{split}.json").read_text()).get("metric", {})
    except (OSError, ValueError):
        return "metric"
    current = block.get("checkpoint_sha256") == sha256_of(ckpt) and block.get("metrics_version") == METRICS_VERSION
    return "scored" if current else "metric"


def run_metric(todo: list[tuple[Path, Path]], split: str, workers: int) -> int:
    """The CPU stage over `todo`; one broken run does not stop the sweep."""
    from poreml.config import Config
    from poreml.inference import metric_inference

    failed = 0
    for run, ckpt in todo:
        started = time.perf_counter()
        rel = run.relative_to(REPO) if run.is_relative_to(REPO) else run
        print(f"\n== {rel}  {ckpt.name}  split {split}  metric", flush=True)
        try:
            cfg = Config.from_yaml(run / "config.yaml")
            payload = metric_inference(cfg, ckpt=ckpt, split=split, out_dir=run, workers=workers)
        except Exception as e:
            failed += 1
            print(f"!! failed: {type(e).__name__}: {e}", flush=True)
            continue
        print(
            f"   {payload['n_windows']} windows, {payload['n_frames']} frames, {len(payload['metrics_spec'])} metrics,"
            f" {workers} workers, {time.perf_counter() - started:.0f} s",
            flush=True,
        )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", type=Path, default=[DEFAULT_TARGET], help="phase roots, job or run dirs")
    parser.add_argument(
        "--which", choices=WHICH, default="auto", help="checkpoint: auto = best.pt (train) / best_rollout.pt (train_push)"
    )
    parser.add_argument("--split", default="val", help="split section of each run's own config (default val)")
    parser.add_argument("--device", default=None, help="override train.device of each run's config (e.g. cuda, cuda:1, cpu)")
    parser.add_argument("--horizon", type=int, default=None, help="override rollout.horizon")
    parser.add_argument("--stride", type=int, default=None, help="frames between window starts (default: the horizon)")
    parser.add_argument("--keyframes", type=int, default=None, help="override rollout.keyframes (frames stored per run)")
    parser.add_argument("--fields", default="phi,p", help="fields to score with mae / rel_mae (default phi,p)")
    parser.add_argument("--metric", action="store_true", help="CPU stage: the run's whole metric list on the stored keyframes")
    parser.add_argument("--workers", type=int, default=1, help="--metric: scoring processes")
    parser.add_argument("--force", action="store_true", help="redo finished runs")
    parser.add_argument("--dry-run", action="store_true", help="list the runs and their state, roll nothing out")
    args = parser.parse_args(argv)

    runs = find_runs(args.targets)
    if not runs:
        print("no finished run under", ", ".join(str(t) for t in args.targets))
        return 1
    plan: list[tuple[Path, Path | None, str]] = []
    for run in runs:
        ckpt = choose_checkpoint(run, args.which)
        st = metric_state(run, ckpt, args.split) if args.metric else state(run, ckpt)
        plan.append((run, ckpt, st))
        rel = run.relative_to(REPO) if run.is_relative_to(REPO) else run
        print(f"{st:9s} {rel}  {ckpt.name if ckpt else '-'}")
    if args.metric:
        todo = [(r, c) for r, c, st in plan if c is not None and (st == "metric" or (st == "scored" and args.force))]
    else:
        todo = [(r, c) for r, c, st in plan if c is not None and (st != "finished" or args.force)]
    gpu_states = ("waiting", "pending", "resume", "stale", "finished")
    names = ("waiting", "no-frames", "metric", "scored") if args.metric else gpu_states
    counts = {st: sum(1 for _, _, s in plan if s == st) for st in names}
    verb = "to score" if args.metric else "to roll out"
    print(f"\n{len(plan)} runs: " + ", ".join(f"{n} {st}" for st, n in counts.items() if n) + f"; {len(todo)} {verb}")
    if args.dry_run or not todo:
        return 0
    if args.metric:
        return run_metric(todo, args.split, args.workers)

    from poreml.config import Config
    from poreml.inference import rollout_inference
    from poreml.rollout import Interrupted

    # one line per rolled-out run (t0, horizon, forward s/step, the criterion at the horizon), timestamped
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())
    failed = 0
    for run, ckpt in todo:
        cfg = Config.from_yaml(run / "config.yaml")
        if args.device is not None:
            cfg = cfg.updated(train={"device": args.device})
        started = time.perf_counter()
        print(f"\n== {run.relative_to(REPO) if run.is_relative_to(REPO) else run}  {ckpt.name}  split {args.split}", flush=True)
        try:
            payload = rollout_inference(
                cfg,
                ckpt=ckpt,
                split=args.split,
                out_dir=run,
                horizon=args.horizon,
                stride=args.stride,
                keyframes=args.keyframes,
                fields=fields,
                force=args.force,
            )
        except Interrupted as e:
            print(f"!! interrupted: {e}", flush=True)
            return 75  # EX_TEMPFAIL, as the study workers: the SLURM task requeues, the rerun resumes
        except Exception as e:  # one broken run must not stop the sweep
            failed += 1
            print(f"!! failed: {type(e).__name__}: {e}", flush=True)
            continue
        fwd = payload["timing"]["forward_s"]
        first = payload["metrics_spec"][0]["name"] if payload.get("metrics_spec") else None
        at_h = payload["at_horizon"].get(first) if first else None
        print(
            f"   {payload['n_runs']} runs, {payload['n_windows']} windows, {fwd.get('n', 0)} steps,"
            f" forward {fwd.get('mean', float('nan')):.3f} s/step"
            f" (median {fwd.get('median', float('nan')):.3f}), {first} at horizon {at_h if at_h is None else f'{at_h:.5g}'},"
            f" {time.perf_counter() - started:.0f} s",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
