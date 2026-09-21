"""Submit the pending timed-rollout stages as SLURM job arrays on the main partition.

    uv run python util/inference/submit.py [TARGET ...] [--dry-run] [--only SUBSTR] [--throttle 12]
                                      [--qos auto|compact|normal] [--which auto|best|best_rollout|last]

Targets default to `case/train` and `case/train_push`; each resolves to its finished runs as
`util/inference/inference.py` does, and every run that is `pending`, `resume` or `stale` (not
`finished`, not `waiting`, not already in the queue — running *and pending* array tasks are
mapped back to their run dirs through `_manifests/job_<jobid>.tsv`, so resubmitting never
double-runs) becomes one array task of `inference_task.slurm` (1 GPU + 8 CPUs, no reservation:
the tasks land beside the push trainings, never on their node).

Each task's time limit is estimated from its window count (the run's own split and rollout
horizon) and a per-model seconds-per-step table measured on the H200 (`SECONDS_PER_STEP`,
underfill's larger frames scaled by `UNDERFILL_FACTOR`), doubled for safety and rounded up to a
bucket; one array per (bucket, QOS). QOS `compact` (priority 1000, on no QOS's preempt list so
it is never preempted, preempts nobody itself, MaxWall 1 day) lets the whole account hold only
**2 GPUs at once** (`MaxTRESPA gres/gpu=2`, measured 2026-09-12), so `--qos auto` reserves it
for the long cells (estimate >= `COMPACT_MIN_HOURS`), which get a continuous never-preempted
2-GPU stream with a limit of twice their estimate, and puts the short cells on `normal`
(fairshare-ranked at the bottom of the queue, preemptable) with the smallest bucket above their
estimate: short limits fit backfill holes, and a task that is bumped or times out finishes the
window in hand, exits 75, requeues and resumes. The `%throttle` is split between the arrays by
their GPU-hours, so at most that many GPUs are held at once. Resubmitting is the whole recovery
procedure after a failure or a timeout.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "util" / "inference"
MANIFESTS = HERE / "_manifests"
LOGS = HERE / "_logs"
JOB_NAME = "poreml_infer"
TASK_SCRIPT = HERE / "inference_task.slurm"

# Wall seconds per rollout step on an H200 at the 128-class sizes: the stage's own measurement
# for unet3d (0.52, forward 0.08 of it — target loading, scoring and the frame store take the
# rest), the study inference's per-window gaps plus that overhead for the others (2026-09-11).
SECONDS_PER_STEP = {"unet3d": 0.55, "fno3d": 0.45, "p3d": 0.45, "transolver": 1.0, "abupt": 1.7}
DEFAULT_SECONDS_PER_STEP = 1.7
UNDERFILL_FACTOR = {"voxel": 2.5, "points": 4.0}  # 22x482x476 frames: 2.4x the voxels, more pore points
SAFETY = {"compact": 2.0, "normal": 1.0}  # compact never requeues, so its limit must hold the whole cell
BUCKETS = ["01:00:00", "02:00:00", "04:00:00", "08:00:00", "16:00:00", "23:00:00"]  # all under compact's 1-day MaxWall
NORMAL_LIMIT = "2-00:00:00"  # a cell beyond the last bucket: normal QOS, requeue + resume
COMPACT_MAX_HOURS = 24.0
COMPACT_MIN_HOURS = 2.0  # compact's 2-GPU account cap goes to the cells that benefit most from never being preempted

sys.path.insert(0, str(REPO / "src"))
_spec = importlib.util.spec_from_file_location("case_inference", HERE / "inference.py")
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


def hours(limit: str) -> float:
    days, _, clock = limit.rpartition("-")
    h, m, s = (int(x) for x in clock.split(":"))
    return (int(days) if days else 0) * 24 + h + m / 60 + s / 3600


def estimate_hours(model: str, representation: str, campaign: str, n_steps: int) -> float:
    """Expected wall hours of a cell: steps x seconds per step (x the underfill factor)."""
    per_step = SECONDS_PER_STEP.get(model, DEFAULT_SECONDS_PER_STEP)
    if campaign.lower() == "underfill":
        per_step *= UNDERFILL_FACTOR.get(representation, UNDERFILL_FACTOR["points"])
    return n_steps * per_step / 3600


def qos_for(estimate: float, policy: str = "auto") -> str:
    """`auto`: compact for the long cells (its 2-GPU account cap is best spent there), normal otherwise."""
    if policy in ("compact", "normal"):
        return policy
    return "compact" if COMPACT_MIN_HOURS <= estimate <= COMPACT_MAX_HOURS / SAFETY["compact"] else "normal"


def time_limit(estimate: float, qos: str = "normal") -> str:
    """The smallest bucket holding `SAFETY[qos]` x the estimate, else the normal-QOS ceiling."""
    for bucket in BUCKETS:
        if SAFETY[qos] * estimate <= hours(bucket):
            return bucket
    return NORMAL_LIMIT


_WINDOWS: dict[tuple, int] = {}  # (root, campaign, split file, section, history, horizon) -> windows: cells share splits


def split_windows(cfg, split: str) -> int:
    """How many `horizon`-step windows at stride `horizon` the split's runs hold (opens each run's HDF5 once per split)."""
    from poreml.data import Trajectory
    from poreml.evaluate import load_runs
    from poreml.rollout import window_starts

    horizon, history = cfg.rollout.horizon, cfg.task.history
    key = (str(cfg.data.root), cfg.data.campaign, str(cfg.data.split), split, history, horizon)
    if key not in _WINDOWS:
        n = 0
        for ref in load_runs(cfg)[split]:
            traj = Trajectory(ref.h5_path)
            try:
                n += len(window_starts(len(traj.steps), history, horizon, horizon))
            finally:
                traj.close()
        _WINDOWS[key] = n
    return _WINDOWS[key]


def windows_of(run_dir: Path, split: str) -> tuple[int, int, str, str, str]:
    """(n_windows, n_steps, model, representation, campaign) of a run's stage from its own config."""
    from poreml.config import Config
    from poreml.models import representation_of

    cfg = Config.from_yaml(run_dir / "config.yaml")
    n_windows = split_windows(cfg, split)
    return n_windows, n_windows * cfg.rollout.horizon, cfg.model.name, representation_of(cfg.model), cfg.data.campaign


def split_throttle(total: int, sizes: list[float]) -> list[int]:
    """Share `total` concurrent tasks between arrays in proportion to their GPU-hours, at least one each.

    Weighting by hours, not task count, keeps the array of the few longest cells from becoming the
    sweep's long pole (four 5-6 h cells at one task at a time is a day; at three it is an evening).
    """
    if total <= 0:
        return [0] * len(sizes)  # 0 = no cap
    n = sum(sizes)
    shares = [max(1, round(total * s / n)) for s in sizes]
    while sum(shares) > max(total, len(sizes)):
        i = max(range(len(shares)), key=lambda k: shares[k])
        shares[i] -= 1
    return shares


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", type=Path, default=[REPO / "case" / "train", REPO / "case" / "train_push"])
    ap.add_argument("--dry-run", action="store_true", help="report the plan; submit nothing")
    ap.add_argument("--only", default="", help="substring filter on run dirs (e.g. 'gdl', 'abupt', 'train_push')")
    ap.add_argument("--throttle", type=int, default=12, help="max concurrent tasks over all arrays = GPUs held (0 = no cap)")
    ap.add_argument("--qos", choices=("auto", "compact", "normal"), default="auto", help="auto: compact when the limit fits")
    ap.add_argument("--which", choices=driver.WHICH, default="auto", help="checkpoint (see util/inference/inference.py)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--exclude", default="", help="sbatch --exclude node list")
    args = ap.parse_args(argv)

    os.chdir(REPO)
    from poreml.study import queued_configs, squeue_lines

    queued = queued_configs(squeue_lines(JOB_NAME), MANIFESTS)
    plan: list[tuple[Path, str, float, str, str]] = []  # run dir, state, estimate h, limit, qos
    counts: dict[str, int] = {}
    for run in driver.find_runs(args.targets):
        rel = str(run.relative_to(REPO)) if run.is_relative_to(REPO) else str(run)
        if args.only and args.only not in rel:
            continue
        ckpt = driver.choose_checkpoint(run, args.which)
        state = queued.get(rel, "").lower() or driver.state(run, ckpt)
        counts[state] = counts.get(state, 0) + 1
        if state in ("waiting", "finished") or rel in queued:
            print(f"  {state:9s} {'':>6s} {'':>9s} {'':>7s} {rel}")
            continue
        n_windows, n_steps, model, representation, campaign = windows_of(run, args.split)
        est = estimate_hours(model, representation, campaign, n_steps)
        qos = qos_for(est, args.qos)
        limit = time_limit(est, qos)
        plan.append((run, state, est, limit, qos))
        print(f"  {state:9s} {est:5.1f} h {limit:>9s} {qos:>7s} {rel}  ({n_windows} windows, {ckpt.name})")

    summary = ", ".join(f"{n} {st}" for st, n in sorted(counts.items()))
    total = sum(p[2] for p in plan)
    print(f"\n{sum(counts.values())} runs: {summary}; {len(plan)} to submit, {total:.0f} GPU-hours estimated")
    if not plan or args.dry_run:
        return 0

    MANIFESTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    arrays: dict[tuple[str, str], list[Path]] = {}
    for run, _, _, limit, qos in plan:
        arrays.setdefault((limit, qos), []).append(run)
    keys = sorted(arrays, key=lambda k: hours(k[0]))
    hours_by_array = {k: sum(est for run, _, est, limit, qos in plan if (limit, qos) == k) for k in keys}
    throttles = split_throttle(args.throttle, [hours_by_array[k] for k in keys])
    rc = 0
    for (limit, qos), throttle in zip(keys, throttles, strict=True):
        runs = arrays[(limit, qos)]
        manifest = MANIFESTS / f"infer_{stamp}_{limit.replace(':', '')}_{qos}.tsv"
        manifest.write_text("".join(f"{i}\t{r.relative_to(REPO)}\n" for i, r in enumerate(runs)))
        cmd = [
            "sbatch",
            f"--job-name={JOB_NAME}",
            f"--array=0-{len(runs) - 1}" + (f"%{throttle}" if throttle else ""),
            f"--time={limit}",
            f"--qos={qos}",
            f"--output={LOGS}/%A_%a.log",
            f"--export=ALL,POREML_REPO={REPO},MANIFEST={manifest}",
            *([f"--exclude={args.exclude}"] if args.exclude else []),
            str(TASK_SCRIPT),
        ]
        print("\n" + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout.strip() or result.stderr.strip())
        if result.returncode != 0:
            rc = result.returncode
            continue
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match is None:
            print("warning: could not read the job id from sbatch's output; the queued guard will not see this array")
            rc = 1
            continue
        (MANIFESTS / f"job_{match.group(1)}.tsv").symlink_to(manifest.name)
        print(
            f"manifest: {manifest.relative_to(REPO)} (job {match.group(1)}, {len(runs)} tasks, "
            f"{hours_by_array[(limit, qos)]:.0f} GPU-h, %{throttle or 'none'})"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
