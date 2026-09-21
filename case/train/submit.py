"""Submit the pending Train1-10 runs as throttled SLURM job arrays, one per time limit.

Enumerates configs/<campaign>/<model>_<kind>.yaml (the main gen+all phase; --extra adds
the parked configs/<campaign>/extra/ uCT-oracle phase) and submits every task that is
neither finished nor already held by SLURM. The time limit is a property of the model
(`TIME_LIMITS`): the voxel models finish a run inside a day and ask for one, which lets
the scheduler backfill them into gaps that a five-day request never fits; AB-UPT asks for
two days, Transolver keeps the five-day ceiling. A job array has one limit, so the pending
configs are grouped by limit into one array each, with `--throttle` split between them by
group size so the arrays together never hold more GPUs than one did. Two guards make
resubmitting safe at any time — it is the resume path after failures, preemption and
time limits:

* the queue is authoritative: every running *or pending* `poreml_train` array task is
  mapped back to its config through the manifest its submission wrote
  (`_manifests/job_<jobid>.tsv`), so a config in the queue is never submitted twice;
* a run that stopped short (`ckpts/resume.pt` present, status not finished) is reported
  as `resume` — the worker's `poreml train --resume auto` continues it in place.

See case/train/README.md for the plan.

Usage:
  uv run python case/train/submit.py [--dry-run] [--only SUBSTR] [--throttle 20] [--extra]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAMPAIGNS = ("drainage", "gdl", "trapping", "underfill")
MANIFESTS = REPO / "case/train/_manifests"
LOGS = REPO / "case/train/_logs"
JOB_NAME = "poreml_train"


def queued_configs(squeue_lines: Iterable[str], manifests: Path) -> dict[str, str]:
    """{config path: SLURM state} for every array task of ours the queue still holds.

    `squeue_lines` are `<arrayjob>_<task> <STATE>` rows (squeue -r -o "%i %T"); the task
    index is resolved through `job_<arrayjob>.tsv`, the symlink `submit` leaves beside the
    manifest. A job without one is reported and skipped — it is not one of ours to reason about.
    """
    found: dict[str, str] = {}
    for line in squeue_lines:
        parts = line.split()
        if len(parts) < 2 or "_" not in parts[0]:
            continue
        job_id, state = parts[0], parts[1]
        array_id, task_id = job_id.split("_", 1)
        link = manifests / f"job_{array_id}.tsv"
        if not link.exists():
            print(f"  warning: queued {job_id} ({state}) has no manifest under {manifests}; ignoring it")
            continue
        for row in link.read_text().splitlines():
            idx, cfg = row.split("\t", 1)
            if idx == task_id:
                found[cfg] = state
    return found


# Wall-clock limit per model. A run that outlives its limit is not lost: the worker saves
# 15 min before the limit and requeues itself (train_task.slurm), so a long run becomes
# segments — the shorter the limit, the more often the scheduler can fit the task into a gap.
TIME_LIMITS = {"unet3d": "1-00:00:00", "fno3d": "1-00:00:00", "p3d": "1-00:00:00", "abupt": "2-00:00:00"}
TIME_LIMIT_CEILING = "5-00:00:00"  # transolver, and anything unlisted


def time_limit(model: str) -> str:
    return TIME_LIMITS.get(model, TIME_LIMIT_CEILING)


def split_throttle(total: int, sizes: list[int]) -> list[int]:
    """Share `total` concurrent tasks between groups in proportion to their size, at least
    one each (a throttle above an array's size is harmless; Slurm caps it there). `total` 0:
    no cap on any array — every task starts as soon as a GPU is free, as on a reservation."""
    if not sizes:
        return []
    if total == 0:
        return [0] * len(sizes)
    n = sum(sizes)
    shares = [max(1, total * size // n) for size in sizes]
    while sum(shares) > total and max(shares) > 1:  # the minimum of one overflowed the total
        shares[shares.index(max(shares))] -= 1
    for i in sorted(range(len(sizes)), key=lambda i: -(total * sizes[i] % n)):  # largest remainders first
        if sum(shares) >= total:
            break
        shares[i] += 1
    return shares


@dataclass(frozen=True)
class Group:
    time_limit: str
    configs: list[Path]
    throttle: int


def plan(pending: list[tuple[Path, str]], throttle: int) -> list[Group]:
    """One array per time limit, shortest first, configs in their enumeration order."""
    by_limit: dict[str, list[Path]] = {}
    for path, model in pending:
        by_limit.setdefault(time_limit(model), []).append(path)
    limits = sorted(by_limit, key=lambda t: (len(t), t))  # "1-00:00:00" < "2-..." < "5-..."
    shares = split_throttle(throttle, [len(by_limit[t]) for t in limits])
    return [Group(t, by_limit[t], share) for t, share in zip(limits, shares, strict=True)]


def squeue_lines() -> list[str]:
    cmd = ["squeue", "-r", "-h", "-u", os.environ.get("USER", ""), "-n", JOB_NAME, "-o", "%i %T"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"squeue failed: {result.stderr.strip()} — not submitting blind")
    return result.stdout.splitlines()


def _status(meta: Path) -> str:
    return str(json.loads(meta.read_text()).get("status", "unknown"))


def classify(out_dir: Path) -> tuple[str, Path | None]:
    """('finished', None) | ('resume', run_dir) | ('fresh', None) for a config's out_dir."""
    metas = sorted(out_dir.glob("*/run_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True) if out_dir.is_dir() else []
    if any(_status(m) == "finished" for m in metas):
        return "finished", None
    for meta in metas:
        if (meta.parent / "ckpts" / "resume.pt").is_file():
            return "resume", meta.parent
    return "fresh", None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report; submit nothing")
    ap.add_argument("--only", default="", help="substring filter on config paths (e.g. 'trapping' or 'fno')")
    ap.add_argument(
        "--throttle",
        type=int,
        default=20,
        help="max concurrent array tasks = GPUs held (default 20; 0 = no cap, for a reservation)",
    )
    ap.add_argument("--extra", action="store_true", help="include the parked extra phase (configs/<campaign>/extra/)")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from poreml.config import Config

    queued = queued_configs(squeue_lines(), MANIFESTS)
    patterns = ["*.yaml"] + (["extra/*.yaml"] if args.extra else [])
    configs = sorted(p for c in CAMPAIGNS for pat in patterns for p in (REPO / "configs" / c).glob(pat))
    pending: list[tuple[Path, str]] = []
    for path in configs:
        rel = str(path.relative_to(REPO))
        if args.only and args.only not in rel:
            print(f"  filtered  {rel}")
            continue
        if rel in queued:
            print(f"  queued    {rel} ({queued[rel]})")
            continue
        cfg = Config.from_yaml(path)
        kind, run_dir = classify(REPO / cfg.out_dir)
        if kind == "finished":
            print(f"  finished  {rel}")
            continue
        limit = time_limit(cfg.model.name)
        print(f"  {kind:9s} {rel}  [{limit}]" + (f" (continues {run_dir.relative_to(REPO)})" if run_dir else ""))
        pending.append((path, cfg.model.name))
    print(f"\n{len(pending)} pending of {len(configs)} configs ({len(queued)} in the queue)")
    groups = plan(pending, args.throttle)
    for g in groups:
        print(f"  array: {len(g.configs)} tasks, time limit {g.time_limit}, throttle {g.throttle or 'none'}")
    if not pending or args.dry_run:
        return 0

    MANIFESTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    for g in groups:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        manifest = MANIFESTS / f"train_{stamp}_{g.time_limit.split('-')[0]}d.tsv"
        manifest.write_text("".join(f"{i}\t{p.relative_to(REPO)}\n" for i, p in enumerate(g.configs)))
        cmd = [
            "sbatch",
            f"--job-name={JOB_NAME}",
            f"--array=0-{len(g.configs) - 1}" + (f"%{g.throttle}" if g.throttle else ""),
            f"--time={g.time_limit}",
            f"--output={LOGS}/%A_%a.log",
            f"--export=ALL,POREML_REPO={REPO},MANIFEST={manifest}",
            str(REPO / "case/train/train_task.slurm"),
        ]
        print("\n" + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout.strip() or result.stderr.strip())
        if result.returncode != 0:
            return result.returncode
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match is None:
            print("warning: could not read the job id from sbatch's output; queued_configs will not see this array")
            return 1
        # The symlink is what lets a later `submit` map `<jobid>_<task>` in the queue back to a config.
        (MANIFESTS / f"job_{match.group(1)}.tsv").symlink_to(manifest.name)
        print(f"manifest: {manifest.relative_to(REPO)} (job {match.group(1)})")
        time.sleep(1)  # distinct manifest stamps
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
