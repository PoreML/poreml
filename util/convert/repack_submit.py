"""Submit the repack of data/case (drop the derivable fields rho and umag) as SLURM arrays.

Enumerates every run directory under the mirror (those holding a <run_id>.conversion.json),
skips runs whose record already says `repacked` and runs the queue still holds (running *and*
pending array tasks are mapped back through _manifests/repack_job_<jobid>.tsv), orders the rest
largest file first so the long tail is short, writes a manifest (array index -> run dir) and
submits util/convert/repack_task.slurm over it. `--only` (repeatable) keeps the runs whose path
contains any of the substrings, so the arrays can follow the readers: a campaign's files may be
repacked only when nothing reads them — pass `--dependency afterany:<jobid>_<task>:...` with the
jobs that do. Resubmitting is the recovery procedure (idempotent tasks).

Usage:
  uv run python util/convert/repack_submit.py --dry-run
  uv run python util/convert/repack_submit.py --only drainage/ --only /256x256x128/ --throttle 48
  uv run python util/convert/repack_submit.py --only GDL/runs --dependency afterany:37724_10:37724_11
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "util/convert/_manifests"
LOGS = REPO / "util/convert/_logs"
JOB_NAME = "poreml_repack"


def queued_runs() -> set[str]:
    """Run dirs held by running or pending repack tasks, via the manifests of live jobs."""
    held: set[str] = set()
    try:
        out = subprocess.run(
            ["squeue", "-h", "-n", JOB_NAME, "-o", "%i %T", "-r"], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return held
    for line in out.split("\n"):
        if not line.strip():
            continue
        task_id, state = line.split()
        if state not in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "REQUEUED"):
            continue
        m = re.match(r"(\d+)_(\d+)$", task_id)
        if not m:
            continue
        manifest = MANIFESTS / f"repack_job_{m.group(1)}.tsv"
        if manifest.is_file():
            for row in manifest.read_text().splitlines():
                idx, run = row.split("\t")
                if idx == m.group(2):
                    held.add(run)
    return held


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "data/case")
    ap.add_argument("--only", action="append", default=[], help="substring on the run path; repeatable, any match keeps")
    ap.add_argument("--exclude", action="append", default=[], help="substring on the run path; repeatable, any match drops")
    ap.add_argument("--only-file", type=Path, help="a file of run ids (one per line) to keep, e.g. a split's val list")
    ap.add_argument("--exclude-file", type=Path, help="a file of run ids (one per line) to drop")
    ap.add_argument("--throttle", type=int, default=48, help="concurrent array tasks (NFS streams)")
    ap.add_argument("--dependency", default="", help="sbatch --dependency, e.g. afterany:37724_10:37724_11")
    ap.add_argument("--reservation", default="", help="sbatch --reservation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    import json

    from poreml.convert import CONVERSION_SUFFIX

    only_ids = set(args.only_file.read_text().split()) if args.only_file else None
    exclude_ids = set(args.exclude_file.read_text().split()) if args.exclude_file else set()
    runs = []
    for record in sorted(args.root.rglob(f"*{CONVERSION_SUFFIX}")):
        run_dir = record.parent
        rel = str(run_dir.relative_to(args.root))
        if args.only and not any(s in rel + "/" for s in args.only):
            continue
        if any(s in rel + "/" for s in args.exclude) or run_dir.name in exclude_ids:
            continue
        if only_ids is not None and run_dir.name not in only_ids:
            continue
        h5s = list(run_dir.glob("*.h5"))
        if len(h5s) != 1:
            print(f"!! {rel}: {len(h5s)} h5 files, skipped")
            continue
        state = "repacked" if json.loads(record.read_text()).get("repacked") else "pending"
        runs.append((state, h5s[0].stat().st_size, str(run_dir.relative_to(REPO))))
    held = queued_runs()
    pending = sorted((s, r) for st, s, r in runs if st == "pending" and r not in held)
    pending.sort(key=lambda x: -x[0])  # largest first
    n_done = sum(1 for st, _, _ in runs if st == "repacked")
    n_held = sum(1 for st, _, r in runs if st == "pending" and r in held)
    gb = sum(s for s, _ in pending) / 1e9
    print(
        f"{len(runs)} runs selected: {n_done} repacked, {n_held} in the queue, {len(pending)} to submit ({gb:.0f} GB to read)"
    )
    for s, r in pending[:5]:
        print(f"   {s / 1e9:6.1f} GB  {r}")
    if len(pending) > 5:
        print(f"   ... {len(pending) - 5} more")
    if not pending or args.dry_run:
        return 0
    MANIFESTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    manifest = MANIFESTS / f"repack_{stamp}.tsv"
    manifest.write_text("".join(f"{i}\t{r}\n" for i, (_, r) in enumerate(pending)))
    cmd = [
        "sbatch",
        f"--job-name={JOB_NAME}",
        f"--array=0-{len(pending) - 1}%{args.throttle}",
        f"--output={LOGS}/repack_%A_%a.log",
        f"--export=ALL,POREML_REPO={REPO},MANIFEST={manifest}",
    ]
    if args.dependency:
        cmd.append(f"--dependency={args.dependency}")
    if args.reservation:
        cmd.append(f"--reservation={args.reservation}")
    cmd.append(str(REPO / "util/convert/repack_task.slurm"))
    print(" ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    print(out)
    m = re.search(r"(\d+)\s*$", out)
    if m:
        (MANIFESTS / f"repack_job_{m.group(1)}.tsv").write_text(manifest.read_text())
    else:
        print("warning: could not read the job id from sbatch's output; the queued guard will not see this array")
    return 0


if __name__ == "__main__":
    sys.exit(main())
