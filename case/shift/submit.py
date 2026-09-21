"""Submit the pending shift cells as one throttled SLURM job array (1 GPU + 8 CPUs each).

Enumerates configs/shift/<campaign>/*.yaml and submits every cell in state `pending`
(poreml.study.classify: a finished checkpoint exists but its out_dir has no
inference_test.json yet) — skipping `waiting` (no finished checkpoint), `metric`
(inference_test.json exists, metrics_test.json for that checkpoint does not — see
`--metric` below) and `finished` (both exist for the same checkpoint) cells, and any cell
already held by SLURM (running or pending array tasks are mapped back to their cells
through _manifests/job_<jobid>.tsv). Resubmitting is the recovery procedure after failures
and preemption. See README.md.

Usage:
  uv run python case/shift/submit.py [--dry-run] [--only SUBSTR] [--throttle 8] [--reservation NAME]
  uv run python case/shift/submit.py --metric [--throttle 8]   # the CPU-only array for cells whose frames wait

With `--metric` the cells in state `metric` (inference done, frames stored, no metrics_test.json
yet) go to metric_task.slurm — no GPU, 16 CPUs — under the job name `poreml_shift_metric`; a
cell is finished once metrics_test.json exists.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = "shift"
MANIFESTS = REPO / "case" / STUDY / "_manifests"
LOGS = REPO / "case" / STUDY / "_logs"
JOB_NAME = f"poreml_{STUDY}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report; submit nothing")
    ap.add_argument("--only", default="", help="substring filter on cell paths (e.g. 'gdl' or 'abupt')")
    ap.add_argument("--throttle", type=int, default=8, help="max concurrent array tasks = GPUs held (default 8)")
    ap.add_argument("--metric", action="store_true", help="submit the CPU metric array for cells whose frames wait")
    ap.add_argument("--exclude", default="", help="sbatch --exclude list")
    ap.add_argument("--reservation", default="", help="sbatch --reservation")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    import os

    os.chdir(REPO)
    from poreml.study import Cell, cell_paths, classify, queued_configs, squeue_lines

    job_name = JOB_NAME + ("_metric" if args.metric else "")
    task_script = REPO / "case" / STUDY / ("metric_task.slurm" if args.metric else "inference_task.slurm")
    wanted = "metric" if args.metric else "pending"
    queued = queued_configs(squeue_lines(JOB_NAME) + squeue_lines(JOB_NAME + "_metric"), MANIFESTS)
    pending: list[Path] = []
    paths = cell_paths(STUDY, repo=REPO)
    for path in paths:
        rel = str(path.relative_to(REPO))
        if args.only and args.only not in rel:
            print(f"  filtered  {rel}")
            continue
        if rel in queued:
            print(f"  queued    {rel} ({queued[rel]})")
            continue
        state = classify(Cell.load(path))
        print(f"  {state:9s} {rel}")
        if state == wanted:
            pending.append(path)
    print(f"\n{len(pending)} {wanted} of {len(paths)} cells ({len(queued)} in the queue)")
    if not pending or args.dry_run:
        return 0

    MANIFESTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    manifest = MANIFESTS / f"{STUDY}{'_metric' if args.metric else ''}_{time.strftime('%Y%m%d-%H%M%S')}.tsv"
    manifest.write_text("".join(f"{i}\t{p.relative_to(REPO)}\n" for i, p in enumerate(pending)))
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--array=0-{len(pending) - 1}%{args.throttle}",
        f"--output={LOGS}/%A_%a.log",
        f"--export=ALL,POREML_REPO={REPO},MANIFEST={manifest}",
        *([f"--exclude={args.exclude}"] if args.exclude else []),
        *([f"--reservation={args.reservation}"] if args.reservation else []),
        str(task_script),
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
    (MANIFESTS / f"job_{match.group(1)}.tsv").symlink_to(manifest.name)  # lets a later submit map queued tasks back to cells
    print(f"manifest: {manifest.relative_to(REPO)} (job {match.group(1)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
