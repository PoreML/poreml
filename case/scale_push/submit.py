"""Submit the pending scale cells as one throttled SLURM job array (1 GPU + 8 CPUs each).

Enumerates configs/scale_push/<campaign>/*.yaml and submits every cell in state `pending`
(poreml.study.classify: a finished checkpoint exists but its out_dir has no
inference_test.json yet) — skipping `waiting` (no finished checkpoint), `metric`
(inference_test.json exists, metrics_test.json for that checkpoint does not — see
`--metric` below) and `finished` (both exist for the same checkpoint) cells, and any cell
already held by SLURM (running or pending array tasks are mapped back to their cells
through _manifests/job_<jobid>.tsv). Resubmitting is the recovery procedure after failures
and preemption. See README.md.

Usage:
  uv run python case/scale_push/submit.py [--dry-run] [--only SUBSTR] [--throttle 8] [--reservation NAME]
  uv run python case/scale_push/submit.py --metric [--throttle 8]   # the CPU-only array for cells whose frames wait
  uv run python case/scale_push/submit.py --render [--reservation NAME]   # GPU array: only the GIFs of cells that lack them
  uv run python case/scale_push/submit.py --only abupt --workers 4 [--qos compact --time 06:00:00]   # 4 GPUs per cell

`--workers N` (default 1) puts N tasks of the same cell in the array, all `--shared`: they claim
windows through the frame store as they go, so N GPUs finish the cell N times sooner, a worker
that starts late takes what is left, and the last one writes the sentinel (the cell's GIFs are
then the `--render` array's). `--add` submits workers for a cell the queue already holds — only
safe when its queued tasks are shared workers too (an unshared task ignores claims). `--qos`
and `--time` pass through to sbatch, so extra workers can go off the reservation as short
backfill tasks; `--throttle 0` removes the concurrency cap.

With `--metric` the cells in state `metric` (inference done, frames stored, no metrics_test.json
yet) go to metric_task.slurm — no GPU, 16 CPUs — under the job name `poreml_scale_metric`; a
cell is finished once metrics_test.json exists.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = "scale_push"
MANIFESTS = REPO / "case" / STUDY / "_manifests"
LOGS = REPO / "case" / STUDY / "_logs"
JOB_NAME = f"poreml_{STUDY}"
# Inference arrays go out fast-first so the first numbers land early: the voxel models (~1 min per
# 256-class window) before the point models (AB-UPT ~3.7 min per window in bf16, Transolver ~1.6,
# measured 2026-09-12 on trapping), shorter campaigns first. Host memory is sized per group and
# small: RAM, not GPUs, bounds how many cells share a node, and the measured RSS is a few GB — unet
# 10.6 GB, abupt 5.6 GB, transolver 2.4 GB — plus the render stage's 64-frame stack (~21 GB at 256)
# and its pyvista workers (~2.3 GB each). The 200G the point cells first asked for left two GPUs of
# the reservation idle for want of RAM.
HEAVY = ("transolver", "abupt")
WINDOWS = {"trapping": 48, "gdl": 119, "drainage": 157}
GROUPS = (("light", lambda m: m not in HEAVY, "100G"), ("heavy", lambda m: m in HEAVY, "64G"))


def _model(path: Path) -> str:
    return path.stem.split("_")[0]


def fast_first(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: (_model(p) in HEAVY, WINDOWS.get(p.parent.name, 999), _model(p)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report; submit nothing")
    ap.add_argument("--only", default="", help="substring filter on cell paths (e.g. 'gdl' or 'abupt')")
    ap.add_argument("--throttle", type=int, default=8, help="max concurrent array tasks = GPUs held (default 8)")
    ap.add_argument("--metric", action="store_true", help="submit the CPU metric array for cells whose frames wait")
    ap.add_argument("--reservation", default="", help="sbatch --reservation")
    ap.add_argument("--render", action="store_true", help="submit the GPU render array for cells whose GIFs are missing")
    ap.add_argument("--workers", type=int, default=1, help="inference: shared workers per cell (default 1: one task, GIFs too)")
    ap.add_argument("--add", action="store_true", help="inference: also submit workers for cells the queue already holds")
    ap.add_argument("--qos", default="", help="sbatch --qos (e.g. compact: a never-preempted 2-GPU stream off the reservation)")
    ap.add_argument("--time", default="", help="sbatch --time override (e.g. 06:00:00 for a backfill-sized worker)")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    shared = args.workers > 1 or args.add
    if shared and (args.metric or args.render):
        ap.error("--workers/--add apply to the inference array only")

    sys.path.insert(0, str(REPO / "src"))
    import os

    os.chdir(REPO)
    from poreml.study import Cell, cell_paths, classify, queued_configs, render_gifs, squeue_lines

    if args.metric and args.render:
        ap.error("--metric and --render are separate arrays")
    job_name = JOB_NAME + ("_metric" if args.metric else "_render" if args.render else "")
    task_script = REPO / "case" / STUDY / ("metric_task.slurm" if args.metric else "inference_task.slurm")
    wanted = "metric" if args.metric else "pending"
    queued = queued_configs(
        squeue_lines(JOB_NAME) + squeue_lines(JOB_NAME + "_metric") + squeue_lines(JOB_NAME + "_render"), MANIFESTS
    )
    pending: list[Path] = []
    paths = cell_paths(STUDY, repo=REPO)
    for path in paths:
        rel = str(path.relative_to(REPO))
        if args.only and args.only not in rel:
            print(f"  filtered  {rel}")
            continue
        if rel in queued and not args.add:
            print(f"  queued    {rel} ({queued[rel]})")
            continue
        cell = Cell.load(path)
        state = classify(cell)
        if rel in queued:
            state = f"{state} (queued: {queued[rel]}, adding workers)"
        if args.render:  # inference done (frames stored), fewer GIFs than the scheme asks for
            have, want = len(render_gifs(cell.config.out_dir)), cell.scheme.render
            state = f"{state}, {have}/{want} GIFs"
            print(f"  {state} {rel}")
            if classify(cell) in ("finished", "metric") and have < want:
                pending.append(path)
            continue
        print(f"  {state:9s} {rel}")
        if state.split(" ")[0] == wanted:
            pending.append(path)
    print(f"\n{len(pending)} {wanted} of {len(paths)} cells ({len(queued)} in the queue)")
    if not pending or args.dry_run:
        return 0

    MANIFESTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    if args.metric:
        groups = [("metric", pending, None)]
    else:
        pending = fast_first(pending)
        groups = [(name, [p for p in pending if pick(_model(p))], mem) for name, pick, mem in GROUPS]
    for name, group, mem in groups:
        if not group:
            continue
        tasks = [p for p in group for _ in range(args.workers if shared else 1)]  # a cell's workers are consecutive tasks
        manifest = MANIFESTS / f"{STUDY}_{name}{'_shared' if shared else ''}_{time.strftime('%Y%m%d-%H%M%S')}.tsv"
        manifest.write_text("".join(f"{i}\t{p.relative_to(REPO)}\n" for i, p in enumerate(tasks)))
        cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--array=0-{len(tasks) - 1}" + (f"%{args.throttle}" if args.throttle > 0 else ""),
            f"--output={LOGS}/%A_%a.log",
            f"--export=ALL,POREML_REPO={REPO},MANIFEST={manifest}"
            + (",STAGE=render" if args.render else "")
            + (",SHARED=1" if shared else ""),
            *([f"--mem={mem}"] if mem else []),
            *(["--time=12:00:00"] if args.render else [f"--time={args.time}"] if args.time else []),
            *([f"--qos={args.qos}"] if args.qos else []),
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
        (MANIFESTS / f"job_{match.group(1)}.tsv").symlink_to(manifest.name)  # maps queued tasks back to cells later
        print(f"manifest: {manifest.relative_to(REPO)} ({name}, {len(group)} cells, {len(tasks)} tasks, job {match.group(1)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
