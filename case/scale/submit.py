"""Submit the pending scale cells as one throttled SLURM job array (1 GPU + 8 CPUs each).

Enumerates configs/scale/<campaign>/*.yaml and submits every cell in state `pending`
(poreml.study.classify: a finished checkpoint exists but its out_dir has no
inference_test.json yet) — skipping `waiting` (no finished checkpoint), `metric`
(inference_test.json exists, metrics_test.json for that checkpoint does not — see
`--metric` below) and `finished` (both exist for the same checkpoint) cells, and any cell
already held by SLURM (running or pending array tasks are mapped back to their cells
through _manifests/job_<jobid>.tsv). Resubmitting is the recovery procedure after failures
and preemption. See README.md.

Usage:
  uv run python case/scale/submit.py [--dry-run] [--only SUBSTR] [--throttle 8] [--reservation NAME]
  uv run python case/scale/submit.py --metric [--throttle 8]   # the CPU-only array for cells whose frames wait
  uv run python case/scale/submit.py --render [--reservation NAME]   # GPU array: only the GIFs of cells that lack them

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
STUDY = "scale"
MANIFESTS = REPO / "case" / STUDY / "_manifests"
LOGS = REPO / "case" / STUDY / "_logs"
JOB_NAME = f"poreml_{STUDY}"
# Inference arrays go out fast-first so the first numbers land early: the voxel models (minutes per
# 256-class window) before the point models (AB-UPT ~33 min per window), shorter campaigns first.
# Host memory is sized per group — the 200G in inference_task.slurm is what the point models need
# (~3.5 M pore points per sample); the voxel models get 100G — because RAM, not GPUs, bounds how
# many cells share a node.
HEAVY = ("transolver", "abupt")
WINDOWS = {"trapping": 48, "gdl": 119, "drainage": 157}
GROUPS = (("light", lambda m: m not in HEAVY, "100G"), ("heavy", lambda m: m in HEAVY, "200G"))


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
    args = ap.parse_args()

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
        if rel in queued:
            print(f"  queued    {rel} ({queued[rel]})")
            continue
        cell = Cell.load(path)
        state = classify(cell)
        if args.render:  # inference done (frames stored), fewer GIFs than the scheme asks for
            have, want = len(render_gifs(cell.config.out_dir)), cell.scheme.render
            state = f"{state}, {have}/{want} GIFs"
            print(f"  {state} {rel}")
            if classify(cell) in ("finished", "metric") and have < want:
                pending.append(path)
            continue
        print(f"  {state:9s} {rel}")
        if state == wanted:
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
        manifest = MANIFESTS / f"{STUDY}_{name}_{time.strftime('%Y%m%d-%H%M%S')}.tsv"
        manifest.write_text("".join(f"{i}\t{p.relative_to(REPO)}\n" for i, p in enumerate(group)))
        cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--array=0-{len(group) - 1}%{args.throttle}",
            f"--output={LOGS}/%A_%a.log",
            f"--export=ALL,POREML_REPO={REPO},MANIFEST={manifest}" + (",STAGE=render" if args.render else ""),
            *([f"--mem={mem}"] if mem else []),
            *(["--time=12:00:00"] if args.render else []),
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
        print(f"manifest: {manifest.relative_to(REPO)} ({name}, {len(group)} cells, job {match.group(1)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
