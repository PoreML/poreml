"""Submit the pending push-forward fine-tunes as throttled SLURM job arrays.

A thin wrapper over case/train/submit.py — the same guards (a config the queue holds is
never submitted twice, resolved through the manifests; a run that stopped short resumes
in place), the same worker (case/train/train_task.slurm: `poreml train --resume auto`,
requeue on preemption and before the time limit) and the same per-model time limits —
pointed at configs/push/<campaign>/, under its own job name `poreml_push` with manifests
and logs in this folder. Run make_configs.py first; see README.md.

Usage:
  uv run python case/train_push/submit.py [--dry-run] [--only SUBSTR] [--throttle 20] [--reservation NAME]

`--reservation` puts the arrays on a Slurm reservation (sbatch reads SBATCH_RESERVATION);
there `--throttle 0` (no lane cap) keeps every reserved GPU busy — a cap only ever delays
your own tasks, and the node is yours.
"""

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

_spec = importlib.util.spec_from_file_location("train_submit", REPO / "case/train/submit.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

base.JOB_NAME = "poreml_push"
base.CAMPAIGNS = tuple(f"push/{c}" for c in base.CAMPAIGNS)
base.MANIFESTS = HERE / "_manifests"
base.LOGS = HERE / "_logs"

if __name__ == "__main__":
    if "--reservation" in sys.argv:  # consumed here; the wrapped main() keeps its own arguments
        i = sys.argv.index("--reservation")
        os.environ["SBATCH_RESERVATION"] = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    raise SystemExit(base.main())
