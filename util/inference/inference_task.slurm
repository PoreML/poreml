#!/bin/bash
#SBATCH --qos=normal            # submit.py passes --qos=compact when the cell's estimate fits its 1-day MaxWall
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00       # the ceiling; submit.py passes the cell's own limit on the command line
#SBATCH --requeue
#SBATCH --signal=B:TERM@900

# One timed-rollout stage (util/inference/inference.py) for one finished training run: every 64-step
# window of its validation split, scored at every step, 12 keyframes per window stored. Resumable
# per window: SIGTERM (preemption, or --signal 15 min before the limit) is forwarded to python,
# which finishes the window in hand and exits 75, on which this script requeues itself; the rerun
# keeps every complete window. Uses the repo venv directly (not `uv run`): concurrent
# tasks must not race an environment sync, and compute nodes should not need the network.
set -euo pipefail
cd "${POREML_REPO:-${SLURM_SUBMIT_DIR:-.}}"   # the repo root: the submitter exports POREML_REPO; by hand, sbatch from the root

MANIFEST="${MANIFEST:?submit via util/inference/submit.py, which sets MANIFEST}"
RUN=$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i {print $2}' "$MANIFEST")
[ -n "$RUN" ] || { echo "no manifest line for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}"; exit 1; }

export PYTHONFAULTHANDLER=1   # a silent death is a segfault first — always get the traceback
echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  task ${SLURM_ARRAY_TASK_ID}  restarts ${SLURM_RESTART_COUNT:-0}  qos ${SLURM_JOB_QOS:-?}  run ${RUN}  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

pid=""
trap 'echo "SIGTERM at $(date): forwarding to python"; [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null' TERM
.venv/bin/python util/inference/inference.py "$RUN" &
pid=$!
set +e
wait "$pid"; rc=$?
if [ "$rc" -gt 128 ] && kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; fi   # the trap interrupted the first wait
set -e

if [ "$rc" -eq 75 ]; then
    echo "=== ${RUN} stopped by SIGTERM with its windows stored; requeueing  $(date) ==="
    scontrol requeue "${SLURM_JOB_ID}" || true
    exit 0
fi
echo "=== done ${RUN}  exit ${rc}  $(date) ==="
exit "$rc"
