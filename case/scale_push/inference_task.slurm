#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@3600

# One scale cell's inference stage: every 64-step window of every test run, keyframes stored
# under <out_dir>/frames/, nothing scored. Resumable per window: SIGTERM (preemption, or
# --signal before the limit) is forwarded to poreml, which finishes the window in hand and
# exits 75, on which this script requeues itself; the rerun skips stored windows. The stop
# flag is only polled between windows (not mid-window), so the lead time must exceed one
# window's wall time: at 256 a window is ~33 min for AB-UPT, the slowest model, so this uses
# a 60 min lead — shift stays at 900s (15 min) because its windows are on the 128 class, much
# faster per window.
set -euo pipefail
cd "${POREML_REPO:-${SLURM_SUBMIT_DIR:-.}}"   # the repo root: the submitter exports POREML_REPO; by hand, sbatch from the root

MANIFEST="${MANIFEST:?submit via case/scale_push/submit.py, which sets MANIFEST}"
CFG=$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i {print $2}' "$MANIFEST")
[ -n "$CFG" ] || { echo "no manifest line for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}"; exit 1; }

export PYTHONFAULTHANDLER=1   # a silent death is a segfault first — always get the traceback
EXTRA=${SHARED:+--shared}     # submit.py --workers N: one of several workers on the cell, windows claimed through the store
echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  task ${SLURM_ARRAY_TASK_ID}  restarts ${SLURM_RESTART_COUNT:-0}  stage ${STAGE:-inference}${EXTRA:+ (shared worker)}  cell ${CFG}  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

pid=""
trap 'echo "SIGTERM at $(date): forwarding to poreml"; [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null' TERM
.venv/bin/python case/scale_push/run.py "${STAGE:-inference}" --cell "$CFG" $EXTRA &   # STAGE=render: only the GIFs (submit.py --render)
pid=$!
set +e
wait "$pid"; rc=$?
if [ "$rc" -gt 128 ] && kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; fi   # the trap interrupted the first wait
set -e

if [ "$rc" -eq 75 ]; then
    echo "=== ${CFG} stopped by SIGTERM with its windows stored; requeueing  $(date) ==="
    scontrol requeue "${SLURM_JOB_ID}" || true
    exit 0
fi
echo "=== done ${CFG}  exit ${rc}  $(date) ==="
exit "$rc"
