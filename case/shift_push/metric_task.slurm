#!/bin/bash
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue

# One cell's metric stage: every metric on the stored frames, one worker per CPU, no GPU.
# Idempotent: rerunning rewrites metrics_test.csv / .json, so resubmitting is the recovery procedure.
set -euo pipefail
cd "${POREML_REPO:-${SLURM_SUBMIT_DIR:-.}}"   # the repo root: the submitter exports POREML_REPO; by hand, sbatch from the root

MANIFEST="${MANIFEST:?submit via case/shift_push/submit.py --metric, which sets MANIFEST}"
CFG=$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i {print $2}' "$MANIFEST")
[ -n "$CFG" ] || { echo "no manifest line for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}"; exit 1; }

export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1
echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  task ${SLURM_ARRAY_TASK_ID}  cell ${CFG}  $(date) ==="
.venv/bin/python case/shift_push/run.py metric --cell "$CFG" --workers "${SLURM_CPUS_PER_TASK}"
echo "=== done ${CFG}  $(date) ==="
