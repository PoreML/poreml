#!/bin/bash
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue

# One array task = one the solver run re-encoded to zstd under poreml/data (see poreml.convert).
# CPU only: a task decodes gzip and encodes zstd at ~0.35 s per 128^3 frame; underfill
# frames are 2.4x larger and its files 15 GB (10-48 min each under a 32-way array); the
# 256-class files are up to 73 GB, hence the 12 h limit (a timeout loses the .tmp and requeues from scratch). The task is
# idempotent (a current copy is skipped, a torn copy is never renamed into place), so a
# requeued or resubmitted task simply redoes its run.
set -euo pipefail
cd "${POREML_REPO:-${SLURM_SUBMIT_DIR:-.}}"   # the repo root: the submitter exports POREML_REPO; by hand, sbatch from the root
MANIFEST="${MANIFEST:?submit via util/convert/submit.py}"
LINE=$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i {print $2 "\t" $3}' "$MANIFEST")
[ -n "$LINE" ] || { echo "no manifest line for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }
SRC=${LINE%%$'\t'*}; DST=${LINE#*$'\t'}
echo "=== $(hostname) job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: ${SRC} -> ${DST}  $(date) ==="
.venv/bin/poreml convert run "$SRC" "$DST"
echo "=== done $(date) ==="
