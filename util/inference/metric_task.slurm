#!/bin/bash
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G               # 16 workers on a 128^3 drainage or an underfill cell peak at 58G; 48G got workers OOM-killed
#SBATCH --time=02:00:00
#SBATCH --requeue

# One finished training's CPU metric stage (util/inference/inference.py --metric): the run's whole
# metric list on the keyframes its timed rollout stored, one worker per CPU, no GPU, no
# reservation. Idempotent: a rerun rewrites inference/metrics_val.csv / .json, so resubmitting is
# the recovery procedure. Uses the repo venv directly, as inference_task.slurm does.
set -euo pipefail
cd "${POREML_REPO:-${SLURM_SUBMIT_DIR:-.}}"   # the repo root: the submitter exports POREML_REPO; by hand, sbatch from the root

MANIFEST="${MANIFEST:?set MANIFEST to a <task id>\t<run dir> file under util/inference/_manifests}"
RUN=$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i {print $2}' "$MANIFEST")
[ -n "$RUN" ] || { echo "no manifest line for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}"; exit 1; }

export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1
echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  task ${SLURM_ARRAY_TASK_ID}  run ${RUN}  $(date) ==="
.venv/bin/python util/inference/inference.py "$RUN" --metric --workers "${SLURM_CPUS_PER_TASK}"
echo "=== done ${RUN}  $(date) ==="
