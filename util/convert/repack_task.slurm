#!/bin/bash
#SBATCH --job-name=poreml_repack
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --requeue

# One run's repack (2026-09-13): `poreml convert repack <run_dir>` copies the HDF5 without the
# derivable fields rho and umag — raw compressed chunks, no decode — anonymises the root
# attributes, trims the XDMF, rehashes, and records it in <run_id>.conversion.json. Idempotent:
# a repacked run is skipped, a torn copy is never renamed into place, so requeue or resubmit
# is the recovery procedure. CPU only, NFS-bound: read ~2/3 of the file, write the same.
# NO reader may have the run's HDF5 open — submit with --dependency on every job that reads it.
set -euo pipefail
cd "${POREML_REPO:-${SLURM_SUBMIT_DIR:-.}}"   # the repo root: the submitter exports POREML_REPO; by hand, sbatch from the root

MANIFEST="${MANIFEST:?submit via util/convert/repack_submit.py, which sets MANIFEST}"
RUN=$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i {print $2}' "$MANIFEST")
[ -n "$RUN" ] || { echo "no manifest line for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}"; exit 1; }

export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  task ${SLURM_ARRAY_TASK_ID}  run ${RUN}  $(date) ==="
.venv/bin/poreml convert repack "$RUN"
echo "=== done ${RUN}  $(date) ==="
