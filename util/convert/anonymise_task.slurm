#!/bin/bash
#SBATCH --job-name=poreml_anonymise
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=util/convert/_logs/anonymise_%j.log
# The HDF5 pass of `poreml convert anonymise` over data/case (2026-09-13): rewrite the
# root attribute `run_meta` of every run's HDF5 (the cluster username -> poreml_author) and
# recompute the sha256 in its conversion record. HDF5 has no writer-beside-readers mode, so
# submit this with --dependency=afterany:<every job that reads the tree> and start nothing
# that reads data/case until it has finished. The JSON pass runs separately, at once
# (atomic replaces are safe beside readers). CPU only; the hash's read of 4.5 TB over NFS
# is the cost (~225 MB/s per stream, 8 in parallel).
set -euo pipefail
cd "${POREML_REPO:-${SLURM_SUBMIT_DIR:-.}}"   # the repo root: the submitter exports POREML_REPO; by hand, sbatch from the root
echo "=== node $(hostname)  job ${SLURM_JOB_ID}  $(date) ==="
squeue -h -u "$USER" -o "%j %T" | sort | uniq -c || true
.venv/bin/poreml convert anonymise data/case --no-json --h5 --rehash --workers "${SLURM_CPUS_PER_TASK}" \
  --ledger "util/convert/_logs/anonymise_h5_${SLURM_JOB_ID}.json"
echo "=== verifying: no username left in any root attribute or JSON  $(date) ==="
.venv/bin/poreml convert anonymise data/case --dry-run --workers "${SLURM_CPUS_PER_TASK}"
echo "=== done  $(date) ==="
