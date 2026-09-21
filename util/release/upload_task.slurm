#!/bin/bash
#SBATCH --job-name=poreml_hfpush
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=util/release/_logs/hfpush_%j.log
# Push data/case (3.3 TB, 3 953 files) to the HuggingFace dataset repo. CPU + network only.
#
# Measured 2026-09-20: the full push took 0.88 h at 1 052 MB/s on one 8-CPU node, so a
# foreground run inside a normal allocation is usually fine; this exists for when the queue
# should own it instead. `upload_large_folder` keeps its state under
# data/case/.cache/huggingface, so a requeue or a resubmit RESUMES: files already on the Hub
# are skipped after a hash check. Rerunning is always safe.
#
# Never run two uploaders against one folder — they share that .cache state.
#
# Nothing may repack or scrub the tree while this runs — it reads every HDF5 file.
set -euo pipefail
cd "${POREML_REPO:-${SLURM_SUBMIT_DIR:-.}}"   # the repo root: the submitter exports POREML_REPO; by hand, sbatch from the root
echo "=== node $(hostname)  job ${SLURM_JOB_ID}  $(date -u) ==="
REPO="${HF_REPO:?set HF_REPO, e.g. PoreML/PoreML_data}"
.venv/bin/python util/release/upload_hf.py --repo "$REPO" --workers "${SLURM_CPUS_PER_TASK}"
echo "=== done $(date -u) ==="
