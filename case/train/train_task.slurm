#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=5-00:00:00       # the ceiling; submit.py passes the model's own limit (TIME_LIMITS) on the command line
#SBATCH --requeue
#SBATCH --signal=B:TERM@900

# One Train1-10 array task: look up this task's config in the manifest that
# case/train/submit.py wrote, and train it. 8 CPUs: the 8 loader workers spend most of
# their time waiting on the share, so they and the main process fit; 10 left tasks
# unable to use the nodes that have a spare GPU and 8 idle CPUs. Everything the run does — data,
# split, epochs, optimiser, eval cadence, out_dir — comes from that config.
#
# Preemption-safe. `poreml train --resume auto` continues the newest run of this
# config that stopped short (ckpts/resume.pt, written every train.checkpoint_every
# seconds, at every epoch end and on SIGTERM), else starts fresh. SIGTERM reaches
# poreml two ways: SLURM preempting the job (QOS normal is preemptable, and
# --requeue puts it back in the queue), or --signal 15 min before the time limit —
# the trap below forwards it to the python child, which saves and exits 75, on
# which this script requeues itself. Either way the task comes back and resumes.
#
# Uses the repo venv directly (not `uv run`): concurrent tasks must not race on
# an environment sync, and compute nodes should not need the network. No srun:
# its cpu-bind step fails when jobs share a node (see the solver's run_drainage.slurm).
set -euo pipefail
cd "${POREML_REPO:-${SLURM_SUBMIT_DIR:-.}}"   # the repo root: the submitter exports POREML_REPO; by hand, sbatch from the root

MANIFEST="${MANIFEST:?submit via case/train/submit.py, which sets MANIFEST}"
CFG=$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i {print $2}' "$MANIFEST")
[ -n "$CFG" ] || { echo "no manifest line for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}"; exit 1; }

export PYTHONFAULTHANDLER=1   # a silent death is a segfault first — always get the traceback

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  task ${SLURM_ARRAY_TASK_ID}  restarts ${SLURM_RESTART_COUNT:-0}  config ${CFG}  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true

pid=""
trap 'echo "SIGTERM at $(date): forwarding to poreml"; [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null' TERM
.venv/bin/poreml train -c "$CFG" --resume auto &
pid=$!
set +e
wait "$pid"; rc=$?
if [ "$rc" -gt 128 ] && kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; fi   # the trap interrupted the first wait
set -e

if [ "$rc" -eq 75 ]; then
    echo "=== ${CFG} stopped by SIGTERM with its resume state saved; requeueing  $(date) ==="
    scontrol requeue "${SLURM_JOB_ID}" || true
    exit 0
fi
echo "=== done ${CFG}  exit ${rc}  $(date) ==="
exit "$rc"
