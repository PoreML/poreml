# util

Tools around the benchmark. The studies themselves live under `case/`.

| Path | What it does |
|---|---|
| `download.py` | `poreml download` as a script: fetch the dataset from HuggingFace, whole or in part, into `data/case` |
| `inference/` | timed validation rollouts of every finished training under `case/train` and `case/train_push` — `inference.py` (GPU stage, `--metric` for the CPU stage), `submit.py` (the same as SLURM arrays), the two workers |
| `configs/` | `make_push_study_configs.py` derives `configs/shift_push/` and `configs/scale_push/` from the base study configs |
| `convert/` | how the dataset was built from the solver's raw output: re-encoding to zstd (`submit.py`), dropping the derivable fields (`repack_submit.py`), anonymising (`anonymise_task.slurm`); all through `poreml convert` |
| `release/` | publishing: `upload_hf.py` pushes `data/case` to the Hub (rerunning resumes), `export_anonymous.sh` exports this repository for double-blind review, `check_anonymous.py` sweeps a tree for identifying strings |
| `archive/` | development history, kept for the authors and left out of exports: the dev-era per-model cases, their template configs and split, design specs and plans, recovery notes, one-off scripts |

Everything here is run from the repository root, e.g. `uv run python util/inference/submit.py --dry-run`.
SLURM workers take the partition and account from `SBATCH_PARTITION` / `SBATCH_ACCOUNT`.
