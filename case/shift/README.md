# Geo-Shift: trained on generated rocks, tested on µCT rocks

Does a model that only saw *generated* geometry transfer to real micro-CT rocks of the same
campaign at the size it was trained at? Every `<model>_gen` checkpoint (`case/train/<campaign>/<model>_gen/`)
is scored, without retraining, on every finished 128-class µCT run of its campaign. Same physics,
same conditions (M, θ), same voxel spacing; only the geometry source changes.

## Cells: 3 campaigns × 5 models

| campaign | checkpoints | tested on (`splits/shift/<campaign>.yaml`, frozen 2026-09-07) |
|---|---|---|
| drainage | `drainage/{unet,fno,p3d,transolver,abupt}_gen` | 80 runs: bentheimer 28, buffberea 26, castlegate 26 @128³ |
| gdl | `gdl/{…}_gen` | 80 runs: gdl_ct 28, gdl_ct_20 26, gdl_ct_40 26 @128×128×64 |
| trapping | `trapping/{…}_gen` | 80 runs, same families @128³ — 56 have a full window (24 are shorter than 65 frames) |

The split is train ∪ val of `splits/<campaign>/uCT.yaml`; `run.py split --check` verifies it.
A cell is `configs/shift/<campaign>/<model>_gen.yaml`: the training case, the campaign's
`data.yaml`, the study's `scheme.yaml`. It scores the newest *finished* run's `ckpts/best.pt`
with the `config.yaml` saved beside it (never the current `configs/<campaign>/`).

## Protocol (`configs/shift/scheme.yaml`)

| stage | where | what |
|---|---|---|
| `inference` | GPU | Windows start every 64 frames from frame 0; each rolls 64 steps on the model's own output. Only full windows count (a run under 65 frames yields none and is listed). 12 predicted frames per window are stored — steps 1, 7, 12, 18, 24, 30, 35, 41, 47, 53, 58, 64, every field — under `frames/<run>/w<t0>.h5`. Nothing is scored. Resumes per window. |
| `metric` | CPU, 16 cores | Every stored frame vs truth with the whole metric list: `mae`/`rel_mae` for phi and p, `mae_vec`/`rel_mae_vec` for u, `iou@phi`; saturation, volume, area, euler, mesh area, ∫H dS, ⟨H⟩ each as pred / target / abs_err / rel_err; the curvature histogram with W1 and W1/σ. Writes `metrics_test.csv` (one row per run, window, step — the raw record) and `metrics_test.json` (per-step mean: per run first, then over runs; `n_runs_per_step`, `n_windows_per_step`; per gen/uCT group). |
| `report` | CPU | `curves/<metric>.svg` + `.csv` (mean vs step, one line per model) and `report.md` per campaign. |

Any other aggregation is a group-by over `metrics_test.csv`; nothing needs rescoring.

## Layout

    case/shift/
      README.md  run.py  submit.py  inference_task.slurm  metric_task.slurm
      _logs/  _manifests/                                        gitignored
      <campaign>/<model>_gen/  frames/<run>/w<t0>.h5              gitignored (~50 GB per drainage cell)
                               inference_test.json  metrics_test.csv  metrics_test.json
      <campaign>/report.md  curves/<metric>.svg  curves/<metric>.csv

## Operate

    uv run python case/shift/run.py split --check
    uv run python case/shift/submit.py --dry-run            # waiting / pending / metric / finished per cell
    uv run python case/shift/submit.py [--only gdl] [--throttle 8]     # GPU array: inference (2 days, self-requeues)
    uv run python case/shift/submit.py --metric              # CPU array for cells whose frames wait
    uv run python case/shift/run.py inference --campaign drainage --model unet     # one cell, foreground
    uv run python case/shift/run.py metric --cell configs/shift/gdl/abupt_gen.yaml --workers 16
    uv run python case/shift/run.py report

Resubmitting is the recovery procedure: finished and queued cells are skipped, an interrupted
inference resumes at its first missing window. Windows per cell: drainage 421, gdl 274, trapping 96.
AB-UPT is the slow one (~1 s per step at 128).
