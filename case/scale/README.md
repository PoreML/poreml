# Scale-Up: the same weights on the 256-class domains

Does a model trained on 128-class domains hold up on domains with eight times the voxels, at
fixed voxel spacing? Every `<model>_all` checkpoint (`case/train/<campaign>/<model>_all/`,
trained on generated + µCT rocks) is scored, without retraining, on the campaign's 256-class
runs, generated and µCT alike. Domain-size extrapolation, not super-resolution: voxel size and
local discretisation are unchanged while the number of pores, the connectivity paths and the
room for localised events grow. Because the `all` model has seen both geometry sources at 128,
domain size is the only thing that changes.

## Cells: 3 campaigns × 5 models

| campaign | checkpoints | tested on (`splits/scale/<campaign>.yaml`, frozen 2026-09-07) |
|---|---|---|
| drainage | `drainage/{unet,fno,p3d,transolver,abupt}_all` | 16 runs: gen (blob 3, poly 3, sphere 2) + µCT (bentheimer 3, buffberea 3, castlegate 2) @256³ |
| gdl | `gdl/{…}_all` | 16 runs: gen (fiber 8) + µCT (gdl_ct 3, gdl_ct_20 3, gdl_ct_40 2) @256×256×128 |
| trapping | `trapping/{…}_all` | 16 runs: gen (blob 3, poly 3, sphere 2) + µCT (bentheimer 3, buffberea 3, castlegate 2) @256³ |

Every campaign draws 16 finished 256-class runs, 8 generated + 8 µCT; the size class is the
rock's max axis (`_rock_shape` crops the padded domain to `regions["rock"]`). No family
exclusion, no quota — the whole pool. `run.py split --check` verifies it. A cell is
`configs/scale/<campaign>/<model>_all.yaml`: the training case, the campaign's `data.yaml`,
the study's `scheme.yaml`. It scores the newest *finished* run's `ckpts/best.pt` with the
`config.yaml` saved beside it (never the current `configs/<campaign>/`).

## Protocol (`configs/scale/scheme.yaml`)

| stage | where | what |
|---|---|---|
| `inference` | GPU | Windows start every 64 frames from frame 0; each rolls 64 steps on the model's own output. Only full windows count (a run under 65 frames yields none and is listed). 12 predicted frames per window are stored — steps 1, 7, 12, 18, 24, 30, 35, 41, 47, 53, 58, 64, every field — under `frames/<run>/w<t0>.h5`. Nothing is scored. Resumes per window. |
| `metric` | CPU, 16 cores | Every stored frame vs truth with the whole metric list: `mae`/`rel_mae` for phi and p, `mae_vec`/`rel_mae_vec` for u, `iou@phi`; saturation, volume, area, euler, mesh area, ∫H dS, ⟨H⟩ each as pred / target / abs_err / rel_err; the curvature histogram with W1 and W1/σ. Writes `metrics_test.csv` (one row per run, window, step — the raw record) and `metrics_test.json` (per-step mean: per run first, then over runs; `n_runs_per_step`, `n_windows_per_step`; per gen/uCT group). A 256-class mesh costs 30–60 s per side (measured 2026-08-31), both sides per scored frame — the reason this is a separate CPU stage on 16 cores. |
| `render` | GPU, at the end of `inference` (or alone) | `scheme.render` (3) runs per cell as truth \| prediction GIFs of one 64-step rollout starting a quarter into the run — training's periodic render (`evaluate.render_rollout`): one run per rock family, ids sorted, round-robin, so every model renders the same runs. `<cell>/render/<run_id>/rollout_<run_id>.gif` + `frames/` + `.json`. The rollout scores only the non-`eval_only` metrics for the frame labels (the mesh family at 256 would turn a 1-minute render into hours; the metric stage scores it on the stored frames). Existing GIFs are kept (`--force` redoes); a failed render is reported in `render_failed.txt` and skipped, never a failed cell. |
| `report` | CPU | `curves/<metric>.svg` + `.csv` (mean vs step, one line per model) and `report.md` per campaign. |

Any other aggregation is a group-by over `metrics_test.csv`; nothing needs rescoring.

## Layout

    case/scale/
      README.md  run.py  submit.py  inference_task.slurm  metric_task.slurm
      _logs/  _manifests/                                        gitignored
      <campaign>/<model>_all/  frames/<run>/w<t0>.h5              gitignored (~145 GB per drainage cell)
                               render/<run>/rollout_<run>.gif  frames/  rollout_<run>.json   3 runs per cell
                               inference_test.json  metrics_test.csv  metrics_test.json
      <campaign>/report.md  curves/<metric>.svg  curves/<metric>.csv

## Operate

    uv run python case/scale/run.py split --check
    uv run python case/scale/submit.py --dry-run            # waiting / pending / metric / finished per cell
    uv run python case/scale/submit.py [--only gdl] [--throttle 8]     # GPU array: inference (2 days, self-requeues)
    uv run python case/scale/submit.py --metric              # CPU array for cells whose frames wait
    uv run python case/scale/submit.py --render [--reservation NAME]   # GPU array: only the GIFs of cells that lack them
    uv run python case/scale/run.py render --cell configs/scale/gdl/abupt_all.yaml   # one cell's GIFs, foreground (GPU)
    uv run python case/scale/run.py inference --campaign drainage --model unet     # one cell, foreground
    uv run python case/scale/run.py metric --cell configs/scale/gdl/abupt_all.yaml --workers 16
    uv run python case/scale/run.py report

Resubmitting is the recovery procedure: finished and queued cells are skipped, an interrupted
inference resumes at its first missing window. Windows per cell: drainage 157, gdl 119,
trapping 48. AB-UPT is the slow one — ~31 s per step at 256, so ~90 h for a full drainage
cell (157 windows × 64 steps) — which is why the inference task's time limit is 2 days and it
self-requeues on SIGTERM rather than trying to fit in one segment. The stop flag is polled
between windows, not mid-window, so `inference_task.slurm`'s `--signal=B:TERM@3600` gives a
60 min lead — comfortably over one AB-UPT window's ~33 min at 256 — instead of shift's 900 s,
which is enough only at the 128 class.

## Known walls (findings, not fixes)

- **Point models at 256³**: ~3.5 M pore points per sample. Transolver++ (1.74 M params, fp32)
  peaked at 88.7 GiB on a 143.8 GiB H200 in the 2026-08-31 smoke — an H200 is required, a
  smaller card is a recorded OOM. AB-UPT keeps its 16 384 supernodes and 16 384 + 16 384
  anchors with every other point cross-attending; memory grows linearly in the queries.
- **FNO's 24³ modes** on GDL's 256×256×128 domain are within every axis; no change.
- **UNet/P3D receptive field** stays fixed while the domain grows — the point of the study.
- **GDL's plate-less layout** (`inbuf | rock | outbuf`) at 256×256×141: `roi: rock` crops as
  at 128; nothing size-dependent.
- A curvature sample that crashes libigl is a NaN counted in `n_invalid`, never a failed cell.

## Cost

Measured on an H200: the voxel models take ~1 min per 256-class window (~50 min trapping, ~2 h gdl, ~2.6 h
drainage per cell).
