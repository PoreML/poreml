# Training the benchmark matrix

One SLURM array trains **5 models × 7 tasks = 35 runs** (gen and all for drainage, GDL and
trapping, plus underfill), one GPU each, at most 20 at a time. Everything a run does comes
from its config in `configs/<campaign>/<model>_<kind>.yaml`; this folder only launches,
resumes and collects. The train-on-uCT-only oracle is parked under
`configs/<campaign>/extra/` (`submit.py --extra`); the uCT splits stay frozen because
their val lists are the Geo-Shift test sets.
A second phase fine-tunes every finished run with the push-forward trick, half the
epochs, from its `best.pt`: `../train_push/README.md`.

| Kind | Split (train/val) | Runs |
|---|---|---|
| gen | `splits/<campaign>/gen.yaml`, 64/16 generated rocks | Train1, 4, 7 |
| all | `splits/<campaign>/all.yaml`, 128/32 = gen ∪ uCT | Train3, 6, 9 |
| underfill | `splits/underfill/all.yaml`, 25/7 flipchip | Train10 |
| *uCT (extra)* | `splits/<campaign>/uCT.yaml`, 64/16 | Train2, 5, 8 |

## Protocol: 20 effective epochs

Every model makes the same number of passes over the same runs: **20 epochs on gen and
uCT, 10 on all** (all is the union of the two, twice the runs, so 10 epochs is the same
run-passes and windows seen), **20 on underfill** (a single split, not a union).
`tests/test_config.py` enforces it together with the knobs derived from the epoch count:
the periodic eval every epochs/5 (5 per run), FNO's StepLR halving every epochs/5, P3D's
one halving at two thirds of the run. Selection is the best per-epoch val `mae@phi`;
periodic evals are reporting only.

## Model sizes (2026-09-03)

Matched near 10 M parameters on the 128-class campaigns: unet3d 11.2 M (`depth 4, base 20,
groups 4`), fno3d 9.7 M (`modes 24³, hidden 18`), p3d-S 10.7 M (`size: s`), abupt 10.5 M
(8 + 8 blocks). Transolver stays at 1.74 M (`dim 256, depth 4`): every parameter of it is a
per-point matmul, so 10 M costs 5.6× the FLOPs (measured, `tests/test_output/flops.py`) on a
model that is already GPU-bound at 0.56 s/window — ~400 h per drainage run in fp32. Underfill
keeps its shape-forced unet (depth 1) and fno (11 modes on the gap axis) and takes only
p3d-S and abupt 8 + 8. Forward FLOPs per 128^3 window at these sizes: fno ~60 G (incl. FFTs),
p3d-S 256 G, unet 416 G, transolver 1.8 T, abupt 5.7 T (494k points).

## Cost estimate

Seconds per window measured 2026-09-02 on an idle H200 with 8 loader workers reading the
zstd mirror: UNet and FNO 0.065 (loader-bound: two cold frames from the share cost a
worker ~0.5 s), P3D 0.08 (GPU), Transolver 0.56 (GPU, fp32), AB-UPT 0.22 (bf16). GDL
windows cost 0.6× of drainage, underfill 2.4× (measured for the voxel models; assumed for
the point models). Training hours per run:

| Task | epochs | windows/ep | windows seen | UNet | FNO | P3D | Transolver | AB-UPT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drainage gen | 20 | 23.1k | 461k | 8 | 8 | 10 | 72 | 28 |
| drainage all | 10 | 45.8k | 458k | 8 | 8 | 10 | 71 | 28 |
| GDL gen | 20 | 20.3k | 406k | 4 | 4 | 5 | 38 | 15 |
| GDL all | 10 | 36.7k | 367k | 4 | 4 | 5 | 34 | 13 |
| trapping gen | 20 | 8.1k | 162k | 3 | 3 | 4 | 25 | 10 |
| trapping all | 10 | 14.9k | 149k | 3 | 3 | 3 | 23 | 9 |
| underfill | 20 | 9.3k | 187k | 8 | 8 | 10 | 70 | 27 |
| **total** | | | | **39** | **39** | **48** | **333** | **131** |

**Training ≈ 589 GPU-h for the main phase.** The periodic evals add about
350 GPU-h on top (5 per run, 1–5 h each: curvature on CPU plus a 64-step rollout of the
val split), so budget **≈ 939 GPU-h**. Wall-clock is set by Transolver on
drainage (~72 h training + evals ≈ 3.5 days) and on underfill (~70 h on the assumed rate):
both fit the 5-day limit in one segment, and the resume path covers a miss. The extra uCT
phase would add about a third.

## What makes the numbers

In effect on every run, no flags needed:

- **zstd mirror** (`data/case`, `poreml convert`, `.claude/skills/converter/`): the solver's
  runs re-encoded bit-identically; frame decode 134 → 37 ms.
- **In-place frame assembly** (`WindowDataset.frame`): scalar fields read cropped straight
  into a preallocated output, velocity decoded once and split with one strided copy, three
  in-place passes instead of seven allocating ones — numpy 161 → 52 ms per frame,
  bit-identical output guarded by `tests/test_data_windows.py`.
- **8 loader workers on 8 CPUs** per task (2026-09-03, was 10: the workers spend most of
  a window waiting on the share, so they and the main process share 8 cores with little
  loss, and 8 is what the nodes with a spare GPU tend to have idle; the node's fair share
  is 13 per GPU).
- **Per-model time limits** (`submit.py: TIME_LIMITS`): unet3d, fno3d and p3d ask for one
  day, abupt two, transolver the five-day ceiling — one job array per limit, the
  `--throttle` split between them by size. A short request is what lets the scheduler
  backfill a task into a gap before the next high-priority job; a run that outlives its
  limit is saved 15 min before it and requeued as another segment, at no cost to results.
- **AB-UPT in bf16** (`model.precision`, its class default) with attention pinned off cuDNN
  (a plan per new token count made bf16 5× slower until excluded): 0.50 → ~0.25 s/window,
  same loss; eval stays fp32.
- **Checkpoint/resume**: `ckpts/resume.pt` every 30 min, at epoch ends and on SIGTERM; the
  worker runs `poreml train --resume auto`, requeues itself 15 min before the time limit,
  and SLURM requeues on preemption. A resumed run is bit-equal to an uninterrupted one.

Measured, not switched on: TF32 matmul for Transolver (2.2×) or bf16 (3.3×, val noisier
over 240 steps) — upstream trains it fp32. Designed, not built: a node-local file cache on
`/scratch` (cold frame 288 → ~100 ms, takes the array off the share), a faster point
subsample (111 → ~10 ms per AB-UPT window), `eval.stride` 16 to halve the eval cost.

What is left in a window: fetching one frame's compressed chunks from the share costs
~190 ms of latency (the solver's one-slab chunks are 384 requests per frame), decode 44, numpy 50.

## Per-model recipe

| Model | Recipe |
|---|---|
| UNet3D | AdamW 3e-4, constant |
| FNO 3-D | 2.4M params, H1 loss, StepLR ×½ every epochs/5, wd 1e-4 |
| P3D-b | lr 2e-4, clip 1.0, StepLR ×½ once at ⅔ of the run |
| Transolver++ | OneCycle 1e-3, `geometry_radii [1,2,4,8]`, fp32 (tf32 for anything started after 2026-09-10 — the project default) |
| AB-UPT | OneCycle 3e-4, wd 0.05, 16k supernodes, bf16 |

All models: conditions `M` and `theta` (never `ca`), wall channel on the point stream,
`val_stride 8` for the selection pass, `rollout_stride 64` in the periodic eval.

## Outputs, checkpoints, evaluation cadence

Every task writes one run directory (`poreml checkpoints --phase train` downloads the finished
ones — config, metadata, logs, `best.pt` and `last.pt` — into this same layout, and `submit.py`
then skips those cells):
```
case/train/<campaign>/<model>_<kind>/ckpts/<name>_<stamp>/
  config.yaml          the config as trained (resolved defaults included)
  run_meta.json        status running/finished/preempted/failed, segments, best epoch, model precision
  metrics.csv/.jsonl   one row per epoch: train loss, seconds, peak memory, lr, every validation metric
  train_log.csv        loss every 100 batches — the first hour's rows give the true s/window
  ckpts/last.pt        weights after the latest epoch
  ckpts/best.pt        weights of the best val mae@phi epoch — the checkpoint the benchmark scores
  ckpts/resume.pt      weights + optimiser + scheduler + position + RNG, for continuing in place
  eval/epoch_<k>/      periodic evaluation: results_val.json, distributions_val.csv,
                       rollout_val.json, rollout_<run>.json + .gif of the first val run
  eval/metrics.csv     one row per periodic evaluation (<metric>, rollout/<metric>); curves.svg
```

| What | When | On what | Cost |
|---|---|---|---|
| `train_log.csv` row | every 100 batches | — | none |
| `resume.pt` | every 30 min, every epoch end, on SIGTERM | — | seconds |
| selection pass → `metrics.csv`, `last.pt`, `best.pt` | every epoch | val split, every 8th window (`val_stride`), validation metrics only (no curvature) | minutes |
| periodic eval → `eval/epoch_<k>/` | every epochs/5, so 5 per run, always including the last | val split: every 8th window with the **full** metric list incl. curvature; rollout of every val run over 64 steps scored at the horizon; one GIF | 1–5 h |
| benchmark scoring (`poreml eval` / `rollout`) | after training, separate stage | test splits, `best.pt`, every window, every step | — |

Selection never depends on the periodic eval, and no evaluation depends on the training
precision: all of them run in fp32.

## Operate

```bash
uv run python case/train/submit.py --dry-run      # pending / queued / finished / resume per config
uv run python case/train/submit.py                # submit the pending tasks as one array (%20)
uv run python case/train/submit.py --only underfill --throttle 8
squeue -u $USER -n poreml_train; tail -f case/train/_logs/<jobid>_<task>.log
```

Resubmitting is the whole recovery procedure: a config that is finished or still in the
queue (running *or pending*, resolved through `_manifests/job_<id>.tsv`) is skipped, a run
that stopped short is continued in place. Runs land in
`case/train/<campaign>/<model>_<kind>/ckpts/<name>_<stamp>/` with `config.yaml`,
`run_meta.json` (status, segments, best epoch), `metrics.csv`, `train_log.csv`, `ckpts/`,
`eval/`. Never edit `src/poreml/` while the array runs (lazy imports), never touch the
frozen splits, and submit underfill first: its per-window rates are the only unmeasured
ones, and `train_log.csv` shows them within the hour.
