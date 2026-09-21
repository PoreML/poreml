# Scale-Up on the push-forward checkpoints

`case/scale` repeated with one change: every cell scores the **push-forward fine-tune's
rollout-best checkpoint** (`case/train_push/<campaign>/<model>_all/ckpts/<run>/ckpts/best_rollout.pt`,
chosen by `poreml.finalise`) instead of the base run's one-step `best.pt`. Protocol
(`configs/scale_push/scheme.yaml`, incl. the 3 rendered runs per cell), campaigns, models and the
frozen 256-class test splits (`splits/scale/<campaign>.yaml`) are identical to `case/scale`, so every
number here is directly comparable with its `case/scale` counterpart.

The configs are generated: `uv run python util/configs/make_push_study_configs.py` derives
`configs/scale_push/` from `configs/scale/`. `run.py`, `submit.py` and the slurm task files are
`case/scale`'s with the study name swapped (`STUDY = "scale_push"`, job name `poreml_scale_push`).

Six push runs (drainage fno/transolver gen+all, gdl transolver gen+all) had their `best_rollout.pt`
marked on 2026-09-12 by selection between `best.pt` / `last.pt` only — see `case/shift_push/README.md`.

## Operate

    uv run python case/scale_push/submit.py --dry-run
    uv run python case/scale_push/submit.py --reservation <name> --throttle 0    # GPU arrays, fast-first
    uv run python case/scale_push/submit.py --metric --reservation <name> --throttle 4   # CPU metric array
    uv run python case/scale_push/submit.py --render --reservation <name>            # GIFs only, if missing
    uv run python case/scale_push/run.py report

Cost, measured 2026-09-12 on the trapping cells: voxel models ~1 min per 256-class window; Transolver
~1.6 min per window (tf32), AB-UPT ~3.7 min (bf16, ~3.5 s per step — `case/scale/README.md`'s 31 s per
step is the fp32 figure), so a drainage AB-UPT cell is ~10 GPU-h, not 90. Host RSS is a few GB (unet
10.6, abupt 5.6, transolver 2.4) plus the render stage's frame stack; the point cells' original 200G
request idled two reservation GPUs for want of RAM, hence 64G now. The render stage (3 GIFs, ~1 h per
cell of pyvista on CPU) holds the task's GPU idle — the one unavoidable idle hour per unshared cell.

**Shared workers** (`submit.py --workers N`, `run.py inference --shared`, `poreml rollout --inference
--shared`): N tasks of one cell run against one frame store, each claiming a window through an
exclusively created `<window>.h5.claim` marker before computing it and skipping a window another
worker holds (`frames.H5FrameStore.claim`; a marker untouched for 30 min is a dead worker's and is
taken over). No fixed shares: a worker that starts late takes what is left, one that finds nothing
exits, the worker that finds every window stored writes the sentinel, and the GIFs are left to
`submit.py --render`. An *unshared* task ignores claims, so never mix the two on one cell — cancel
the unshared task first (note that `scancel` on a running inference task fires its SIGTERM
self-requeue: it comes back pending, and needs a second `scancel`). `--qos`/`--time` pass through to
sbatch so extra workers can go off the reservation as short backfill tasks; `--add` puts workers on a
cell the queue already holds (shared tasks only).

## Status

2026-09-12: 15 cells submitted on the reservation (see `_manifests/`); a finisher job submits the
metric array after each inference array and the report after that (`_logs/finish_after_*.sh`).
Same night: the 9 voxel cells and both trapping point cells done in one task each; the three cells
still pending (gdl abupt, drainage abupt, drainage transolver) resubmitted as shared workers — arrays
37583 (transolver ×2) and 37584 (abupt ×4 each) — with their own finisher (`--render` for the GIFs,
then `--metric`, then the report).
