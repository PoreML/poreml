# Geo-Shift on the push-forward checkpoints

`case/shift` repeated with one change: every cell scores the **push-forward fine-tune's
rollout-best checkpoint** (`case/train_push/<campaign>/<model>_gen/ckpts/<run>/ckpts/best_rollout.pt`,
chosen by `poreml.finalise`) instead of the base run's one-step `best.pt`. Protocol
(`configs/shift_push/scheme.yaml`), campaigns, models and the frozen test splits
(`splits/shift/<campaign>.yaml`, via `configs/shift_push/<campaign>/data.yaml`) are identical to
`case/shift`, so every number here is directly comparable with its `case/shift` counterpart.

The configs are generated: `uv run python util/configs/make_push_study_configs.py` derives
`configs/shift_push/` from `configs/shift/` (train_case -> case/train_push, which -> best_rollout,
out_dir -> case/shift_push). `run.py`, `submit.py` and the two slurm task files are `case/shift`'s
with the study name swapped (`STUDY = "shift_push"`, job name `poreml_shift_push`).

Six push runs finished before per-epoch checkpoints and the finalise step existed
(drainage fno/transolver gen+all, gdl transolver gen+all); their `best_rollout.pt` was marked on
2026-09-12 by selection between the evaluated `best.pt` / `last.pt` only (`progress.final.error`
in their run_meta.json says so), without the final val evaluation.

## Operate

    uv run python case/shift_push/submit.py --dry-run
    uv run python case/shift_push/submit.py --reservation <name> --throttle 0    # GPU inference array
    uv run python case/shift_push/submit.py --metric --reservation <name> --throttle 8   # CPU metric array
    uv run python case/shift_push/run.py report

Layout, stages and states: see `case/shift/README.md`. Outputs: `case/shift_push/<campaign>/<model>_gen/`
(frames/, inference_test.json, metrics_test.csv/.json) and per campaign `report.md` + `curves/`.

## Status

2026-09-12: 15 cells submitted on the reservation (see `_manifests/`); a finisher job submits the
metric array after the inference array and the report after that (`_logs/finish.sh`).
