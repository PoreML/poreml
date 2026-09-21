# Push-forward fine-tuning of the benchmark matrix

Second training phase: every finished Train1-10 run under `case/train` is fine-tuned with
the **push-forward trick** so that its rollouts drift less. Same models, same splits, same
recipe; the only things that change are where the weights start (the base run's
`best.pt`), how many epochs (half), the learning rate (a third), and what the model is shown
during training.

## The trick in three sentences

One-step training only ever shows the model clean simulator frames, but at rollout it
eats its own slightly wrong frames and the errors compound. Push-forward (Brandstetter
et al. 2022, *Message Passing Neural PDE Solvers*; BubbleML, Hassan et al. 2023,
`sciml/op_lib/push_vel_trainer.py` in `../../../repo/BubbleML`) lets the model step on its
own output `N` times *without gradients*, then takes one graded step scored against the
true frame `N + 1` ahead — it learns to correct the errors it will make. Batches that are
not pushed get Gaussian noise on their history channels instead, BubbleML's cheap stand-in
for the same distribution shift. Study notes: `../../../repo/bubbleml_paper_notes.md`.

## Protocol

| Knob | Value | Why |
|---|---|---|
| start weights | base run's `ckpts/best.pt` (`train.init_from`) | a fine-tune: BubbleML found push-forward from scratch fails without a long ramp; a trained model is where their ramp ends |
| epochs | half the base: **10** gen and underfill, **5** all | manageable budget; the periodic rollout every epoch (AB-UPT: every `epochs // 5`, its rollout is the expensive part), the checkpoint of every epoch kept, the rollout-best one marked and fully evaluated when the run ends (`finalise.py`, 2026-09-10) |
| learning rate | **a third** of the base peak lr (`LR_DIVISOR`), the base schedule otherwise | a fine-tune: restarting the base schedule at its full peak would first move the weights away from the checkpoint it starts from |
| push steps `N` | **1** (code takes any `N`) | BubbleML's only push-forward result uses one ungraded step |
| push probability | linear ramp **0.5 → 1.0** over the run, drawn per step | BubbleML ramps 0 → 1 from scratch; we start at their midpoint |
| noise on the other batches | std **0.01**, pore voxels of the history channels only | BubbleML's value; the solid encoding stays exact |
| precision | **tf32** (the project default since 2026-09-10; AB-UPT stays bf16) | 2–4x faster Transolver steps at a 2e-5 deviation on phi; recorded in every artifact |
| Transolver activation checkpointing | **off** (`NO_CHECKPOINT`) | 35 GiB instead of 14 on a 143 GiB H200 buys a 2x faster step; the numbers are identical |
| everything else | the base recipe: schedule (StepLR steps rescaled to the epoch count), weight decay, batch size, loss, metrics, rollout protocol, seed | one variable at a time |

Selection stays the per-epoch val `mae@phi` on one-step windows. **Judge the phase on the
rollout columns** (`eval/metrics.csv: rollout/*` at the horizon, same protocol as the base
runs' periodic evals): one-step accuracy may get slightly worse while rollouts get better —
that is what the paper reports (UNet-mod temperature rollout error 0.074 → 0.040).

## Implementation (`src/poreml`)

- `config.py`: `train.push_forward: {steps, prob_start, prob_end, noise}` and
  `train.init_from`. Both default off/None — a config without them trains exactly as before.
- `pushforward.py`: the whole trick. `n_pushes` draws the per-step decision as a pure
  function of `(seed, step)`; `unroll` takes the ungraded steps; `advance` slides the
  history and feeds the prediction back — voxels through `WindowDataset.decode` (solid
  forced to each field's fill, the rollout's own rule), points by replacing the last `F`
  feature columns; `perturb` adds the noise from a generator that is also pure in
  `(seed, step)`. Static channels (rock mask or geometry, then `M` and `theta`) are
  re-injected unchanged at every step, as BubbleML feeds the true bubble marker back.
- `data.py` / `points.py`: `WindowDataset(future=K)` stacks the `K` next frames as the
  target, `(K, F, D, H, W)` or `(N, K, F)`; `future=1` keeps the historical shapes.
- `train.py`: one branch in the batch loop (push or perturb, then index the target `n`
  ahead), `load_init` for the fine-tune, `push_frac` (fraction of pushed batches) as a
  column of `metrics.csv`, `init_from` (path, epoch, sha256) in `run_meta.json`. The
  pushed count rides in `resume.pt`, so a resumed run is still bit-equal to an
  uninterrupted one (`tests/test_push.py`).

Cost: one extra forward per pushed batch, no extra memory (the pushed steps run under
`no_grad`). The voxel models are loader-bound, so they barely notice; the point models
run ~1.3–1.5× per batch. With half the epochs a fine-tune costs well under a base run.

## Operate

```bash
uv run python case/train_push/make_configs.py --dry-run   # which bases have finished; what would be written
uv run python case/train_push/make_configs.py             # configs/push/<campaign>/<model>_<kind>.yaml
uv run python case/train_push/submit.py --dry-run         # pending / queued / finished / resume per config
uv run python case/train_push/submit.py --reservation <name> --throttle 0   # arrays on a reservation, no throttle, job name poreml_push
squeue -u $USER -n poreml_push; tail -f case/train_push/_logs/<jobid>_<task>.log
```

`make_configs.py` derives each `configs/push/...` file from its base config and the newest
finished run under the base's `out_dir` (`tests/test_push.py` checks every shipped file
against `derive`); a base still training is reported as *waiting* — rerun the script when
it has finished. `submit.py` wraps `case/train/submit.py` (same queue guards, same
`train_task.slurm` worker with `--resume auto`, requeue on preemption and before the time
limit) with its own job name, manifests and logs, so it never collides with the training
arrays. To change the protocol, edit `PUSH` / the epoch rule in `make_configs.py` and
regenerate: the configs are the record.

Runs land in `case/train_push/<campaign>/<model>_<kind>/ckpts/<name>_<stamp>/` with the
same artifacts as the base runs (`config.yaml`, `run_meta.json`, `metrics.csv` with the
extra `push_frac` column, `train_log.csv`, `ckpts/`, `eval/`).

## Do not

- Edit `src/poreml/` while a training array runs (lazy imports in the periodic eval).
  The push-forward changes were made additive and default-off for that reason: a running
  or resumed base run trains bit-identically with or without them.
- Fine-tune from a base that is not `finished`: its `best.pt` is still moving.
