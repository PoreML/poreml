---
license: mit
pretty_name: PoreML V1 — reference model checkpoints
library_name: pytorch
datasets:
- PoreML/PoreML_data
tags:
- physics
- computational-fluid-dynamics
- porous-media
- multiphase-flow
- scientific-machine-learning
- neural-operator
- 3d
---

# PoreML V1 — reference model checkpoints

The trained weights behind the PoreML benchmark: five reference models — `unet3d`, `fno3d`,
`p3d`, `transolver`, `abupt` — on the next-frame forecasting task of
[`PoreML/PoreML_data`](https://huggingface.co/datasets/PoreML/PoreML_data), in two phases.

| Phase | Folder | Runs | Report this checkpoint |
|---|---|---:|---|
| base training, 20 effective epochs | `train/` | 35 | `ckpts/best.pt` — best one-step validation `mae@phi` |
| push-forward fine-tune of each base run | `train_push/` | 35 | `ckpts/best_rollout.pt` — best 64-step rollout `mae@phi` |

35 = 5 models × 7 tasks: the `gen` (generated media) and `all` (generated ∪ micro-CT) splits of
drainage, GDL and trapping, plus underfill's single `all` split.

## Download

With the PoreML code repository checked out and installed (`uv sync`):

```bash
uv run poreml checkpoints --dry-run                           # what is there and how big it is (7 GB)
uv run poreml checkpoints --phase train_push --which best_rollout   # the weights the paper reports
uv run poreml checkpoints --campaign drainage --model unet --kind gen
uv run poreml checkpoints                                     # everything
```

Filters are `--phase`, `--campaign`, `--model`, `--kind` and `--which`; they intersect and each
is repeatable. Files land under `case/`, which is where the push configs' `train.init_from`, the
transfer studies (`case/shift`, `case/scale`) and `util/inference` look a finished run up —
nothing has to be moved. Rerunning resumes. Without the package:

```bash
hf download PoreML/PoreML_checkpoint --local-dir case --exclude README.md checkpoints.csv
```

## Layout

The repository root is the benchmark's `case/` folder:

```
<phase>/<campaign>/<model>_<kind>/ckpts/<run>/
    config.yaml        the full config the run was trained with; a checkpoint is always scored with it
    run_meta.json      status, resolved data and split (with sha256), model size, provenance, progress
    metrics.csv        one row per epoch: losses, learning rate, every validation metric
    train_log.csv      the training loss every 100 steps
    ckpts/best.pt            best one-step validation metric
    ckpts/last.pt            the final epoch
    ckpts/best_rollout.pt    best rollout metric (push-forward runs)
checkpoints.csv        every weight file: phase, campaign, model, kind, path, epoch, bytes, sha256
```

A `.pt` is a `torch.save`d dict: `model` (the state dict), `config` (the model's name, params
and precision), `epoch`, `metrics`. `run_meta.json` of a push-forward run records the path and
sha256 of the base `best.pt` it started from; it matches `checkpoints.csv`.

Not included: optimiser state, the per-epoch rollout candidates, and stored rollout frames —
`poreml inference`, `poreml metric` and `poreml render` reproduce those from the weights.

## Use

```bash
RUN=case/train_push/drainage/unet_gen/ckpts/<run>
uv run poreml rollout -c $RUN/config.yaml --ckpt $RUN/ckpts/best_rollout.pt   # one scored 64-step rollout per validation run
uv run python case/shift_push/submit.py --dry-run                               # the transfer studies see the runs as finished
```

The runs were trained on single NVIDIA H200 GPUs with PyTorch 2.11 (CUDA 12.8), `tf32`
everywhere except AB-UPT's `bf16`. Paths inside the text files are relative to the code
repository's root; the node's hostname has been removed from `run_meta.json`, and nothing else
was changed.

## Licence

MIT for the weights of `unet3d`, `fno3d`, `p3d` and `transolver`. The `abupt` weights are
derived from the AB-UPT architecture and follow its upstream Emmi AI Non-Production License:
research and evaluation use only.
