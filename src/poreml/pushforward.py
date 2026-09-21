"""The push-forward trick for the training loop (`train.push_forward`).

Teacher forcing shows the model true frames only; at rollout it eats its own predictions
and the errors compound. Push-forward (Brandstetter et al. 2022, "Message Passing Neural
PDE Solvers"; BubbleML, Hassan et al. 2023, `PushVelTrainer`) trains the correction: on a
pushed batch the model first steps `steps` times on its own output with gradients off,
then takes one graded step scored against the true frame that far ahead. Only the last
step is differentiated, so memory stays that of one-step training and the extra cost is
`steps` forward passes. The batches that are not pushed get Gaussian noise on their
history channels instead — BubbleML's cheap stand-in for the same distribution shift.

Both streams are covered. A voxel batch `(B, Cin, D, H, W)` slides its history channels
and appends the prediction with solid voxels forced to each field's fill
(`WindowDataset.decode`, the rollout's own rule); a `Points` batch replaces its last `F`
feature columns, the prediction already living on the input's own points. The static
channels (solid mask or geometry, then the run's conditions) are re-injected unchanged
at every step, as BubbleML feeds the true bubble marker back in.
"""

from dataclasses import replace

import torch
from torch import Tensor, nn

from .config import PushForwardConfig
from .points import Points


def probability(cfg: PushForwardConfig, step: int, total_steps: int) -> float:
    """The chance of pushing at `step`: linear from `prob_start` to `prob_end` over `total_steps`."""
    frac = min(step / total_steps, 1.0) if total_steps > 0 else 1.0
    return cfg.prob_start + (cfg.prob_end - cfg.prob_start) * frac


def n_pushes(cfg: PushForwardConfig, seed: int, step: int, total_steps: int) -> int:
    """`cfg.steps` when this step pushes, else 0.

    A pure function of (seed, step), like `train.epoch_permutation`: a resumed run makes the
    same decisions as an uninterrupted one, and the global RNG streams are left alone.
    """
    gen = torch.Generator().manual_seed(seed * 2_000_003 + step)
    return cfg.steps if torch.rand((), generator=gen).item() < probability(cfg, step, total_steps) else 0


def generator(seed: int, step: int, device: str | torch.device) -> torch.Generator:
    """The noise generator of `step`, pure in (seed, step) like `n_pushes`. The global streams
    are not used because a resumed epoch re-creates its DataLoader, whose iterator draws one
    base seed from the global RNG — one draw the uninterrupted run never made."""
    gen = torch.Generator(device=torch.device(device).type)
    gen.manual_seed(seed * 2_000_003 + step + 1_000_003)
    return gen


def perturb(inputs: Tensor | Points, std: float, n_static: int, gen: torch.Generator | None = None) -> Tensor | Points:
    """`inputs` with Gaussian noise of `std` on the history channels (everything after the
    `n_static` static ones), drawn from `gen` (None: the global RNG). Voxels: pore voxels only
    — the solid encoding stays exact, as `decode` keeps it at rollout. Points are pore voxels
    already."""
    if std <= 0:
        return inputs
    if isinstance(inputs, Points):
        feats = inputs.feats.clone()
        history = feats[:, n_static:]
        history += torch.randn(history.shape, generator=gen, device=history.device, dtype=history.dtype) * std
        return replace(inputs, feats=feats)
    out = inputs.clone()
    history = out[:, n_static:]
    fluid = (inputs[:, :1] == 0).to(history.dtype)
    history += torch.randn(history.shape, generator=gen, device=history.device, dtype=history.dtype) * (std * fluid)
    return out


def advance(dataset, inputs: Tensor | Points, pred: Tensor, n_static: int, n_fields: int) -> Tensor | Points:
    """The next model input: the history slid one frame, `pred` as the newest frame, static
    channels unchanged. `dataset.decode` forces solid voxels of a voxel prediction to fill."""
    pred = pred.float()
    if isinstance(inputs, Points):
        feats = torch.cat([inputs.feats[:, :n_static], inputs.feats[:, n_static + n_fields :], pred], dim=1)
        return replace(inputs, feats=feats)
    frame = dataset.decode(inputs, pred)
    return torch.cat([inputs[:, :n_static], inputs[:, n_static + n_fields :], frame], dim=1)


def unroll(
    model: nn.Module, dataset, inputs: Tensor | Points, n: int, n_static: int, n_fields: int, autocast
) -> Tensor | Points:
    """`inputs` after `n` ungraded steps of `model` on its own output (the model stays in
    whatever mode it is in; BubbleML pushes in train mode too)."""
    with torch.no_grad():
        for _ in range(n):
            with autocast:
                pred = model(inputs)
            inputs = advance(dataset, inputs, pred, n_static, n_fields)
    return inputs
