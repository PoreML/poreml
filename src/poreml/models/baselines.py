"""Sanity baselines: the floor every learned model has to clear."""

from torch import Tensor, nn

from ..points import Points
from ..registry import MODELS


def recent_slice(in_channels: int, out_channels: int, recent_channel: int) -> slice:
    """The `out_channels` input channels ending at `recent_channel`: the most recent frame under the task convention."""
    if not -in_channels <= recent_channel < in_channels:
        raise ValueError(f"recent_channel {recent_channel} is out of range for in_channels={in_channels}")
    last = recent_channel % in_channels
    start = last - out_channels + 1
    if start < 0:
        raise ValueError(f"out_channels={out_channels} does not fit before recent_channel {last} in in_channels={in_channels}")
    return slice(start, last + 1)


@MODELS.register("persistence")
class Persistence(nn.Module):
    """Predict that nothing changes: return the most recent input frame.

    Parameter-free, runs anywhere, and is the leaderboard floor every learned model has
    to clear. If a model cannot beat this, it has learned nothing about the flow.

    Which channels are "the most recent frame" is the task's convention (see the `Task`
    protocol: static geometry first, history frames oldest to newest, each contributing
    `out_channels` channels, so the default recent_channel -1 is right). A task that
    stacks channels differently configures `recent_channel` through `ModelConfig.params`
    — silently returning the solid mask as the prediction would leave a floor every model
    trivially beats, with nothing to notice it by.

    Registered as a voxel model but fed either stream: on a `Points` batch it returns the
    recent channels of `feats` (the same convention), so the baseline scores beside point
    models on the same loader.
    """

    def __init__(self, in_channels: int, out_channels: int = 1, recent_channel: int = -1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.recent = recent_slice(in_channels, out_channels, recent_channel)

    def forward(self, x) -> Tensor:
        # Channel dim 1 in both layouts: (B, C, *spatial) voxels and (N, C) point features.
        feats = x.feats if isinstance(x, Points) else x
        return feats[:, self.recent]
