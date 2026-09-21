"""A plain 3-D UNet: the dense-grid baseline every physics-ML benchmark starts from.

Encoder–decoder with `depth` resolution levels, `base` channels doubling per level, two
`Conv3d → GroupNorm → SiLU` per block, stride-2 convolutions down, transposed
convolutions up, skip concatenation, and a 1×1×1 head to `out_channels`.

`residual=True` adds the most recent input frame (all its fields) to the head's output,
so the network predicts the *change* between two saved frames. Between saved frames the
front moves a few voxels, so most of the target is the input; predicting the delta puts
the capacity where the change is, and the zero-initialised head makes an untrained UNet
exactly persistence on every field — training starts from the floor, not from noise.
Which channels are "the most recent frame" is the task's convention (see `Task`), same
as persistence.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..registry import MODELS
from .baselines import recent_slice


def _block(cin: int, cout: int, groups: int) -> nn.Sequential:
    def conv(i: int, o: int) -> list[nn.Module]:
        return [nn.Conv3d(i, o, 3, padding=1, bias=False), nn.GroupNorm(min(groups, o), o), nn.SiLU(inplace=True)]

    return nn.Sequential(*conv(cin, cout), *conv(cout, cout))


@MODELS.register("unet3d")
class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base: int = 16,
        depth: int = 3,
        residual: bool = True,
        recent_channel: int = -1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.residual = residual
        self.recent = recent_slice(in_channels, out_channels, recent_channel)

        widths = [base * 2**i for i in range(depth + 1)]  # depth encoder levels + bottleneck
        self.stem = _block(in_channels, widths[0], groups)
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i in range(depth):
            self.down.append(nn.Conv3d(widths[i], widths[i + 1], 2, stride=2))
            self.enc.append(_block(widths[i + 1], widths[i + 1], groups))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(nn.ConvTranspose3d(widths[i + 1], widths[i], 2, stride=2))
            self.dec.append(_block(2 * widths[i], widths[i], groups))
        self.head = nn.Conv3d(widths[0], out_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        # An axis the depth cannot halve (underfill's 23-voxel chip gap) is replicate-padded on
        # its far side up to the next multiple of 2**depth and the output cropped back, as P3D
        # does: the padded slab copies the boundary voxels (solid stays solid, phi at its fill)
        # and never reaches the loss or a metric. Even shapes pad by 0 — the path is unchanged.
        factor = 2**self.depth
        shape = tuple(x.shape[2:])
        pads = [(-s) % factor for s in shape]
        if any(pads):
            x = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]), mode="replicate")
        h = self.stem(x)
        skips = []
        for down, enc in zip(self.down, self.enc, strict=True):
            skips.append(h)
            h = enc(down(h))
        for up, dec in zip(self.up, self.dec, strict=True):
            h = dec(torch.cat([up(h), skips.pop()], dim=1))
        out = self.head(h)
        if self.residual:
            out = out + x[:, self.recent]
        return out[..., : shape[0], : shape[1], : shape[2]] if any(pads) else out
