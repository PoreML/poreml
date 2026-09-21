"""Invariants of the P3D port: windowing, the relative-position bias, conditioning, the wrapper.

Numerical parity against tum-pbs/P3D was checked separately with a weight-copying harness
(`util/archive/docs/sketch/p3d.md` §5); these tests pin what that harness cannot, because upstream is not
importable here.
"""

import pytest
import torch

from poreml.config import ModelConfig
from poreml.models import build_model, count_parameters
from poreml.models.p3d import P3D, PRESETS
from poreml.models.p3d_blocks import (
    ConditionedConv3DBlock,
    Conditioning,
    P3DStage,
    PixelShuffle3D,
    RelativePositionBias,
    WindowAttention3D,
    timestep_embedding,
    window_partition,
    window_reverse,
)


def test_window_partition_round_trips():
    x = torch.randn(2, 8, 8, 4, 3)
    windows = window_partition(x, 4)
    assert windows.shape == (2 * 2 * 2 * 1, 64, 3)
    assert torch.equal(window_reverse(windows, 4, 8, 8, 4, 2), x)


def test_pixel_shuffle_moves_channels_into_space():
    x = torch.arange(8 * 2 * 2 * 2, dtype=torch.float32).view(1, 8, 2, 2, 2)
    out = PixelShuffle3D(2)(x)
    assert out.shape == (1, 1, 4, 4, 4)
    assert sorted(out.flatten().tolist()) == sorted(x.flatten().tolist())  # a permutation, nothing lost
    with pytest.raises(ValueError, match="divisible"):
        PixelShuffle3D(2)(torch.randn(1, 7, 2, 2, 2))


def test_relative_position_bias_depends_only_on_the_offset():
    bias = RelativePositionBias(window_size=4, num_heads=2)
    out = bias(torch.zeros(1, 2, 64, 64))
    assert out.shape == (1, 2, 64, 64)
    assert (out > 0).all() and (out < 16).all()  # 16 * sigmoid
    # voxels (0,0,0)->(0,0,1) and (1,1,1)->(1,1,2) share an offset, so they share a bias
    flat = lambda i, j, k: (i * 4 + j) * 4 + k  # noqa: E731
    assert torch.allclose(out[0, :, flat(0, 0, 0), flat(0, 0, 1)], out[0, :, flat(1, 1, 1), flat(1, 1, 2)])


def test_window_attention_of_a_constant_field_is_constant():
    torch.manual_seed(0)
    attn = WindowAttention3D(dim=8, num_heads=2, window_size=4).eval()
    x = torch.ones(3, 64, 8)
    out = attn(x)
    assert out.shape == (3, 64, 8)
    assert torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)


def test_stage_pads_grids_that_the_window_does_not_divide():
    torch.manual_seed(0)
    stage = P3DStage(dim=8, depth=2, num_heads=2, window_size=4)
    x = torch.randn(2, 8, 6, 4, 10)
    assert stage(x, torch.randn(2, 8)).shape == x.shape


def test_conditioned_block_is_the_identity_when_its_convolutions_are_zero():
    block = ConditionedConv3DBlock(4, embed_dim=6, num_groups=2)
    torch.nn.init.zeros_(block.conv_2.weight)
    torch.nn.init.zeros_(block.conv_2.bias)
    x = torch.randn(2, 4, 4, 4, 4)
    assert torch.equal(block(x, torch.randn(2, 6)), x)  # the residual path carries everything


def test_timestep_embedding_of_zero_is_cosines_then_sines():
    emb = timestep_embedding(torch.zeros(3), 8)
    assert emb.shape == (3, 8)
    assert torch.allclose(emb[:, :4], torch.ones(3, 4)) and torch.allclose(emb[:, 4:], torch.zeros(3, 4))


def test_conditioning_is_shared_across_a_batch_of_identical_inputs():
    cond = Conditioning(embedding_dim=16).eval()
    out = cond(torch.zeros(4), torch.zeros(4))
    assert out.shape == (4, 16)
    assert torch.allclose(out, out[:1].expand_as(out))


def test_p3d_pads_shapes_its_downsampling_cannot_halve_and_crops_back():
    # Underfill's 26x482x476 domain divides by no preset's factor, so non-divisible
    # inputs are replicate-padded up to the next multiple and the output cropped back —
    # a no-op for divisible domains, like fno's domain_padding.
    model = P3D(in_channels=2, out_channels=1, size="s")
    assert model.factor == 16

    out = model(torch.randn(1, 2, 12, 18, 20))

    assert out.shape == (1, 1, 12, 18, 20)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match=r"\(B, C, D, H, W\)"):
        model(torch.randn(1, 2, 32, 32))


def test_p3d_rejects_an_unknown_size():
    with pytest.raises(ValueError, match="unknown size"):
        build_model(ModelConfig(name="p3d", params={"size": "tiny"}), in_channels=2, out_channels=1)


def test_p3d_presets_grow_and_match_upstream_counts_minus_the_label_tables():
    counts = {size: count_parameters(P3D(in_channels=6, out_channels=5, size=size)) for size in ("s", "b", "l")}
    assert counts["s"] < counts["b"] < counts["l"]
    # upstream reports 11.2M / 46.2M / 181M including a 1001-row label table per level, which this
    # port replaces with one vector: 1001 * (sum of level widths + the wrapper's embedding width)
    for size, published in (("s", 11.2e6), ("b", 46.2e6), ("l", 181e6)):
        hidden, _, _, _, _ = PRESETS[size]
        tables = 1001 * (hidden + 2 * hidden + 4 * hidden + 64)
        assert abs(counts[size] + tables - published) / published < 0.02


def test_p3d_starts_as_persistence_and_trains():
    torch.manual_seed(0)
    model = build_model(ModelConfig(name="p3d", params={"size": "s"}), in_channels=4, out_channels=3)
    x = torch.randn(1, 4, 16, 16, 16)
    model.eval()

    assert torch.allclose(model(x), x[:, 1:4], atol=1e-6)

    model.train()
    out = model(x)
    out.pow(2).mean().backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert unused == [], unused
