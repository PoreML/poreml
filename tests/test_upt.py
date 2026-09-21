import math

import torch

from poreml.models.upt import (
    AnchorAttention,
    ContinuousSincosEmbed,
    PerceiverBlock,
    RopeFrequency,
    SharedweightsCrossAttention,
    SharedweightsSplitAttention,
    SupernodePooling,
    TransformerBlock,
    radius_neighbours,
    rope,
)


def test_sincos_embed_has_the_requested_width_and_zero_padding():
    emb = ContinuousSincosEmbed(dim=12, ndim=3)
    out = emb(torch.rand(2, 5, 3) * 100)
    assert out.shape == (2, 5, 12)
    emb4 = ContinuousSincosEmbed(dim=12, ndim=4)  # 12 % 4 == 0 but 3 per axis is odd -> 4 zero pads
    out4 = emb4(torch.rand(5, 4))
    assert out4.shape == (5, 12) and (out4[:, -4:] == 0).all()


def test_rope_is_a_rotation_that_preserves_norms():
    freq = RopeFrequency(dim=6, ndim=3)
    f = freq(torch.rand(1, 4, 3) * 100)
    assert f.shape == (1, 4, 3) and f.is_complex()
    x = torch.randn(1, 2, 4, 6)
    y = rope(x, f)
    assert y.shape == x.shape
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_anchor_attention_queries_only_read_anchors():
    torch.manual_seed(0)
    attn = AnchorAttention(dim=12, num_heads=2).eval()
    freqs = RopeFrequency(dim=6, ndim=3)(torch.rand(1, 6, 3) * 10)
    x = torch.randn(1, 6, 12)
    out = attn(x, freqs=freqs, num_anchor_tokens=4)
    assert out.shape == (1, 6, 12)
    # changing a query token changes only its own output, not the anchors' or other queries'
    x2 = x.clone()
    x2[0, 5] += 1.0
    out2 = attn(x2, freqs=freqs, num_anchor_tokens=4)
    assert torch.allclose(out[0, :5], out2[0, :5], atol=1e-6) and not torch.allclose(out[0, 5], out2[0, 5])


def test_shared_split_and_cross_attention_shapes():
    torch.manual_seed(0)
    freqs = RopeFrequency(dim=6, ndim=3)(torch.rand(1, 10, 3) * 10)
    x = torch.randn(1, 10, 12)
    split = [3, 2, 4, 1]  # surface anchors, surface queries, volume anchors, volume queries
    assert SharedweightsSplitAttention(dim=12, num_heads=2)(x, split_size=split, freqs=freqs).shape == (1, 10, 12)
    assert SharedweightsCrossAttention(dim=12, num_heads=2)(x, split_size=split, freqs=freqs).shape == (1, 10, 12)
    two = [4, 6]
    assert SharedweightsSplitAttention(dim=12, num_heads=2)(x, split_size=two, freqs=freqs).shape == (1, 10, 12)
    assert SharedweightsCrossAttention(dim=12, num_heads=2)(x, split_size=two, freqs=freqs).shape == (1, 10, 12)


def test_blocks_keep_shape():
    freqs = RopeFrequency(dim=6, ndim=3)(torch.rand(1, 5, 3) * 10)
    kf = RopeFrequency(dim=6, ndim=3)(torch.rand(1, 3, 3) * 10)
    x, kv = torch.randn(1, 5, 12), torch.randn(1, 3, 12)
    assert TransformerBlock(dim=12, num_heads=2)(x, freqs=freqs).shape == (1, 5, 12)
    assert PerceiverBlock(dim=12, num_heads=2)(x, kv, q_freqs=freqs, k_freqs=kf).shape == (1, 5, 12)


def test_radius_neighbours_on_the_grid_are_exact_and_capped():
    # 5^3 grid, every voxel a point; the centre voxel within radius 1 has itself + 6 face neighbours
    grid = torch.arange(125).view(5, 5, 5)
    src, dst = radius_neighbours(grid, torch.tensor([[2, 2, 2]]), radius=1.0, max_degree=32)
    assert dst.tolist() == [0] * 7 and sorted(src.tolist()) == sorted([62, 37, 87, 57, 67, 61, 63])
    # radius sqrt(2) adds the 12 edge neighbours; the self-loop comes first
    src, _ = radius_neighbours(grid, torch.tensor([[2, 2, 2]]), radius=math.sqrt(2) + 1e-6, max_degree=32)
    assert len(src) == 19 and src[0] == 62
    # max_degree caps, nearest first
    src, _ = radius_neighbours(grid, torch.tensor([[2, 2, 2]]), radius=2.0, max_degree=3)
    assert len(src) == 3 and src[0] == 62
    # a corner voxel: neighbours outside the grid and voxels without a point (-1) are skipped
    grid2 = grid.clone()
    grid2[0, 0, 1] = -1
    src, _ = radius_neighbours(grid2, torch.tensor([[0, 0, 0]]), radius=1.0, max_degree=32)
    assert sorted(src.tolist()) == [0, 5, 25]


def test_supernode_pooling_returns_one_token_per_supernode():
    torch.manual_seed(0)
    n = 27
    ijk = torch.stack(torch.meshgrid(*[torch.arange(3)] * 3, indexing="ij"), -1).view(-1, 3)
    pos = (ijk.float() + 0.5) / 3 * 1000
    grid = torch.arange(n).view(3, 3, 3)
    pool = SupernodePooling(hidden_dim=12, ndim=3, radius=1.0, max_degree=8)
    out = pool(pos, ijk, supernode_idx=torch.tensor([0, 13, 26]), index_grid=grid)
    assert out.shape == (1, 3, 12) and torch.isfinite(out).all()
