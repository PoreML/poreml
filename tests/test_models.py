import pytest
import torch

from poreml.config import ModelConfig
from poreml.models import Persistence, build_model
from poreml.registry import MODELS


def test_persistence_is_registered():
    assert "persistence" in MODELS


def test_build_model_returns_the_registered_class():
    model = build_model(ModelConfig(name="persistence"), in_channels=3)

    assert isinstance(model, Persistence)


def test_build_model_with_unknown_name_lists_options():
    with pytest.raises(KeyError, match="persistence"):
        build_model(ModelConfig(name="nope"), in_channels=3)


def test_persistence_returns_the_most_recent_input_frame():
    model = build_model(ModelConfig(name="persistence"), in_channels=3)
    solid = torch.zeros(1, 1, 2, 2, 2)
    older = torch.full((1, 1, 2, 2, 2), -1.0)
    latest = torch.full((1, 1, 2, 2, 2), 0.5)
    x = torch.cat([solid, older, latest], dim=1)

    out = model(x)

    assert out.shape == (1, 1, 2, 2, 2)
    assert torch.equal(out, latest)


def test_persistence_honours_a_non_default_recent_channel():
    # A task free to stack channels differently configures the baseline through params
    # rather than silently getting the solid mask back as a prediction.
    model = build_model(ModelConfig(name="persistence", params={"recent_channel": 1}), in_channels=3)
    solid = torch.zeros(1, 1, 2, 2, 2)
    older = torch.full((1, 1, 2, 2, 2), -1.0)
    latest = torch.full((1, 1, 2, 2, 2), 0.5)
    x = torch.cat([solid, older, latest], dim=1)

    out = model(x)

    assert torch.equal(out, older)


def test_persistence_rejects_a_recent_channel_outside_the_input():
    with pytest.raises(ValueError, match="recent_channel"):
        build_model(ModelConfig(name="persistence", params={"recent_channel": 3}), in_channels=3)


def test_persistence_has_no_trainable_parameters():
    model = build_model(ModelConfig(name="persistence"), in_channels=2)

    assert list(model.parameters()) == []


def test_build_model_passes_params_through():
    # Unknown params must not silently vanish: persistence accepts and ignores none.
    with pytest.raises(TypeError):
        build_model(ModelConfig(name="persistence", params={"width": 8}), in_channels=2)


# --- unet3d -------------------------------------------------------------------------


def test_unet3d_is_registered_and_maps_inputs_to_one_channel():
    model = build_model(ModelConfig(name="unet3d", params={"base": 4, "depth": 2}), in_channels=3)

    out = model(torch.randn(2, 3, 16, 16, 16))

    assert out.shape == (2, 1, 16, 16, 16)


def test_unet3d_residual_starts_as_persistence():
    # The head is zero-initialised, so an untrained residual UNet returns the most recent
    # frame exactly: training starts from the floor rather than from noise.
    model = build_model(ModelConfig(name="unet3d", params={"base": 4, "depth": 2}), in_channels=2)
    x = torch.randn(1, 2, 8, 8, 8)

    assert torch.equal(model(x), x[:, -1:])


def test_unet3d_without_residual_starts_at_zero():
    model = build_model(ModelConfig(name="unet3d", params={"base": 4, "depth": 2, "residual": False}), in_channels=2)

    assert torch.equal(model(torch.randn(1, 2, 8, 8, 8)), torch.zeros(1, 1, 8, 8, 8))


def test_unet3d_pads_odd_spatial_sizes_and_crops_back():
    # An underfill flipchip run is 23 x 482 x 472: the gap axis is odd, and depth 1 halves once.
    # The forward replicate-pads each axis up to the next multiple of 2**depth and crops the
    # output back, as P3D does — the loss and the metrics never see the padded slab.
    model = build_model(ModelConfig(name="unet3d", params={"base": 4, "depth": 1}), in_channels=2)
    x = torch.randn(1, 2, 23, 10, 12)

    out = model(x)

    assert out.shape == (1, 1, 23, 10, 12)
    deep = build_model(ModelConfig(name="unet3d", params={"base": 4, "depth": 3}), in_channels=2)
    assert deep(torch.randn(1, 2, 12, 12, 12)).shape == (1, 1, 12, 12, 12)


def test_unet3d_even_shapes_are_not_padded():
    # pad 0 on every axis: the pre-fix path bit for bit, so no 128-class checkpoint changes.
    torch.manual_seed(0)
    model = build_model(ModelConfig(name="unet3d", params={"base": 4, "depth": 2}), in_channels=2)
    x = torch.randn(1, 2, 8, 8, 8)
    with torch.no_grad():
        out = model(x)
        h = model.stem(x)
        skips = []
        for down, enc in zip(model.down, model.enc, strict=True):
            skips.append(h)
            h = enc(down(h))
        for up, dec in zip(model.up, model.dec, strict=True):
            h = dec(torch.cat([up(h), skips.pop()], dim=1))
        ref = model.head(h) + x[:, model.recent]
    assert torch.equal(out, ref)


def test_unet3d_parameter_count_grows_with_base():
    from poreml.models import count_parameters

    small = build_model(ModelConfig(name="unet3d", params={"base": 4, "depth": 2}), in_channels=2)
    large = build_model(ModelConfig(name="unet3d", params={"base": 8, "depth": 2}), in_channels=2)

    assert 0 < count_parameters(small) < count_parameters(large)


def test_unet3d_trains_a_step_and_changes_its_output():
    model = build_model(ModelConfig(name="unet3d", params={"base": 4, "depth": 2}), in_channels=2)
    x = torch.randn(2, 2, 8, 8, 8)
    target = torch.randn(2, 1, 8, 8, 8)
    before = model(x).detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

    assert not torch.equal(model(x), before)


# --- out_channels --------------------------------------------------------------------


def test_persistence_returns_the_last_out_channels_channels():
    model = build_model(ModelConfig(name="persistence"), in_channels=1 + 2 * 3, out_channels=3)
    x = torch.arange(7.0).view(1, 7, 1, 1, 1).expand(1, 7, 2, 2, 2)

    out = model(x)

    assert out.shape == (1, 3, 2, 2, 2)
    assert out[0, :, 0, 0, 0].tolist() == [4.0, 5.0, 6.0]


def test_persistence_rejects_out_channels_that_do_not_fit_before_recent_channel():
    with pytest.raises(ValueError, match="out_channels"):
        build_model(ModelConfig(name="persistence", params={"recent_channel": 1}), in_channels=4, out_channels=3)


def test_unet3d_maps_inputs_to_out_channels_and_starts_as_persistence_on_every_field():
    model = build_model(ModelConfig(name="unet3d", params={"base": 4, "depth": 1}), in_channels=1 + 3, out_channels=3)
    x = torch.randn(1, 4, 4, 4, 4)

    out = model(x)

    assert out.shape == (1, 3, 4, 4, 4)
    assert torch.allclose(out, x[:, 1:4])


def test_persistence_returns_the_recent_channels_of_a_point_batch():
    from poreml.points import Points

    model = build_model(ModelConfig(name="persistence"), in_channels=4, out_channels=2)
    feats = torch.arange(3 * 4, dtype=torch.float32).view(3, 4)
    pts = Points(
        pos=torch.rand(3, 3),
        feats=feats,
        index=torch.arange(3),
        wall=torch.zeros(3, dtype=torch.bool),
        ptr=torch.tensor([0, 3]),
        shape=(1, 1, 3),
        fill=torch.tensor([-1.0, 0.0]),
    )
    assert torch.equal(model(pts), feats[:, 2:])


def test_models_declare_their_representation():
    from poreml.models import representation_of

    assert representation_of(ModelConfig(name="persistence")) == "voxel"
    assert representation_of(ModelConfig(name="unet3d")) == "voxel"
    assert representation_of(ModelConfig(name="fno3d")) == "voxel"
    assert representation_of(ModelConfig(name="p3d")) == "voxel"


def _points(ns=(7, 5), c=4, shape=(2, 2, 3)):
    from poreml.points import Points, collate

    samples = []
    for n in ns:
        wall = torch.zeros(n, dtype=torch.bool)
        wall[: n // 2] = True
        pts = Points(
            pos=torch.rand(n, 3) * 0.9 + 0.05,
            feats=torch.randn(n, c),
            index=torch.arange(n),
            wall=wall,
            ptr=torch.tensor([0, n]),
            shape=shape,
            fill=torch.tensor([-1.0]),
        )
        samples.append((pts, torch.zeros(n, 1), torch.ones(n, 1, dtype=torch.bool), {"run": 0, "t": 0}))
    return collate(samples)[0]


def test_transolver_maps_points_to_out_channels_and_starts_as_persistence():
    torch.manual_seed(0)
    pts = _points(c=4)
    params = {"dim": 16, "depth": 2, "heads": 2, "slices": 4}
    model = build_model(ModelConfig(name="transolver", params=params), in_channels=4, out_channels=2)
    assert model.representation == "points"
    model.eval()
    out = model(pts)
    assert out.shape == (12, 2)
    assert torch.allclose(out, pts.feats[:, 2:])  # zero-init head + residual = persistence
    model.train()
    out = model(pts)
    assert out.shape == (12, 2) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if "head" not in n)


def test_transolver_without_residual_is_not_persistence():
    torch.manual_seed(0)
    pts = _points(c=4)
    model = build_model(
        ModelConfig(name="transolver", params={"dim": 16, "depth": 1, "heads": 2, "slices": 4, "residual": False}),
        in_channels=4,
        out_channels=2,
    ).eval()
    assert not torch.allclose(model(pts), pts.feats[:, 2:])


def test_abupt_maps_points_to_out_channels_and_starts_as_persistence():
    torch.manual_seed(0)
    pts = _points(ns=(12, 9), c=4, shape=(3, 3, 3))  # half the points are wall in each sample
    pts.index = torch.cat([torch.arange(12), torch.arange(9)])  # unique voxel per point within its sample
    params = {
        "dim": 12,
        "num_heads": 2,
        "blocks": "psc",
        "num_surface_blocks": 1,
        "num_volume_blocks": 1,
        "num_supernodes": 3,
        "supernode_radius": 1.5,
        "num_surface_anchors": 3,
        "num_volume_anchors": 4,
    }
    model = build_model(ModelConfig(name="abupt", params=params), in_channels=4, out_channels=2)
    assert model.representation == "points"
    model.eval()
    out = model(pts)
    assert out.shape == (21, 2)
    assert torch.allclose(out, pts.feats[:, 2:], atol=1e-6)  # zero-init decoders + residual
    assert torch.equal(model(pts), out)  # eval anchors are seeded: deterministic
    model.train()
    out = model(pts)
    assert out.shape == (21, 2) and torch.isfinite(out).all()
    out.sum().backward()


def test_abupt_needs_both_wall_and_bulk_points():
    pts = _points(ns=(6,), c=4)
    pts.wall[:] = True
    model = build_model(
        ModelConfig(
            name="abupt", params={"dim": 12, "num_heads": 2, "blocks": "p", "num_surface_blocks": 1, "num_volume_blocks": 1}
        ),
        in_channels=4,
        out_channels=1,
    )
    with pytest.raises(ValueError, match="wall"):
        model(pts)


def _fno(**params):
    defaults = {"modes": [4, 4, 4], "hidden_channels": 6, "n_layers": 2}
    return ModelConfig(name="fno3d", params={**defaults, **params})


def test_fno3d_maps_inputs_to_out_channels_and_starts_as_persistence_on_every_field():
    torch.manual_seed(0)
    model = build_model(_fno(), in_channels=1 + 3, out_channels=3).eval()
    x = torch.randn(2, 4, 8, 8, 8)

    out = model(x)

    assert out.shape == (2, 3, 8, 8, 8)
    assert torch.allclose(out, x[:, 1:4], atol=1e-6)  # zero-init projection + residual = persistence


def test_fno3d_without_residual_is_not_persistence():
    torch.manual_seed(0)
    model = build_model(_fno(residual=False), in_channels=4, out_channels=3).eval()
    x = torch.randn(1, 4, 8, 8, 8)

    assert not torch.allclose(model(x), x[:, 1:4], atol=1e-6)


def test_fno3d_trains_a_step_and_reaches_every_parameter():
    torch.manual_seed(0)
    model = build_model(_fno(), in_channels=4, out_channels=3)
    x = torch.randn(1, 4, 8, 8, 8)
    target = torch.randn(1, 3, 8, 8, 8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = model(x)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    opt.step()

    assert not torch.allclose(before, model(x), atol=1e-6)


def test_fno3d_parameter_count_grows_with_modes():
    from poreml.models import count_parameters

    small = build_model(_fno(modes=[4, 4, 4]), in_channels=4, out_channels=1)
    large = build_model(_fno(modes=[8, 8, 8]), in_channels=4, out_channels=1)

    assert 0 < count_parameters(small) < count_parameters(large)


def test_fno3d_honours_a_non_default_recent_channel():
    torch.manual_seed(0)
    model = build_model(_fno(recent_channel=2), in_channels=5, out_channels=2).eval()
    x = torch.randn(1, 5, 8, 8, 8)

    assert torch.allclose(model(x), x[:, 1:3], atol=1e-6)


def test_fno3d_rejects_impossible_configurations():
    with pytest.raises(ValueError, match="n_layers"):
        build_model(_fno(n_layers=0), in_channels=4, out_channels=1)
    with pytest.raises(ValueError, match="domain_padding"):
        build_model(_fno(domain_padding=0.5), in_channels=4, out_channels=1)
    with pytest.raises(ValueError, match="modes must be three"):
        build_model(_fno(modes=[4, 4]), in_channels=4, out_channels=1)


def test_models_declare_their_precision_and_the_config_can_override():
    """`precision_of` = the config's `model.precision` if given, else the class default:
    bf16 for AB-UPT (its upstream trains in mixed precision; 5x faster here), the project
    default tf32 elsewhere (2026-09-10); `fp32` reproduces an earlier run."""
    from poreml.models import precision_of

    assert precision_of(ModelConfig(name="unet3d")) == "tf32"
    assert precision_of(ModelConfig(name="transolver")) == "tf32"
    assert precision_of(ModelConfig(name="fno3d", precision="tf32")) == "tf32"
    assert precision_of(ModelConfig(name="abupt")) == "bf16"
    assert precision_of(ModelConfig(name="abupt", precision="fp32")) == "fp32"
    assert precision_of(ModelConfig(name="unet3d", precision="bf16")) == "bf16"


def test_abupt_runs_under_bf16_autocast_and_pins_attention_off_cudnn():
    """Under autocast the decoders emit bf16; the output must come back in the input's dtype.
    cuDNN attention rebuilds a plan per new token count (~2.4 s each), and every AB-UPT
    window has a new wall/bulk split, so the model pins SDPA to the plan-free backends."""
    import torch
    from torch.nn.attention import SDPBackend

    from poreml.models import abupt as abupt_module
    from poreml.points import Points

    assert SDPBackend.CUDNN_ATTENTION not in abupt_module.SDPA_BACKENDS
    assert SDPBackend.FLASH_ATTENTION in abupt_module.SDPA_BACKENDS
    model = build_model(
        ModelConfig(
            name="abupt",
            params={"dim": 16, "num_heads": 2, "num_supernodes": 8, "num_surface_anchors": 8, "num_volume_anchors": 8},
        ),
        in_channels=3,
        out_channels=1,
    )
    n = 64
    pos = torch.rand(n, 3)
    wall = torch.zeros(n, dtype=torch.bool)
    wall[:20] = True
    pts = Points(
        pos=pos,
        feats=torch.randn(n, 3),
        index=torch.arange(n),
        wall=wall,
        ptr=torch.tensor([0, n]),
        shape=(4, 4, 4),
        fill=torch.zeros(1),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(pts)
    assert out.dtype == torch.float32 and out.shape == (n, 1) and torch.isfinite(out).all()
