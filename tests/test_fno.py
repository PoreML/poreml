"""Invariants of the FNO 3-D port: the Fourier layer, the grid embedding, the domain padding.

Bit-exact parity against `neuralop` was checked separately with a weight-copying harness
(`util/archive/docs/sketch/fno.md` §5); these tests pin the behaviour that harness cannot, because upstream
is not importable here.
"""

import math

import pytest
import torch

from poreml.models.fno import FNO3D, SpectralConv3d, grid_embedding


def _identity_weight(conv: SpectralConv3d) -> None:
    """Multiply every kept mode by the identity over channels: the layer becomes a band-pass."""
    with torch.no_grad():
        conv.weight.zero_()
        for c in range(conv.in_channels):
            conv.weight[c, c] = 1.0


def test_spectral_conv_keeping_every_mode_is_the_identity():
    n = 8
    conv = SpectralConv3d(2, 2, modes=(n, n, n), bias=False)
    _identity_weight(conv)
    x = torch.randn(2, 2, n, n, n)
    assert torch.allclose(conv(x), x, atol=1e-5)


def test_spectral_conv_truncation_is_a_low_pass_filter():
    n = 16
    z = torch.arange(n, dtype=torch.float32)
    low = torch.cos(2 * math.pi * z / n)  # frequency 1: kept
    high = torch.cos(2 * math.pi * (n // 2 - 1) * z / n)  # frequency 7: dropped
    x = (low + high).view(1, 1, 1, 1, n).expand(1, 1, n, n, n).contiguous()
    conv = SpectralConv3d(1, 1, modes=(n, n, 4), bias=False)  # last axis keeps frequencies 0, 1, 2
    _identity_weight(conv)
    out = conv(x)[0, 0, 0, 0]
    assert torch.allclose(out, low, atol=1e-5)


def test_spectral_conv_is_discretisation_invariant():
    """The same weights on the same continuous field, sampled twice: this is why fft_norm='forward'."""

    def field(n: int) -> torch.Tensor:
        g = torch.arange(n, dtype=torch.float32) / n
        x, y, zz = torch.meshgrid(g, g, g, indexing="ij")
        return (torch.sin(2 * math.pi * x) * torch.cos(2 * math.pi * y) + torch.sin(2 * math.pi * zz)).view(1, 1, n, n, n)

    torch.manual_seed(0)
    conv = SpectralConv3d(1, 1, modes=(4, 4, 4), bias=False)
    coarse = conv(field(16))
    fine = conv(field(32))[:, :, ::2, ::2, ::2]
    assert torch.allclose(coarse, fine, atol=1e-5)


def test_spectral_conv_rejects_modes_larger_than_the_grid():
    conv = SpectralConv3d(1, 1, modes=(8, 8, 8))
    with pytest.raises(ValueError, match="exceed the grid"):
        conv(torch.randn(1, 1, 8, 4, 8))


def test_spectral_conv_weight_covers_the_half_spectrum_of_the_last_axis():
    conv = SpectralConv3d(3, 5, modes=(8, 6, 10))
    assert conv.weight.shape == (3, 5, 8, 6, 6) and conv.weight.is_complex()
    assert conv.bias.shape == (5, 1, 1, 1)


def test_grid_embedding_appends_periodic_coordinates():
    x = torch.zeros(2, 1, 4, 4, 2)
    out = grid_embedding(x)
    assert out.shape == (2, 4, 4, 4, 2)
    assert torch.allclose(out[0, 1, :, 0, 0], torch.tensor([0.0, 0.25, 0.5, 0.75]))  # linspace without the endpoint
    assert torch.allclose(out[0, 3, 0, 0, :], torch.tensor([0.0, 0.5]))


def test_domain_padding_is_symmetric_per_axis_and_round_trips():
    model = FNO3D(in_channels=2, out_channels=1, modes=(4, 4, 4), hidden_channels=4, n_layers=1, domain_padding=0.25)
    x = torch.zeros(1, 1, 16, 24, 32)
    padded, unpad = model._pad(x)
    assert tuple(padded.shape[2:]) == (24, 36, 48)  # each axis grows by 2 * round(0.25 * n), as upstream does
    assert tuple(padded[unpad].shape[2:]) == (16, 24, 32)
    assert torch.equal(padded[unpad], x)


def test_fno3d_keeps_the_spatial_shape_with_and_without_padding():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 12, 16, 8)
    for padding in (0.0, 0.125):
        model = FNO3D(
            in_channels=3, out_channels=2, modes=(4, 4, 4), hidden_channels=6, n_layers=2, domain_padding=padding
        ).eval()
        assert model(x).shape == (1, 2, 12, 16, 8)


def test_fno3d_rejects_a_non_volumetric_input():
    model = FNO3D(in_channels=1, out_channels=1, modes=(2, 2, 2), hidden_channels=4, n_layers=1)
    with pytest.raises(ValueError, match=r"\(B, C, D, H, W\)"):
        model(torch.randn(1, 1, 8, 8))
