"""Visual validation of the topology descriptors (`euler`, `area`) on analytic shapes.

Canonical solids — a ball, two balls, a solid torus, a hollow shell, a block with a
tunnel, a block with a cavity — are voxelised and scored, and every figure panel is
titled with the computed chi next to the expected one, so a wrong number is visible at
a glance. Written to `tests/test_output/topology/`:

  shapes_3d.png         the six solids as 3-D voxel plots, chi computed vs expected
  shapes_3d_extreme.png shapes far from chi = 1 in both directions: a strut lattice,
                        a 16-tunnel waffle, Menger sponges (genus 5 and 81), 27 hollow
                        shells, swiss cheese with 27 cavities
  shapes_2d.png         the 2-D counterparts (blob, diagonal pair, ring, a blob cut by
                        a solid wall, a random smoothed field), chi = components - holes
  random_sweep.png      a smoothed random field thresholded across saturation: chi(Sw)
                        and area(Sw) curves beside 3-D interface views at three thresholds

The numbers are asserted, not just drawn: chi must equal the analytic value on every
shape (6-connectivity: the diagonal pair counts 2), match skimage's `euler_number`
(connectivity=1) across the whole random sweep and on the level-3 Menger sponge
(genus 1409, no figure), and `area` must equal `area_interface + area_contact`
everywhere and land on the Cauchy staircase area `6*pi*r^2` of a digitised ball.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("skimage")
pytest.importorskip("matplotlib")

from skimage.measure import euler_number as sk_euler  # noqa: E402

from poreml.metrics import surface  # noqa: E402
from poreml.metrics.descriptors import area, area_contact, area_interface, euler  # noqa: E402
from tests.plot3d import SECTION_LIGHT, mesh_panel, voxel_panel  # noqa: E402

OUT = Path(__file__).parent / "test_output" / "topology"


def _tensors(blob: np.ndarray, solid: np.ndarray | None = None):
    """`(field, mask)` for a boolean phase array: +1 on the blob, -1 elsewhere."""
    field = torch.where(torch.from_numpy(blob), 1.0, -1.0)[None, None]
    mask = torch.ones_like(field, dtype=torch.bool) if solid is None else torch.from_numpy(~solid)[None, None]
    return field, mask


def _grid(n: int):
    c = (n - 1) / 2
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float64)] * 3, indexing="ij")
    return z - c, y - c, x - c


def ball(r: float, n: int) -> np.ndarray:
    z, y, x = _grid(n)
    return x * x + y * y + z * z < r * r


def shapes_3d() -> dict[str, tuple[np.ndarray, int]]:
    """`{title: (blob, expected chi)}` — chi = components - handles + cavities."""
    n = 18
    z, y, x = _grid(n)
    two = ball(3.5, n)
    two |= np.roll(two, (6, 6, 6), axis=(0, 1, 2))
    torus = np.sqrt((np.sqrt(x * x + y * y) - 5.0) ** 2 + z * z) < 2.2
    shell = ball(7.0, n) & ~ball(4.0, n)
    tunnel = np.zeros((n, n, n), dtype=bool)
    tunnel[3:15, 3:15, 3:15] = True
    tunnel[:, 7:11, 7:11] = False  # a hole straight through: one handle
    cavity = np.zeros((n, n, n), dtype=bool)
    cavity[3:15, 3:15, 3:15] = True
    cavity[7:11, 7:11, 7:11] = False  # a hole sealed inside: one cavity
    return {
        "ball": (ball(6.0, n), 1),
        "two balls": (two, 2),
        "solid torus": (torus, 0),
        "hollow shell": (shell, 2),
        "block + tunnel": (tunnel, 0),
        "block + cavity": (cavity, 2),
    }


def menger(level: int) -> np.ndarray:
    """The level-k Menger sponge on a 3^k grid: a cell dies when, at any level, at least
    two of its three base-3 digits are 1."""
    n = 3**level
    idx = np.arange(n)
    z, y, x = np.meshgrid(idx, idx, idx, indexing="ij")
    keep = np.ones((n, n, n), dtype=bool)
    for k in range(level):
        digit_is_1 = [(a // 3**k) % 3 == 1 for a in (x, y, z)]
        keep &= sum(d.astype(int) for d in digit_is_1) < 2
    return keep


def shapes_3d_extreme() -> dict[str, tuple[np.ndarray, int]]:
    """Shapes that push chi far from the ball's 1 in both directions.

    The strut lattice retracts to its grid graph (chi = V - E = 27 - 54); the waffle is a
    genus-16 handlebody (chi = 1 - 16); the Menger sponges are the genus-5 and genus-81
    solids; the shells and cavities push chi positive (+2 per shell, +1 per cavity).
    """
    on = np.zeros(18, dtype=bool)
    for p in (2, 8, 14):
        on[p : p + 2] = True
    z, y, x = np.meshgrid(on, on, on, indexing="ij")
    lattice = (y & z) | (x & z) | (x & y)
    waffle = np.zeros((18, 18, 18), dtype=bool)
    waffle[6:12, 2:16, 2:16] = True  # a slab, so the tunnels read as punched-through holes
    for py in (4, 7, 10, 13):
        for px in (4, 7, 10, 13):
            waffle[:, py : py + 2, px : px + 2] = False
    c = np.arange(7) - 3.0
    zz, yy, xx = np.meshgrid(c, c, c, indexing="ij")
    r2 = xx * xx + yy * yy + zz * zz
    shell = (r2 < 2.6**2) & ~(r2 < 1.3**2)
    shells = np.zeros((21, 21, 21), dtype=bool)
    cheese = np.zeros((20, 20, 20), dtype=bool)
    cheese[1:19, 1:19, 1:19] = True
    for a in (0, 7, 14):
        for b in (0, 7, 14):
            for c_ in (0, 7, 14):
                shells[a : a + 7, b : b + 7, c_ : c_ + 7] = shell
    for a in (3, 9, 15):  # every cavity needs a sealing wall on all six sides
        for b in (3, 9, 15):
            for c_ in (3, 9, 15):
                cheese[a : a + 2, b : b + 2, c_ : c_ + 2] = False
    return {
        "strut lattice: $V - E$ = 27 $-$ 54": (lattice, -27),
        "waffle, 16 tunnels": (waffle, -15),
        "Menger sponge level 1 (genus 5)": (menger(1), -4),
        "Menger sponge level 2 (genus 81)": (menger(2), -80),
        "27 hollow shells": (shells, 54),
        "swiss cheese, 27 cavities": (cheese, 28),
    }


def shapes_2d() -> dict[str, tuple[np.ndarray, np.ndarray | None, int]]:
    """`{title: (blob, solid, expected chi)}` — chi = components - holes."""
    blob = np.zeros((9, 9), dtype=bool)
    blob[2:7, 2:7] = True
    diag = np.zeros((9, 9), dtype=bool)
    diag[3, 3] = diag[4, 4] = True
    ring = np.zeros((9, 9), dtype=bool)
    ring[2:7, 2:7] = True
    ring[3:6, 3:6] = False
    walled = np.zeros((9, 9), dtype=bool)
    walled[2:7, 1:8] = True
    solid = np.zeros((9, 9), dtype=bool)
    solid[:, 4] = True  # a solid wall: the blob it cuts is two components
    rng = np.random.default_rng(3)
    from scipy.ndimage import gaussian_filter

    rand = gaussian_filter(rng.standard_normal((48, 48)), 2.5) > 0.0
    return {
        "blob": (blob, None, 1),
        "diagonal pair\n(6-connectivity: 2, not 1)": (diag, None, 2),
        "ring": (ring, None, 0),
        "blob cut by a solid wall": (walled, solid, 2),
        "random smoothed field": (rand, None, int(sk_euler(rand, connectivity=1))),
    }


def _chi(blob: np.ndarray, solid: np.ndarray | None = None) -> int:
    field, mask = _tensors(blob, solid)
    out = euler(field, mask).item()
    assert out == int(out)
    return int(out)


def test_shape_gallery_3d_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 8.5))
    for i, (title, (blob, expected)) in enumerate(shapes_3d().items()):
        got = _chi(blob)
        show = blob.copy()
        cut = "shell" in title or "cavity" in title
        if cut:  # display-only half-cut: the hollow is the point
            show[:, :, blob.shape[2] // 2 :] = False
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        voxel_panel(ax, show, light=SECTION_LIGHT if cut else None)
        if cut:
            ax.view_init(elev=22, azim=-20)  # face the section plane
        ax.set_title(f"{title}\n$\\chi$ = {got} (expect {expected})", fontsize=11)
        assert got == expected, f"{title}: chi {got} != {expected}"
    fig.suptitle("euler descriptor on canonical solids: $\\chi$ = components $-$ handles + cavities", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "shapes_3d.png", dpi=130)
    plt.close(fig)
    assert (OUT / "shapes_3d.png").exists()


def test_extreme_shape_gallery_3d_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 8.5))
    for i, (title, (blob, expected)) in enumerate(shapes_3d_extreme().items()):
        got = _chi(blob)
        show = blob.copy()
        cut = "shell" in title or "cheese" in title
        if cut:  # display-only half-cut: shells become rings, cavities open up
            show[:, :, blob.shape[2] // 2 :] = False
        if min(blob.shape) < 12:  # refine the coarse Menger levels for display (chi is scale-free)
            show = np.kron(show, np.ones((2, 2, 2), dtype=bool))
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        voxel_panel(ax, show, linewidth=0.12, light=SECTION_LIGHT if cut else None)
        if cut:
            ax.view_init(elev=22, azim=-20)  # face the section plane
        elif "waffle" in title:
            ax.view_init(elev=72, azim=-60)  # look down the tunnels to the page behind
        ax.set_title(f"{title}\n$\\chi$ = {got} (expect {expected})", fontsize=11)
        assert got == expected, f"{title}: chi {got} != {expected}"
    fig.suptitle(
        "euler descriptor at the extremes: many handles ($\\chi \\ll 0$), many shells and cavities ($\\chi \\gg 0$)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "shapes_3d_extreme.png", dpi=130)
    plt.close(fig)
    assert (OUT / "shapes_3d_extreme.png").exists()


def test_menger_level_3_reaches_genus_1409():
    # 27^3 voxels, 8000 cubes, genus 1409: chi = 1 - 1409, agreeing with skimage exactly.
    blob = menger(3)
    assert _chi(blob) == -1408 == int(sk_euler(blob, connectivity=1))


def test_shape_gallery_2d_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2))
    palette = ListedColormap(["#f4f2ee", "#7a9cc4", "#555555"])  # other fluid, phase, solid
    for ax, (title, (blob, solid, expected)) in zip(axes, shapes_2d().items(), strict=True):
        got = _chi(blob[None] if blob.ndim == 2 else blob, None if solid is None else solid[None])
        image = blob.astype(int)
        if solid is not None:
            image[solid] = 2
        ax.imshow(image, cmap=palette, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"{title}\n$\\chi$ = {got} (expect {expected})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        assert got == expected, f"{title}: chi {got} != {expected}"
    fig.suptitle("euler descriptor in 2-D: $\\chi$ = components $-$ holes (solid voxels join neither phase)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "shapes_2d.png", dpi=130)
    plt.close(fig)
    assert (OUT / "shapes_2d.png").exists()


def test_random_field_sweep_figure():
    """A smoothed random field thresholded from dry to full: chi and area vs saturation,
    the euler curve pinned to skimage at every threshold and area to interface + contact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    smooth = gaussian_filter(rng.standard_normal((32, 32, 32)), 2.0)
    smooth /= smooth.std()
    thresholds = np.linspace(1.5, -1.5, 25)
    fields = torch.stack([torch.from_numpy((smooth - t).astype(np.float32)) for t in thresholds])[:, None]
    mask = torch.ones_like(fields, dtype=torch.bool)

    chi = euler(fields, mask)
    total = area(fields, mask)
    sw = fields.gt(0).float().mean(dim=(1, 2, 3, 4))
    for b, t in enumerate(thresholds):
        assert chi[b].item() == float(sk_euler(smooth > t, connectivity=1)), f"threshold {t}"
    assert torch.equal(total, area_interface(fields, mask) + area_contact(fields, mask))

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.1, 1))
    picks = [4, 12, 20]
    for col, b in enumerate(picks):
        ax = fig.add_subplot(grid[0, col], projection="3d")
        verts, faces = surface.marching_cubes(fields[b, 0].numpy())
        mesh_panel(ax, verts, faces, color="#8ab0d6", cube_aspect=True)
        ax.set_title(f"$S_w$ = {sw[b]:.2f}: $\\chi$ = {int(chi[b])}, area = {int(total[b])}", fontsize=11)
    for col, (values, label) in enumerate([(chi, "Euler characteristic $\\chi$"), (total, "surface area [faces]")]):
        ax = fig.add_subplot(grid[1, col])
        ax.plot(sw.numpy(), values.numpy(), "o-", ms=4)
        for b in picks:
            ax.axvline(sw[b].item(), color="k", ls=":", lw=0.8, alpha=0.6)
        ax.set_xlabel("phase saturation")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(grid[1, 2])
    ax.plot(chi.numpy(), total.numpy(), "o-", ms=4, alpha=0.7)
    ax.set_xlabel("$\\chi$")
    ax.set_ylabel("surface area [faces]")
    ax.grid(alpha=0.3)
    fig.suptitle(
        "Smoothed random field thresholded across saturation: blobs appear ($\\chi$ up), merge and tunnel ($\\chi$ down), "
        "seal cavities ($\\chi$ back up)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "random_sweep.png", dpi=130)
    plt.close(fig)
    assert (OUT / "random_sweep.png").exists()


def test_ball_area_lands_on_the_cauchy_staircase_area():
    # Face counting doubles each of the three axis projections: 6 * pi * r^2 for a ball
    # (the staircase 3/2 of the smooth 4 * pi * r^2), within ~1% by r = 10.
    for r in (10.0, 16.0):
        field, mask = _tensors(ball(r, int(2 * r + 6)))
        assert area(field, mask).item() == pytest.approx(6 * math.pi * r * r, rel=0.02)
        assert area(field, mask).item() == area_interface(field, mask).item()  # no solid anywhere
