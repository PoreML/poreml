"""Interface curvature against analytic surfaces.

Voxelised (binary, staircase — what a segmented image looks like) spheres and catenoids
at several resolutions go through the whole pipeline (marching cubes → Taubin → quadric
fit); per-vertex mean and Gaussian curvature are plotted against the z-coordinate beside
the analytic curves — the Figure-2 layout of the CT-curvature literature — into
`tests/test_output/curvature/`, with 3-D views of the smoothed meshes coloured by
curvature beside them, and the finest resolution must land within tolerance.

Sphere: H = 1/r, K = 1/r². Catenoid x² + y² = c² cosh²(z/c): H = 0, K = −1/(c² cosh⁴(z/c)).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("igl")
pytest.importorskip("skimage")

from poreml.metrics import curvature as cv  # noqa: E402
from poreml.metrics import surface  # noqa: E402
from poreml.metrics.descriptors import curvature_gauss, curvature_hist, curvature_mean  # noqa: E402
from poreml.registry import DESCRIPTORS  # noqa: E402

OUT = Path(__file__).parent / "test_output" / "curvature"
SPHERE_RADII = (8, 16, 32)
CATENOID_NECKS = (6, 12, 24)


def _grid(shape):
    return np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij")


def sphere_field(r: float) -> tuple[np.ndarray, float]:
    """`(phi, centre)`: +1 inside a sphere of radius `r`, -1 outside."""
    n = int(2 * r + 12)
    c = (n - 1) / 2
    z, y, x = _grid((n, n, n))
    phi = r - np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    return np.sign(phi), c


def catenoid_field(c: float) -> tuple[np.ndarray, float]:
    """`(phi, z_centre)`: +1 inside the catenoid of neck radius `c`, spanning z ∈ [−1.2c, 1.2c]."""
    half = 1.2 * c
    rmax = c * np.cosh(half / c)
    nz = int(2 * half + 8)
    nxy = int(2 * rmax + 8)
    zc = (nz - 1) / 2
    xc = (nxy - 1) / 2
    z, y, x = _grid((nz, nxy, nxy))
    phi = c * np.cosh((z - zc) / c) - np.sqrt((x - xc) ** 2 + (y - xc) ** 2)
    return np.sign(phi), zc


def _stats_vs_z(z: np.ndarray, values: np.ndarray, nbins: int = 24):
    edges = np.linspace(z.min(), z.max(), nbins + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, nbins - 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    med = np.array([np.nanmedian(values[idx == i]) if (idx == i).any() else np.nan for i in range(nbins)])
    lo = np.array([np.nanpercentile(values[idx == i], 25) if (idx == i).any() else np.nan for i in range(nbins)])
    hi = np.array([np.nanpercentile(values[idx == i], 75) if (idx == i).any() else np.nan for i in range(nbins)])
    return centres, med, lo, hi


# Normalised analytic profiles vs normalised z: sphere H·r = 1, K·r² = 1; catenoid H·c = 0,
# K·c² = -1 / cosh⁴(z/c).
ANALYTIC = {
    "sphere": {"H": lambda z: np.ones_like(z), "K": lambda z: np.ones_like(z)},
    "catenoid": {"H": lambda z: np.zeros_like(z), "K": lambda z: -1.0 / np.cosh(z) ** 4},
}


def _plot(results: dict) -> Path:
    """2×2 figure: (a) sphere H, (b) sphere K, (c) catenoid H, (d) catenoid K, one curve per resolution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("sphere", "H", "(a) sphere: mean curvature H·r"),
        ("sphere", "K", "(b) sphere: Gaussian curvature K·r²"),
        ("catenoid", "H", "(c) catenoid: mean curvature H·c"),
        ("catenoid", "K", "(d) catenoid: Gaussian curvature K·c²"),
    ]
    for ax, (shape, q, title) in zip(axes.ravel(), panels, strict=True):
        for scale, (zn, values) in results[shape][q].items():
            centres, med, lo, hi = _stats_vs_z(zn, values)
            (line,) = ax.plot(centres, med, "o-", ms=3, label=f"computed, {'r' if shape == 'sphere' else 'c'}={scale}")
            ax.fill_between(centres, lo, hi, color=line.get_color(), alpha=0.15)
        zz = np.linspace(-1.05, 1.05, 200)
        ax.plot(zz, ANALYTIC[shape][q](zz), "k--", label="analytic")
        ax.set_title(title)
        ax.set_xlabel("z / r" if shape == "sphere" else "z / c")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Computed vs analytic curvature (binary voxel input)")
    fig.tight_layout()
    path = OUT / "curvature_vs_z.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_case(name: str, scale: float, raw: np.ndarray, res, z: np.ndarray, h: np.ndarray, k: np.ndarray) -> Path:
    """One figure per (shape, resolution): 3-D staircase / H / K, then H(z), K(z) and distributions vs analytic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tests.plot3d import mesh_panel, value_mesh_panel

    OUT.mkdir(parents=True, exist_ok=True)
    unit = "r" if name == "sphere" else "c"
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.15, 1))
    columns = [
        (raw, None, f"{name} {unit}={scale}: marching cubes (staircase)", None),
        (res.vertices, res.mean * scale, f"smoothed, H·{unit}", "coolwarm"),
        (res.vertices, res.gaussian * scale**2, f"smoothed, K·{unit}²", "viridis"),
    ]
    for col, (verts, values, title, cmap) in enumerate(columns):
        ax = fig.add_subplot(grid[0, col], projection="3d")
        if values is None:
            mesh_panel(ax, verts, res.faces)
        else:
            face_values = np.nanmean(values[res.faces], axis=1)
            lo, hi = np.nanpercentile(face_values, [2, 98])
            mappable = value_mesh_panel(ax, verts, res.faces, face_values, cmap=cmap, clim=(lo, hi))
            fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)
        ax.set_title(title)

    zz = np.linspace(-1.05, 1.05, 200)
    for col, (q, values, label) in enumerate([("H", h, f"H·{unit}"), ("K", k, f"K·{unit}²")]):
        ax = fig.add_subplot(grid[1, col])
        centres, med, lo, hi = _stats_vs_z(z, values)
        ax.plot(centres, med, "o-", ms=3, label="computed (median, IQR)")
        ax.fill_between(centres, lo, hi, alpha=0.2)
        ax.plot(zz, ANALYTIC[name][q](zz), "k--", label="analytic")
        ax.set_xlabel(f"z / {unit}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Distributions: the analytic value is a single number for the sphere; for the
    # catenoid H is 0 and K is spread over [-1, -1/cosh⁴(1)] along the surface, so the
    # analytic K distribution is drawn from the same vertices' z.
    ax = fig.add_subplot(grid[1, 2])
    finite = np.isfinite(h) & np.isfinite(k)
    ax.hist(h[finite], bins=60, density=True, alpha=0.6, label=f"H·{unit}")
    ax.hist(k[finite], bins=60, density=True, alpha=0.6, label=f"K·{unit}²")
    ax.axvline(float(ANALYTIC[name]["H"](np.zeros(1))[0]), color="C0", ls="--", label="analytic H")
    if name == "sphere":
        ax.axvline(1.0, color="C1", ls="--", label="analytic K")
    else:
        ax.hist(ANALYTIC[name]["K"](z[finite]), bins=60, density=True, histtype="step", color="C1", ls="--", label="analytic K")
    ax.set_xlabel("normalised curvature")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{name}, {unit}={scale} voxels: {len(res.vertices)} vertices, {int(res.valid.sum())} scored")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.07, wspace=0.25, hspace=0.15)
    path = OUT / f"{name}_{unit}{scale}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _run_sphere(r: float):
    phi, c = sphere_field(r)
    res = cv.interface_curvature(phi, np.zeros_like(phi, dtype=bool))
    z = (res.vertices[:, 0] - c) / r
    return res, z, res.mean * r, res.gaussian * r * r


def _run_catenoid(c: float):
    phi, zc = catenoid_field(c)
    res = cv.interface_curvature(phi, np.zeros_like(phi, dtype=bool))
    z = (res.vertices[:, 0] - zc) / c
    # The surface is open at the top and bottom of the box; the quadric fit has no
    # neighbours beyond the rim, so keep the interior |z| < c.
    keep = np.abs(z) < 1.0
    return res, z[keep], res.mean[keep] * c, res.gaussian[keep] * c * c


def test_analytic_surfaces_figure():
    results = {"sphere": {"H": {}, "K": {}}, "catenoid": {"H": {}, "K": {}}}
    for r in SPHERE_RADII:
        res, z, h, k = _run_sphere(r)
        results["sphere"]["H"][r] = (z, h)
        results["sphere"]["K"][r] = (z, k)
        assert _plot_case("sphere", r, surface.marching_cubes(sphere_field(r)[0])[0], res, z, h, k).exists()
    for c in CATENOID_NECKS:
        res, z, h, k = _run_catenoid(c)
        results["catenoid"]["H"][c] = (z, h)
        results["catenoid"]["K"][c] = (z, k)
        assert _plot_case("catenoid", c, surface.marching_cubes(catenoid_field(c)[0])[0], res, z, h, k).exists()
    assert _plot(results).exists()

    # Finest resolution: sphere within 5% (median), 75% of vertices within 20% — the CT
    # best-practice figure is ±30% locally above 6 voxels of radius; catenoid H ≈ 0 and
    # K within 15% of the analytic profile at the neck.
    r, c = SPHERE_RADII[-1], CATENOID_NECKS[-1]
    sphere_h, sphere_k = results["sphere"]["H"][r][1], results["sphere"]["K"][r][1]
    assert abs(np.nanmedian(sphere_h) - 1.0) < 0.05
    assert abs(np.nanmedian(sphere_k) - 1.0) < 0.10
    assert np.nanpercentile(np.abs(sphere_h - 1.0), 75) < 0.20
    z, cat_h = results["catenoid"]["H"][c]
    cat_k = results["catenoid"]["K"][c][1]
    neck = np.abs(z) < 0.25
    assert abs(np.nanmedian(cat_h[neck])) < 0.10  # H·c: analytic 0 (K·c² = -1 there)
    assert abs(np.nanmedian(cat_k[neck]) + 1.0) < 0.15


def test_droplet_of_each_phase_has_positive_mean_curvature():
    r = 12
    phi, _ = sphere_field(r)
    field = torch.tensor(phi, dtype=torch.float32)[None, None]
    mask = torch.ones_like(field, dtype=torch.bool)
    h_nw = curvature_mean(field, mask).item()
    assert h_nw == pytest.approx(1 / r, rel=0.05)
    assert curvature_gauss(field, mask).item() == pytest.approx(1 / r**2, rel=0.10)
    # The same geometry with the sign flipped is a wetting droplet: positive for phase="w",
    # negative (a bubble of the other phase) for phase="nw".
    h_w = curvature_mean(-field, mask, phase="w").item()
    assert h_w == pytest.approx(h_nw, rel=1e-6)
    assert curvature_mean(-field, mask).item() == pytest.approx(-h_nw, rel=1e-6)


def test_no_interface_is_nan_and_batch_is_per_sample():
    r = 10
    phi, _ = sphere_field(r)
    droplet = torch.tensor(phi, dtype=torch.float32)[None, None]
    field = torch.cat([droplet, torch.ones_like(droplet)])  # sample 1 is all non-wetting
    mask = torch.ones_like(field, dtype=torch.bool)
    out = curvature_mean(field, mask)
    assert out.shape == (2,)
    assert out[0].item() == pytest.approx(1 / r, rel=0.05)
    assert torch.isnan(out[1])


def test_vertices_touching_solid_are_excluded():
    # A hemispherical cap of nw sitting on a solid floor: the level set closes on the
    # floor (flagged near_solid); the free surface still reads 1/r.
    r = 12
    phi, c = sphere_field(r)
    solid = np.zeros_like(phi, dtype=bool)
    solid[: int(c)] = True  # solid below the equator along z
    res = cv.interface_curvature(phi, solid)
    assert res.near_solid.any() and res.fluid_vertices.any()
    assert res.fluid_solid_faces.any() and res.contact_vertices.any()
    assert res.vertices[res.near_solid, 0].max() <= c + 1  # flags live at the floor
    assert res.vertices[res.contact_vertices, 0].max() <= c + 1  # so does the contact line
    assert not res.valid[res.near_solid].any() and res.valid.sum() < res.fluid_vertices.sum()  # margin applied
    assert np.nanmedian(res.mean[res.valid]) == pytest.approx(1 / r, rel=0.05)
    field = torch.tensor(phi, dtype=torch.float32)[None, None]
    mask = torch.tensor(~solid)[None, None]
    assert curvature_mean(field, mask).item() == pytest.approx(1 / r, rel=0.15)


def test_registered_with_params():
    for name in ("curvature_mean", "curvature_gauss"):
        fn = DESCRIPTORS.get(name)
        assert fn.kind == "scalar"
        assert fn.params == {"phase"}
    assert DESCRIPTORS.get("curvature_hist").kind == "distribution"
    assert DESCRIPTORS.get("curvature_hist").params == {"phase", "bins", "range"}


def test_curvature_hist_peaks_at_one_over_r_and_scores_with_w1():
    from poreml.metrics.errors import w1, w1_norm

    small, large = 8, 16
    field = torch.stack(
        [
            torch.tensor(np.pad(sphere_field(small)[0], 8, constant_values=-1), dtype=torch.float32),
            torch.tensor(sphere_field(large)[0], dtype=torch.float32),
        ]
    )[:, None]
    mask = torch.ones_like(field, dtype=torch.bool)
    hist = curvature_hist(field, mask, bins=25, range=(0.0, 0.25))  # 0.01 per bin
    assert hist.shape == (2, 25)
    assert hist.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
    assert hist[0].argmax().item() == int(1 / small / 0.01)  # 0.125 -> bin 12
    assert hist[1].argmax().item() == int(1 / large / 0.01)  # 0.0625 -> bin 6
    # W1 between the two spheres' distributions is about the difference of their curvatures, in 1/voxel.
    assert w1(hist[:1], hist[1:], bins=25, range=(0.0, 0.25)).item() == pytest.approx(1 / small - 1 / large, abs=0.01)
    assert w1_norm(hist[:1], hist[1:]).item() > 1.0  # many spreads apart
    assert w1_norm(hist[:1], hist[:1]).item() == 0.0
    # No interface: NaN row.
    assert torch.isnan(
        curvature_hist(torch.ones(1, 1, 8, 8, 8), torch.ones(1, 1, 8, 8, 8, dtype=torch.bool), bins=4, range=(0.0, 1.0))
    ).all()


def test_taubin_smoothing_removes_staircase_without_shrinking():
    r = 16
    phi, c = sphere_field(r)
    v, f = surface.marching_cubes(phi)
    smooth = surface.taubin_smooth(v, f)
    radius = lambda pts: np.linalg.norm(pts - c, axis=1)  # noqa: E731
    assert smooth.shape == v.shape
    assert radius(smooth).std() < 0.5 * radius(v).std()  # staircase filtered out
    assert abs(radius(smooth).mean() - radius(v).mean()) < 0.1  # ... but no shrinkage
    assert np.array_equal(surface.taubin_smooth(v, f, iterations=0), v)


# ---------------------------------------------------------------------------
# Robustness: a metric must never take down a run (an AB-UPT rollout prediction
# once fed libigl a mesh with duplicate vertices and zero-area faces, and the
# quadric fit segfaulted the training process — three times).


def _degenerate_stats(mesh):
    v, f = mesh.vertices, mesh.faces
    dup = len(v) - len(np.unique(v.round(12), axis=0))
    same = ((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])).sum() if len(f) else 0
    if len(f):
        area2 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
        flat = int((area2 < 1e-12).sum())
    else:
        flat = 0
    ref = np.zeros(len(v), dtype=bool)
    if len(f):
        ref[f.ravel()] = True
    return dup, int(same), flat, int((~ref).sum()) if len(v) else 0


def test_interface_mesh_of_a_noisy_field_has_no_degenerate_geometry():
    # A noisy prediction-like field: values crossing 0 with exact hits and steps, the
    # kind of input marching cubes answers with duplicate vertices and zero-area faces.
    rng = np.random.default_rng(0)
    phi = rng.uniform(-1, 1, (24, 24, 24)).astype(np.float32)
    phi[rng.random(phi.shape) < 0.2] = 0.0  # exact level hits: degenerate crossings
    solid = rng.random(phi.shape) < 0.25
    mesh = surface.interface_mesh(phi, solid, iterations=0)  # unsmoothed: raw triangulation
    assert len(mesh.faces) > 0
    dup, same, flat, unref = _degenerate_stats(mesh)
    assert dup == 0, f"{dup} duplicate vertices survived"
    assert same == 0, f"{same} faces with repeated indices survived"
    assert flat == 0, f"{flat} zero-area faces survived"
    assert unref == 0, f"{unref} unreferenced vertices survived"


def test_curvature_survives_a_noisy_field_end_to_end():
    rng = np.random.default_rng(1)
    phi = rng.uniform(-1, 1, (20, 20, 20)).astype(np.float32)
    phi[rng.random(phi.shape) < 0.2] = 0.0
    solid = rng.random(phi.shape) < 0.25
    c = cv.interface_curvature(phi, solid)
    assert len(c.k1) == len(c.vertices)  # finished, whatever the values


def test_principal_curvatures_recover_from_a_dead_worker():
    # The quadric fit runs in a worker subprocess so a C++ crash costs one NaN result,
    # not the training run. Kill the worker; the next call must restart it and succeed.
    field, _ = sphere_field(8.0)
    mesh = surface.interface_mesh(field, np.zeros_like(field, dtype=bool))
    k1, _ = cv.principal_curvatures(mesh.vertices, mesh.faces)
    assert np.isfinite(k1).any()
    assert cv._WORKER.proc is not None
    cv._WORKER.proc.kill()
    cv._WORKER.proc.wait()
    k1_again, _ = cv.principal_curvatures(mesh.vertices, mesh.faces)
    assert np.isfinite(k1_again).any()
    assert np.allclose(k1, k1_again, equal_nan=True)


def test_a_crashing_worker_yields_nan_not_a_dead_process(monkeypatch):
    field, _ = sphere_field(8.0)
    mesh = surface.interface_mesh(field, np.zeros_like(field, dtype=bool))
    broken = cv._IglWorker(source="import sys; sys.exit(1)")
    monkeypatch.setattr(cv, "_WORKER", broken)
    k1, k2 = cv.principal_curvatures(mesh.vertices, mesh.faces)
    assert np.isnan(k1).all() and np.isnan(k2).all()
