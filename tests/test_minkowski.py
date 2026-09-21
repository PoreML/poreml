"""Mesh-based Minkowski functionals against analytic surfaces.

M1 = ∫ dS (surface area), M2 = ∫ H dS (integral mean curvature), M3 = ∫ K dS (integral
Gaussian curvature, 2πχ by Gauss–Bonnet), plus M2 / M1, the area-weighted mean curvature.
Binary voxelised shapes at several resolutions go through the interface pipeline
(marching cubes → Taubin → quadric fit → vertex areas); figures with the classified
meshes and the computed / analytic ratio per resolution land in
`tests/test_output/minkowski/`.

Sphere:   M1 = 4πr²,   M2 = 4πr,   M2/M1 = 1/r,    M3 = 4π
Torus:    M1 = 4π²Rr,  M2 = 2π²R,  M2/M1 = 1/(2r), M3 = 0
Cylinder (open at the domain box, meshed length L−1): M1 = 2πr(L−1), M2 = π(L−1), M2/M1 = 1/(2r), M3 = 0
"""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("igl")
pytest.importorskip("skimage")

from poreml.metrics import curvature as cv  # noqa: E402
from poreml.metrics import surface  # noqa: E402
from poreml.metrics.descriptors import (  # noqa: E402
    area,
    curvature_gauss_integral,
    curvature_mean_integral,
    curvature_mean_integral_norm,
    euler,
    mesh_area,
    mesh_area_contact,
    mesh_area_interface,
)
from poreml.registry import DESCRIPTORS  # noqa: E402
from tests.test_curvature import sphere_field  # noqa: E402

OUT = Path(__file__).parent / "test_output" / "minkowski"
SPHERE_RADII = (8, 16, 32)
TORUS_TUBES = (5, 10, 20)  # tube radius r, ring radius R = 2r
CYLINDER_RADII = (6, 12, 24)


def _grid(shape):
    return np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij")


def torus_field(r: float) -> tuple[np.ndarray, float]:
    """`(phi, R)`: +1 inside a torus of tube radius `r` and ring radius `R = 2r`, axis along z."""
    big = 2.0 * r
    nxy = int(2 * (big + r) + 8)
    nz = int(2 * r + 8)
    cxy, cz = (nxy - 1) / 2, (nz - 1) / 2
    z, y, x = _grid((nz, nxy, nxy))
    rho = np.sqrt((x - cxy) ** 2 + (y - cxy) ** 2)
    phi = r - np.sqrt((rho - big) ** 2 + (z - cz) ** 2)
    return np.sign(phi), big


def cylinder_field(r: float) -> tuple[np.ndarray, int]:
    """`(phi, L)`: +1 inside a cylinder of radius `r` along the last (flow) axis, spanning the whole box."""
    length = int(4 * r)
    nxy = int(2 * r + 8)
    c = (nxy - 1) / 2
    z, y, x = _grid((nxy, nxy, length))
    phi = r - np.sqrt((z - c) ** 2 + (y - c) ** 2)
    return np.sign(phi), length


def _as_batch(phi: np.ndarray, solid: np.ndarray | None = None):
    field = torch.tensor(phi, dtype=torch.float32)[None, None]
    mask = torch.ones_like(field, dtype=torch.bool) if solid is None else torch.tensor(~solid)[None, None]
    return field, mask


def _functionals(phi: np.ndarray, solid: np.ndarray | None = None, **kw) -> dict[str, float]:
    field, mask = _as_batch(phi, solid)
    return {
        "M1": mesh_area(field, mask, **kw).item(),
        "M1_ff": mesh_area_interface(field, mask, **{k: v for k, v in kw.items() if k != "phase"}).item(),
        "M1_fs": mesh_area_contact(field, mask, **kw).item(),
        "M2": curvature_mean_integral(field, mask, **kw).item(),
        "M2/M1": curvature_mean_integral_norm(field, mask, **kw).item(),
        "M3": curvature_gauss_integral(field, mask, **kw).item(),
    }


ANALYTIC = {
    "sphere": lambda r: {"M1": 4 * np.pi * r**2, "M2": 4 * np.pi * r, "M2/M1": 1 / r, "M3": 4 * np.pi},
    "torus": lambda r: {"M1": 4 * np.pi**2 * 2 * r * r, "M2": 2 * np.pi**2 * 2 * r, "M2/M1": 1 / (2 * r), "M3": 0.0},
    # The mesh lives on the voxel-centre lattice: a cylinder spanning L voxels has meshed length L - 1.
    "cylinder": lambda r: {"M1": 2 * np.pi * r * (4 * r - 1), "M2": np.pi * (4 * r - 1), "M2/M1": 1 / (2 * r), "M3": 0.0},
}
INTEGRALS = (curvature_mean_integral, curvature_mean_integral_norm, curvature_gauss_integral)
FIELDS = {"sphere": sphere_field, "torus": torus_field, "cylinder": cylinder_field}
SCALES = {"sphere": SPHERE_RADII, "torus": TORUS_TUBES, "cylinder": CYLINDER_RADII}


# ---------------------------------------------------------------------------
# Mesh geometry


def test_face_and_vertex_areas_partition_the_surface():
    # One unit right triangle: face area 1/2, split 1/6 to each vertex.
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    f = np.array([[0, 1, 2]])
    assert surface.face_areas(v, f).tolist() == [0.5]
    assert surface.vertex_areas(v, f).tolist() == pytest.approx([1 / 6] * 3)
    # On a real mesh the vertex areas sum to the face areas exactly.
    phi, _ = sphere_field(10)
    mesh = surface.interface_mesh(phi, np.zeros_like(phi, dtype=bool))
    per_vertex = surface.vertex_areas(mesh.vertices, mesh.faces)
    assert per_vertex.sum() == pytest.approx(surface.face_areas(mesh.vertices, mesh.faces).sum())
    assert surface.face_areas(v[:0], f[:0]).shape == (0,)
    assert surface.vertex_areas(v, f[:0]).tolist() == [0.0, 0.0, 0.0]


def test_smoothed_mesh_area_of_a_sphere_beats_the_staircase():
    # Marching cubes on a binary field is a staircase whose area overshoots 4πr²; the
    # voxel-face `area` descriptor overshoots by 3/2 (a sphere's faces average 2/3
    # projected). Taubin smoothing brings the mesh within a few percent.
    r = 16
    phi, _ = sphere_field(r)
    analytic = 4 * np.pi * r**2
    raw_v, raw_f = surface.marching_cubes(phi)
    raw = surface.face_areas(raw_v, raw_f).sum()
    smooth = surface.interface_mesh(phi, np.zeros_like(phi, dtype=bool)).area()
    field, mask = _as_batch(phi)
    voxel = area(field, mask).item()
    assert voxel / analytic > 1.4
    assert raw / analytic > 1.05
    assert smooth == pytest.approx(analytic, rel=0.03)
    assert mesh_area(field, mask).item() == pytest.approx(smooth)


def test_area_parts_partition_the_total():
    r = 12
    phi, c = sphere_field(r)
    solid = np.zeros_like(phi, dtype=bool)
    solid[: int(c)] = True  # hemisphere on a solid floor
    mesh = surface.interface_mesh(phi, solid)
    assert mesh.area("fluid") + mesh.area("solid") == pytest.approx(mesh.area("all"))
    assert mesh.area("fluid") == pytest.approx(2 * np.pi * r**2, rel=0.08)
    assert mesh.area("solid") == pytest.approx(np.pi * r**2, rel=0.10)
    with pytest.raises(ValueError):
        mesh.area("buffer")


def test_smoothing_keeps_open_rims_on_the_box_faces():
    # A cylinder along the flow axis is cut open by the box at both ends. A free
    # boundary under the umbrella Laplacian creeps inward about a voxel per end, which
    # shortens every phase cut by the inlet plane of an ROI; the rim must stay on the
    # face it was born on, while the staircase along the rim is still smoothed away.
    r = 12
    phi, length = cylinder_field(r)
    raw_v, _ = surface.marching_cubes(phi)
    mesh = surface.interface_mesh(phi, np.zeros_like(phi, dtype=bool))
    on_face = (raw_v[:, 2] == 0) | (raw_v[:, 2] == length - 1)
    assert on_face.sum() > 0
    assert np.array_equal(mesh.vertices[on_face, 2], raw_v[on_face, 2])  # pinned along the flow axis ...
    c = (phi.shape[0] - 1) / 2
    rim_radius = np.linalg.norm(mesh.vertices[on_face][:, :2] - c, axis=1)
    assert rim_radius.std() < 0.5 * np.linalg.norm(raw_v[on_face][:, :2] - c, axis=1).std()  # ... but smoothed within the face
    assert rim_radius.mean() == pytest.approx(r, abs=0.3)
    assert mesh.area() == pytest.approx(2 * np.pi * r * (length - 1), rel=0.02)


# ---------------------------------------------------------------------------
# Descriptors on analytic shapes


@pytest.mark.parametrize("shape", ["sphere", "torus", "cylinder"])
def test_functionals_converge_to_analytic(shape):
    scale = SCALES[shape][-1]
    phi, _ = FIELDS[shape](scale)
    got = _functionals(phi)
    want = ANALYTIC[shape](scale)
    assert got["M1"] == pytest.approx(want["M1"], rel=0.03)
    assert got["M1_ff"] == pytest.approx(got["M1"])  # nothing touches solid
    assert got["M1_fs"] == 0.0
    assert got["M2/M1"] == pytest.approx(want["M2/M1"], rel=0.05)
    assert got["M2"] == pytest.approx(want["M2"], rel=0.08)
    # Gauss–Bonnet: 4π for a sphere, 0 for a torus and an open cylinder, within a
    # fraction of one sphere's worth (4π).
    assert abs(got["M3"] - want["M3"]) < 0.15 * 4 * np.pi


def test_gauss_bonnet_matches_the_euler_descriptor():
    # Two separate spheres. Gauss–Bonnet gives M3 = 2π·χ(surface), and a body's surface
    # has twice the body's Euler characteristic — the one the voxel `euler` descriptor
    # counts (1 per ball) — so M3 = 4π·euler = 8π here.
    r = 10
    one, _ = sphere_field(r)
    n = one.shape[0]
    phi = -np.ones((n, 2 * n + 4, n), dtype=np.float64)
    phi[:, :n] = one
    phi[:, n + 4 :] = one
    field, mask = _as_batch(phi)
    chi = euler(field, mask).item()
    assert chi == 2.0
    assert curvature_gauss_integral(field, mask).item() == pytest.approx(4 * np.pi * chi, rel=0.10)
    assert mesh_area(field, mask).item() == pytest.approx(2 * 4 * np.pi * r**2, rel=0.03)
    assert curvature_mean_integral(field, mask).item() == pytest.approx(2 * 4 * np.pi * r, rel=0.10)


def test_domain_box_cuts_are_not_surface():
    # A sphere protruding through the z=0 face of the box: the part outside is gone and
    # the cut is open, so M1 is the spherical cap inside (2πrh), no capping disk. This
    # is what excludes the inlet / outlet planes of an ROI from every functional.
    r, d = 16, 6  # centre `d` voxels inside the face: cap height h = r + d
    n = int(2 * r + 12)
    z, y, x = _grid((n, n, n))
    c = (n - 1) / 2
    phi = np.sign(r - np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - d) ** 2))
    got = _functionals(phi)
    cap = 2 * np.pi * r * (r + d)
    assert got["M1"] == pytest.approx(cap, rel=0.05)
    assert got["M1"] < 4 * np.pi * r**2 * 0.8
    assert got["M2/M1"] == pytest.approx(1 / r, rel=0.10)


def test_hemisphere_on_a_wall_scores_only_the_free_surface():
    r = 16
    phi, c = sphere_field(r)
    solid = np.zeros_like(phi, dtype=bool)
    solid[: int(c)] = True
    got = _functionals(phi, solid)
    assert got["M1_fs"] == pytest.approx(np.pi * r**2, rel=0.10)
    assert got["M1_ff"] == pytest.approx(2 * np.pi * r**2, rel=0.08)
    assert got["M1"] == pytest.approx(got["M1_ff"] + got["M1_fs"])
    # The integrals run over the fluid–fluid vertices outside the contact-line margin:
    # the weighted mean is still 1/r, the integral covers less than the full cap.
    assert got["M2/M1"] == pytest.approx(1 / r, rel=0.05)
    assert 0.5 * 2 * np.pi * r < got["M2"] < 2 * np.pi * r
    # M2 is the scored area times the weighted mean by construction.
    field, mask = _as_batch(phi, solid)
    scored = cv.batch_interface_curvature(field, mask)[0]
    assert got["M2"] == pytest.approx(float((scored.vertex_areas * scored.mean)[scored.valid].sum()), rel=1e-5)
    scored_area = float(scored.vertex_areas[scored.valid].sum())
    assert scored_area < got["M1_ff"]
    # H and K are constant on a sphere, so over the scored patch the exact integrals are A/r and A/r².
    assert got["M2"] == pytest.approx(scored_area / r, rel=0.05)
    assert got["M3"] == pytest.approx(scored_area / r**2, rel=0.10)


def test_wetting_phase_is_the_mirror_image():
    r = 12
    phi, _ = sphere_field(r)
    nw = _functionals(phi)
    w = _functionals(-phi, phase="w")  # the same droplet, now of the wetting phase
    for key in ("M1", "M1_ff", "M1_fs", "M2", "M2/M1", "M3"):
        assert w[key] == pytest.approx(nw[key], rel=0.02, abs=1e-3), key
    # Seen from the non-wetting side the same field is a bubble: area unchanged, H < 0.
    bubble = _functionals(-phi)
    assert bubble["M1"] == pytest.approx(nw["M1"], rel=0.02)
    assert bubble["M2"] == pytest.approx(-nw["M2"], rel=0.02)
    assert bubble["M2/M1"] == pytest.approx(-nw["M2/M1"], rel=0.02)
    assert bubble["M3"] == pytest.approx(nw["M3"], rel=0.05)


def test_no_interface_is_zero_area_and_nan_integrals_per_sample():
    r = 10
    phi, _ = sphere_field(r)
    droplet = torch.tensor(phi, dtype=torch.float32)[None, None]
    field = torch.cat([droplet, torch.ones_like(droplet), -torch.ones_like(droplet)])
    mask = torch.ones_like(field, dtype=torch.bool)
    a = mesh_area(field, mask)
    assert a.shape == (3,) and a.dtype == torch.float32
    assert a[0].item() == pytest.approx(4 * np.pi * r**2, rel=0.03)
    assert a[1].item() == 0.0 and a[2].item() == 0.0
    for fn in INTEGRALS:
        out = fn(field, mask)
        assert out.shape == (3,)
        assert torch.isfinite(out[0]) and torch.isnan(out[1:]).all()
    # All solid: no pore space, nothing to mesh.
    assert mesh_area(droplet, torch.zeros_like(droplet, dtype=torch.bool)).item() == 0.0


def test_phase_filling_the_pore_space_has_only_contact_area():
    # Non-wetting everywhere in a pore channel between two walls: no fluid–fluid
    # interface, a contact patch on each wall, and the channel ends are open.
    n = 24
    phi = np.ones((n, n, n))
    solid = np.zeros((n, n, n), dtype=bool)
    solid[:4] = True
    solid[-4:] = True
    got = _functionals(phi, solid)
    assert got["M1_ff"] == 0.0
    assert got["M1_fs"] == pytest.approx(2 * (n - 1) ** 2, rel=0.03)  # two planes on the voxel-centre lattice
    assert got["M1"] == pytest.approx(got["M1_fs"])
    assert np.isnan(got["M2"]) and np.isnan(got["M2/M1"]) and np.isnan(got["M3"])


def test_survives_a_noisy_prediction_like_field():
    rng = np.random.default_rng(3)
    phi = rng.uniform(-1, 1, (2, 1, 20, 20, 20)).astype(np.float32)
    phi[rng.random(phi.shape) < 0.2] = 0.0
    field = torch.tensor(phi)
    mask = torch.tensor(rng.random(phi.shape) >= 0.25)
    for fn in (mesh_area, mesh_area_interface, mesh_area_contact, *INTEGRALS):
        out = fn(field, mask)
        assert out.shape == (2,) and out.dtype == torch.float32
    assert (mesh_area(field, mask) > 0).all()


def _speckled(phi: np.ndarray, n: int = 15) -> np.ndarray:
    """`phi` plus `n` isolated single voxels of the positive phase, three voxels apart along two box edges.

    The shells marching cubes draws around them (6 vertices each) are what speckle noise in a
    model prediction produces by the thousand — and what segfaults libigl's quadric fit.
    """
    out = phi.copy()
    m = phi.shape[-1]
    sites = [(2, 2, k) for k in range(2, m - 2, 3)] + [(2, m - 3, k) for k in range(2, m - 2, 3)]
    assert len(sites) >= n
    for ijk in sites[:n]:
        assert out[ijk] < 0
        out[ijk] = 1.0
    return out


def test_components_below_the_vertex_threshold_are_dropped():
    phi, _ = sphere_field(8)
    solid = np.zeros_like(phi, dtype=bool)
    clean = surface.interface_mesh(phi, solid)
    speckled = surface.interface_mesh(_speckled(phi), solid)
    assert speckled.dropped_components == 15
    assert len(speckled.vertices) == len(clean.vertices)
    field, mask = _as_batch(_speckled(phi))
    assert mesh_area(field, mask).item() == pytest.approx(clean.area())


def test_component_threshold_is_a_keyword_not_a_config_knob():
    phi, _ = sphere_field(8)
    solid = np.zeros_like(phi, dtype=bool)
    clean = surface.interface_mesh(phi, solid)
    kept = surface.interface_mesh(_speckled(phi), solid, min_component_vertices=0)
    assert kept.dropped_components == 0
    assert len(kept.vertices) == len(clean.vertices) + 15 * 6  # a lone voxel meshes to a 6-vertex shell


def test_speckle_only_field_has_no_surface():
    phi, _ = sphere_field(8)
    field, mask = _as_batch(_speckled(-np.ones_like(phi)))
    assert mesh_area(field, mask).item() == 0.0
    for fn in INTEGRALS:  # no interface: NaN, not a ratio over a smoothed-away area
        assert torch.isnan(fn(field, mask)).all()


def test_registered_with_params():
    for name in ("mesh_area", "mesh_area_contact", *(fn.__name__ for fn in INTEGRALS)):
        fn = DESCRIPTORS.get(name)
        assert fn.kind == "scalar"
        assert fn.params == {"phase"}, name
    assert DESCRIPTORS.get("mesh_area_interface").params == set()


def test_composes_in_a_metric_spec():
    from poreml.metrics import MetricSpec, compute, resolve

    r = 12
    phi, _ = sphere_field(r)
    target = torch.tensor(np.pad(phi, 2, constant_values=-1), dtype=torch.float32)[None, None]
    pred = torch.tensor(np.pad(sphere_field(r - 2)[0], 4, constant_values=-1), dtype=torch.float32)[None, None]
    assert pred.shape == target.shape
    mask = torch.ones_like(target, dtype=torch.bool)
    specs = resolve(
        [
            MetricSpec(descriptor="mesh_area", error="rel_err", channel="phi"),
            MetricSpec(descriptor="curvature_mean_integral_norm", error="rel_err", channel="phi"),
            MetricSpec(descriptor="curvature_mean_integral_norm", error="target", channel="phi"),
        ],
        ["phi"],
    )
    out = compute(specs, pred, target, mask)
    assert out["mesh_area/rel_err@phi"].item() == pytest.approx(1 - ((r - 2) / r) ** 2, abs=0.03)
    assert out["curvature_mean_integral_norm/rel_err@phi"].item() == pytest.approx(r / (r - 2) - 1, abs=0.05)
    assert out["curvature_mean_integral_norm/target@phi"].item() == pytest.approx(1 / r, rel=0.05)


# ---------------------------------------------------------------------------
# Figures: classified meshes with their functionals, and convergence with resolution.

SERIES = {"sphere": "#2a78d6", "torus": "#eb6834", "cylinder": "#1baf7a"}  # fixed per shape, never cycled
CLASS_COLOURS = ("#2a78d6", "#eb6834", "#9a9a96")  # scored free surface, contact patch, excluded margin / rim


def _scored_patch_targets(phi: np.ndarray, solid: np.ndarray, r: float, want: dict) -> dict:
    """Analytic M2 / M3 for the part of a sphere the integrals actually cover.

    The contact-line margin (and nothing else) trims the scored surface, so the exact
    integrals over the scored patch of area A are A/r and A/r² — H and K are constant on
    a sphere. This is the analytic target, not a finer mesh.
    """
    field, mask = _as_batch(phi, solid)
    res = cv.batch_interface_curvature(field, mask)[0]
    scored = float(res.vertex_areas[res.valid].sum())
    return {**want, "M2": scored / r, "M3": scored / r**2}


def _hemisphere(r: int):
    phi, c = sphere_field(r)
    solid = np.zeros_like(phi, dtype=bool)
    solid[: int(c)] = True
    want = {"M1": 3 * np.pi * r**2, "M1_ff": 2 * np.pi * r**2, "M1_fs": np.pi * r**2, "M2/M1": 1 / r}
    return phi, solid, _scored_patch_targets(phi, solid, r, want)


def _box_cut_sphere(r: int, d: int):
    n = int(2 * r + 12)
    z, y, x = _grid((n, n, n))
    c = (n - 1) / 2
    phi = np.sign(r - np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - d) ** 2))
    cap = 2 * np.pi * r * (r + d)  # nothing is trimmed here: the rim is scored, so the whole cap is the target
    want = {"M1": cap, "M1_ff": cap, "M1_fs": 0.0, "M2": cap / r, "M2/M1": 1 / r, "M3": cap / r**2}
    return phi, np.zeros_like(phi, dtype=bool), want


def _face_classes(res: cv.InterfaceCurvature) -> np.ndarray:
    """0 scored free surface, 1 contact patch, 2 margin / rim (meshed but not integrated)."""
    valid = res.valid[res.faces].all(axis=1)
    solid = res.near_solid[res.faces].all(axis=1)
    return np.where(valid, 0, np.where(solid, 1, 2))


def _class_mesh_panel(ax, res: cv.InterfaceCurvature, classes: np.ndarray) -> None:
    """Triangle mesh coloured by class with the same Lambert relief as `plot3d.value_mesh_panel`."""
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from tests.plot3d import LIGHT, _style

    tri = res.vertices[res.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    intensity = np.abs(normals[:, ::-1] @ LIGHT.direction)
    rgba = np.array([to_rgba(c) for c in CLASS_COLOURS])[classes]
    rgba[:, :3] *= (0.55 + 0.45 * intensity)[:, None]
    ax.add_collection3d(Poly3DCollection(tri[:, :, ::-1], facecolors=rgba, edgecolors="none"))
    zz, yy, xx = res.vertices.T
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_zlim(zz.min(), zz.max())
    _style(ax, (np.ptp(xx), np.ptp(yy), np.ptp(zz)))


def _plot_case(name: str, phi: np.ndarray, solid: np.ndarray, got: dict, want: dict, unit: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from tests.plot3d import mesh_panel, value_mesh_panel

    OUT.mkdir(parents=True, exist_ok=True)
    field, mask = _as_batch(phi, solid)
    res = cv.batch_interface_curvature(field, mask)[0]
    raw_v, raw_f = surface.marching_cubes(np.where(solid, -2.0, phi))
    fig = plt.figure(figsize=(17, 5.2))
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.9))

    elev = -30 if solid.any() else 20  # a contact patch sits on the floor: look up at it
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    mesh_panel(ax, raw_v, raw_f)
    ax.view_init(elev=elev)
    ax.set_title(f"marching cubes: {len(raw_f)} faces (staircase)", fontsize=10)

    ax = fig.add_subplot(grid[0, 1], projection="3d")
    _class_mesh_panel(ax, res, _face_classes(res))
    ax.view_init(elev=elev)
    ax.set_title("smoothed, by class", fontsize=10)
    labels = ("free surface, scored", "contact patch", "margin / open rim")
    handles = [Patch(color=colour, label=label) for colour, label in zip(CLASS_COLOURS, labels, strict=True)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)

    ax = fig.add_subplot(grid[0, 2], projection="3d")
    face_h = np.nanmean(np.where(res.valid, res.mean, np.nan)[res.faces], axis=1) * unit
    mappable = value_mesh_panel(ax, res.vertices, res.faces, face_h, cmap="coolwarm", clim=(-1.5, 1.5))
    ax.view_init(elev=elev)
    fig.colorbar(mappable, ax=ax, location="bottom", shrink=0.7, pad=0.02, label="H · scale  (analytic sphere = 1)")
    ax.set_title("mean curvature on the scored surface", fontsize=10)

    ax = fig.add_subplot(grid[0, 3])
    ax.set_axis_off()
    rows = []
    for key in ("M1", "M1_ff", "M1_fs", "M2", "M2/M1", "M3"):
        w = want.get(key, float("nan"))
        if not np.isfinite(w):
            ratio, target = "–", "–"
        else:
            ratio, target = (f"{got[key] / w:.3f}" if w != 0 else f"Δ={got[key] - w:+.2f}"), f"{w:.4g}"
        rows.append((key, f"{got[key]:.4g}", target, ratio))
    table = ax.table(cellText=rows, colLabels=("functional", "computed", "analytic", "ratio"), loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (_, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d8d8d4")
        if col == 0:
            cell.set_text_props(ha="left")
    ax.set_title(f"{len(res.vertices)} vertices, {int(res.valid.sum())} scored", fontsize=10)

    fig.suptitle(f"{name}: Minkowski functionals on the interface mesh (voxel² / voxel / 1/voxel / –)", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.04, wspace=0.05)
    path = OUT / f"{name.split(',')[0].replace(' ', '_')}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_convergence(results: dict) -> Path:
    """2×2: computed / analytic vs scale for M1, M2, M2/M1; M3 in units of 4π. One colour per shape."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [
        ("M1", "(a) M1 = ∫dS, computed / analytic"),
        ("M2", "(b) M2 = ∫H dS, computed / analytic"),
        ("M2/M1", "(c) M2 / M1, computed / analytic"),
        ("M3", "(d) M3 = ∫K dS, computed − analytic, in units of 4π"),
    ]
    label_at_end = dict(xytext=(6, 0), textcoords="offset points", fontsize=9, va="center")
    for ax, (key, title) in zip(axes.ravel(), panels, strict=True):
        for shape, colour in SERIES.items():
            scales = list(results[shape])
            if key == "M3":
                # A torus and an open cylinder have M3 = 0 analytically, so a ratio is undefined:
                # the difference, in units of one sphere's worth (4π), reads on one axis for all.
                values = [(results[shape][s]["got"]["M3"] - results[shape][s]["want"]["M3"]) / (4 * np.pi) for s in scales]
                target = 0.0
            else:
                values = [results[shape][s]["got"][key] / results[shape][s]["want"][key] for s in scales]
                target = 1.0
            ax.plot(scales, values, "o-", color=colour, lw=2, ms=6, label=shape)
            if key != "M1":  # every M1 curve ends on the same point; the legend names them there
                ax.annotate(shape, (scales[-1], values[-1]), color=colour, **label_at_end)
            ax.axhline(target, color="#6b6b68", lw=1, ls="--", zorder=0)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("radius [voxels]")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.25)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[1, 1].set_ylim(-0.15, 0.15)
    fig.suptitle("Mesh Minkowski functionals vs resolution (binary voxel input; dashed = analytic)", fontsize=12)
    fig.tight_layout()
    path = OUT / "convergence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def test_figures_and_convergence():
    pytest.importorskip("matplotlib")
    results = {shape: {} for shape in SERIES}
    for shape in SERIES:
        for scale in SCALES[shape]:
            phi, _ = FIELDS[shape](scale)
            results[shape][scale] = {"got": _functionals(phi), "want": ANALYTIC[shape](scale)}
    assert _plot_convergence(results).exists()
    # Every functional with a non-zero analytic value converges: the coarsest sphere's
    # M1 error is at least twice the finest's, and the finest is within tolerance.
    err = lambda s, k: abs(results["sphere"][s]["got"][k] / results["sphere"][s]["want"][k] - 1)  # noqa: E731
    for key in ("M1", "M2", "M2/M1"):
        assert err(SPHERE_RADII[0], key) > 2 * err(SPHERE_RADII[-1], key) or err(SPHERE_RADII[0], key) < 0.01, key
    # 3-D panels at the middle resolution: matplotlib's polygon collections need tens of
    # gigabytes for the ~90k-face meshes of the finest torus (the OOM killer took the
    # test), and the classification reads the same at a quarter of the faces.
    for shape in SERIES:
        scale = SCALES[shape][1]
        phi, _ = FIELDS[shape](scale)
        got, want = results[shape][scale]["got"], results[shape][scale]["want"]
        assert _plot_case(
            f"{shape}, r={scale}",
            phi,
            np.zeros_like(phi, dtype=bool),
            got,
            want,
            unit=scale if shape == "sphere" else 2 * scale,
        ).exists()
    phi, solid, want = _hemisphere(16)
    assert _plot_case("hemisphere on a wall, r=16", phi, solid, _functionals(phi, solid), want, unit=16).exists()
    phi, solid, want = _box_cut_sphere(16, 6)
    assert _plot_case("sphere cut by the box, r=16", phi, solid, _functionals(phi, solid), want, unit=16).exists()
