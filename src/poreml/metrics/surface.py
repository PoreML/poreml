"""Fluid interfaces as triangle meshes: extraction, smoothing, and fluid–fluid / fluid–solid split.

The voxel-to-surface workflow follows the CT contact-angle literature and, in structure,
the geometricContactAngle toolkit (Aljaberi, Belhaj, Foroughi et al., "Spatially
distributed wettability characterization in porous media", Sci Rep 16, 12643, 2026;
https://github.com/ImperialCollegeLondon/geometricContactAngle, MIT): marching cubes on
the phase, Taubin λ|μ smoothing to strip the voxel staircase without shrinking the shape,
then a split of the surface into the fluid–fluid interface and the fluid–solid contact
patch whose common boundary is the three-phase contact line. Their toolkit finds the
split by meshing the solid separately and matching coincident vertices
(`Voxel2Surface.separate_interfaces`); on a continuous phase field the crossing does not
land on the solid mesh, so here a vertex is *near solid* when the grid edge it sits on
has a solid endpoint — the same classification without a second marching cubes.

`interface_mesh` is the entry point; `taubin_smooth` is reusable on any triangle mesh.
scikit-image is imported lazily so the core package installs without the `curvature` group.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Taubin (1995) λ|μ: λ shrinks, μ < -λ inflates, so the pass band `1/λ + 1/μ` keeps the
# low-frequency shape (the radius a curvature measures) while the staircase is filtered
# out. Chosen on voxelised spheres of radius 8–32 (tests/test_curvature.py): the median
# mean curvature lands within ~1–5% of 1/r for staircase and interpolated inputs alike
# and the spread is smallest around here. geometricContactAngle / trimesh's 0.5 | -0.53
# (pass band 0.11) smooths slowly and drifts the radius outward over a few hundred
# iterations; 0.75 | -0.76 (pass band 0.018) converges in ~200 with no drift.
DEFAULT_ITERATIONS = 200
DEFAULT_LAMBDA = 0.75
DEFAULT_MU = -0.76

# A connected component of fewer vertices than this is dropped before anything is
# measured. Marching cubes draws a closed 6-vertex shell around one lone voxel and
# 10–30 around two or three; a noisy prediction carries thousands of them (one 256^3
# Transolver frame: 14 502 of its 15 498 components) while no real meniscus is that small
# (the smallest interface component in a 128^3 target mesh had 40). libigl's quadric fit
# segfaults on meshes that contain them — memory-layout dependent, reproduced
# 2026-09-13 — and a shell that size smooths to ~0 area, so its curvature per area is
# noise. Not a config knob: it changes what every mesh descriptor computes
# (METRICS_VERSION 5); it stays a keyword so the calibration tests can vary it.
DEFAULT_MIN_COMPONENT_VERTICES = 30

_MISSING = "interface meshes need scikit-image and libigl: `uv sync --group curvature`"


def _marching_cubes():
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:  # pragma: no cover - only without the group
        raise ImportError(_MISSING) from exc
    return marching_cubes


@dataclass
class InterfaceMesh:
    """One sample's fluid surface after smoothing, split by what lies on the other side."""

    vertices: np.ndarray  # (N, 3) float64, voxel index coordinates in field axis order
    faces: np.ndarray  # (M, 3) int64
    near_solid: np.ndarray  # (N,) bool: the vertex sits on a fluid–solid grid edge
    dropped_components: int = 0  # components under DEFAULT_MIN_COMPONENT_VERTICES removed before smoothing

    @property
    def fluid_solid_faces(self) -> np.ndarray:
        """Bool `(M,)`: faces of the contact patch — every vertex is on a fluid–solid edge."""
        return self.near_solid[self.faces].all(axis=1) if len(self.faces) else np.zeros(0, dtype=bool)

    @property
    def fluid_fluid_faces(self) -> np.ndarray:
        return ~self.fluid_solid_faces

    @property
    def fluid_vertices(self) -> np.ndarray:
        """Bool `(N,)`: vertices of the fluid–fluid interface away from the contact line."""
        return ~self.near_solid

    @property
    def contact_vertices(self) -> np.ndarray:
        """Bool `(N,)`: the three-phase contact line — near-solid vertices used by a fluid–fluid face."""
        flag = np.zeros(len(self.vertices), dtype=bool)
        if len(self.faces):
            flag[np.unique(self.faces[self.fluid_fluid_faces])] = True
        return flag & self.near_solid

    @property
    def face_areas(self) -> np.ndarray:
        """`(M,)` triangle areas in voxel²."""
        return face_areas(self.vertices, self.faces)

    @property
    def vertex_areas(self) -> np.ndarray:
        """`(N,)` barycentric vertex areas: one third of each incident face, summing to the surface area."""
        return vertex_areas(self.vertices, self.faces)

    def area(self, part: str = "all") -> float:
        """Surface area [voxel²] of `part`: "all", "fluid" (fluid–fluid faces) or "solid" (contact patch).

        The first Minkowski functional M1 = ∫ dS of the phase region, on the smoothed mesh
        rather than the voxel staircase. Only faces present in the mesh count: the surface
        is closed against solid but open where the phase leaves the domain box, so an
        inlet, outlet or buffer plane never contributes.

        The split is by vertex area, not by `fluid_solid_faces`: a triangle straddling the
        contact line is shared between its near-solid and free vertices (a third each),
        so the two parts sum to the total and the corner ring is not all handed to one
        side — on a hemisphere on a floor the face rule leaves the disk 4–13% short, the
        vertex split 3–6% and converging with radius.
        """
        if part == "all":
            return float(self.face_areas.sum())
        if part == "fluid":
            return float(self.vertex_areas[~self.near_solid].sum())
        if part == "solid":
            return float(self.vertex_areas[self.near_solid].sum())
        raise ValueError(f"part must be 'all', 'fluid' or 'solid'; got {part!r}")


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """`(M,)` float64: area of each triangle, half the cross product of two edges."""
    if len(faces) == 0:
        return np.zeros(0, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """`(N,)` float64: barycentric vertex areas — a third of every incident triangle.

    The discrete area element of a per-vertex quantity: `Σ_v q_v A_v` approximates
    `∫ q dS`, and `Σ_v A_v` is exactly the mesh area (each triangle is split three ways).
    """
    out = np.zeros(len(vertices), dtype=np.float64)
    if len(faces):
        np.add.at(out, faces.ravel(), np.repeat(face_areas(vertices, faces) / 3.0, 3))
    return out


def dedupe_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices, drop degenerate faces, drop unreferenced vertices.

    Marching cubes on a field that hits the level exactly (a noisy model prediction does)
    emits duplicate vertices and zero-area triangles. libigl's quadric fit is not defensive
    about them — a rollout prediction once segfaulted it and took the training process
    down — so no such mesh may leave this module.
    """
    if len(faces) == 0:
        return vertices[:0], faces
    # Merge on rounded coordinates: degenerate crossings emit copies that differ by
    # float noise as well as exact ones. 1e-9 voxels is far below anything physical.
    _, first, inverse = np.unique(vertices.round(9), axis=0, return_index=True, return_inverse=True)
    unique = vertices[first]
    faces = inverse[faces]
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[distinct]
    if len(faces):
        e1 = unique[faces[:, 1]] - unique[faces[:, 0]]
        e2 = unique[faces[:, 2]] - unique[faces[:, 0]]
        faces = faces[np.linalg.norm(np.cross(e1, e2), axis=1) >= 1e-12]
    referenced = np.zeros(len(unique), dtype=bool)
    referenced[faces.ravel()] = True
    remap = np.cumsum(referenced) - 1
    return unique[referenced], np.ascontiguousarray(remap[faces], dtype=np.int64)


def drop_small_components(
    vertices: np.ndarray, faces: np.ndarray, min_vertices: int = DEFAULT_MIN_COMPONENT_VERTICES
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove every connected component with fewer than `min_vertices` vertices.

    Returns `(vertices, faces, n_dropped)`; the dropped vertices leave with their faces and
    the rest are renumbered. `min_vertices <= 1` keeps everything.
    """
    if min_vertices <= 1 or len(faces) == 0:
        return vertices, faces, 0
    n_components, labels = connected_components(vertex_adjacency(len(vertices), faces), directed=False)
    keep_component = np.bincount(labels, minlength=n_components) >= min_vertices
    if keep_component.all():
        return vertices, faces, 0
    keep = keep_component[labels]
    remap = np.cumsum(keep) - 1
    faces = faces[keep[faces].all(axis=1)]
    return vertices[keep], np.ascontiguousarray(remap[faces], dtype=np.int64), int((~keep_component).sum())


def marching_cubes(field: np.ndarray, level: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate `field == level` for a `(D, H, W)` array; `(0, 3)` arrays if never crossed.

    Vertices are voxel index coordinates, axis order as the array; the surface normal
    points toward decreasing `field`. The triangulation is passed through `dedupe_mesh`.
    """
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError(f"marching cubes needs a 3-D field; got shape {field.shape}")
    if not (field.min() < level < field.max()):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    vertices, faces, _, _ = _marching_cubes()(field, level=level)
    return dedupe_mesh(vertices.astype(np.float64), faces.astype(np.int64))


def taubin_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    lam: float = DEFAULT_LAMBDA,
    mu: float = DEFAULT_MU,
    fixed: np.ndarray | None = None,
) -> np.ndarray:
    """Taubin λ|μ smoothing of vertex positions with the uniform (umbrella) Laplacian.

    Each iteration moves every vertex by `lam` of the way to its neighbours' mean, then
    by `mu` (negative) of the way back — Taubin, "A signal processing approach to fair
    surface design", SIGGRAPH 1995. Same operator as trimesh's `filter_taubin`, which
    geometricContactAngle applies. Connectivity is unchanged; `iterations=0` is the identity.

    `fixed`, a bool `(N, 3)`, holds the flagged vertex *components* at their input
    values: a vertex on an open rim keeps the coordinate that puts it on the rim's plane
    and is smoothed within it. The umbrella operator has no neighbours beyond a free
    boundary, so an unconstrained rim creeps inward about a voxel per pass band.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if iterations == 0 or len(faces) == 0:
        return vertices.copy()
    adjacency = vertex_adjacency(len(vertices), faces)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0  # isolated vertex: laplacian is 0 below
    mean_of_neighbours = sp.diags(1.0 / degree) @ adjacency
    v = vertices.copy()
    if fixed is not None:
        fixed = np.asarray(fixed, dtype=bool)
        if fixed.shape != vertices.shape:
            raise ValueError(f"fixed must be a bool (N, 3) mask matching vertices {vertices.shape}; got {fixed.shape}")
        held = vertices[fixed]
    for _ in range(iterations):
        v += lam * (mean_of_neighbours @ v - v)
        v += mu * (mean_of_neighbours @ v - v)
        if fixed is not None:
            v[fixed] = held
    return v


def box_face_components(vertices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Bool `(N, 3)`: the vertex coordinate lies on the first or last voxel centre of its axis.

    Marching cubes leaves a surface open where it exits the array, and those rim
    vertices sit exactly on the box face. Holding that one coordinate during smoothing
    keeps an inlet-plane or outlet-plane cut where it is (`interface_mesh`).
    """
    if len(vertices) == 0:
        return np.zeros((0, 3), dtype=bool)
    last = np.asarray(shape, dtype=np.float64) - 1
    return (vertices == 0) | (vertices == last)


def vertex_adjacency(n_vertices: int, faces: np.ndarray) -> sp.csr_matrix:
    """Symmetric 0/1 `(N, N)` vertex adjacency of a triangle mesh."""
    if len(faces) == 0:
        return sp.csr_matrix((n_vertices, n_vertices))
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adjacency = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_vertices, n_vertices)).tocsr()
    return ((adjacency + adjacency.T) > 0).astype(np.float64)


def within_rings(faces: np.ndarray, seed: np.ndarray, rings: int) -> np.ndarray:
    """Bool `(N,)`: `seed` vertices plus everything reachable in at most `rings` edges.

    The mesh analogue of geometricContactAngle's `_remove_edge_layers`: a quadric or
    normal fit near the contact line sees the other side of the corner, so a margin of
    the fit radius is excluded before taking statistics.
    """
    seed = np.asarray(seed, dtype=bool)
    reached = seed.copy()
    if rings <= 0 or not reached.any() or len(faces) == 0:
        return reached
    adjacency = vertex_adjacency(len(seed), faces)
    for _ in range(rings):
        grown = reached | (adjacency @ reached.astype(np.float64) > 0)
        if np.array_equal(grown, reached):
            break
        reached = grown
    return reached


def near_solid_vertices(vertices: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Bool `(N,)`: any voxel of the grid cell containing the vertex is solid.

    A marching-cubes vertex sits on a grid edge between two voxel centres, so `floor` and
    `ceil` of its coordinates enumerate the endpoints; on the fluid–solid level set one
    of them is solid.
    """
    if len(vertices) == 0:
        return np.zeros(0, dtype=bool)
    solid = np.asarray(solid, dtype=bool)
    lo = np.floor(vertices).astype(np.int64)
    hi = np.ceil(vertices).astype(np.int64)
    limit = np.asarray(solid.shape) - 1
    flag = np.zeros(len(vertices), dtype=bool)
    for corner in range(8):
        pick = np.array([(corner >> axis) & 1 for axis in range(3)], dtype=bool)
        idx = np.clip(np.where(pick, hi, lo), 0, limit)
        flag |= solid[idx[:, 0], idx[:, 1], idx[:, 2]]
    return flag


def interface_mesh(
    field: np.ndarray,
    solid: np.ndarray,
    *,
    level: float = 0.0,
    iterations: int = DEFAULT_ITERATIONS,
    lam: float = DEFAULT_LAMBDA,
    mu: float = DEFAULT_MU,
    min_component_vertices: int = DEFAULT_MIN_COMPONENT_VERTICES,
) -> InterfaceMesh:
    """Smoothed, classified surface of the positive side of one `(D, H, W)` field.

    Solid voxels are filled below `level` so a phase touching the wall closes its
    surface there: the mesh is watertight against solid, and the contact patch comes
    back flagged rather than cut away. Connected components of fewer than
    `min_component_vertices` vertices — the shells around speckles — are dropped before
    smoothing and counted in `dropped_components`. The only open boundary left is the
    domain box — an ROI's inlet and outlet planes — and those rim vertices are held on
    their box face while smoothed within it (`box_face_components`), so a cut phase
    neither shrinks nor grows a lip. Classification uses the unsmoothed vertex positions,
    which is where the grid-edge test is exact.
    """
    field = np.asarray(field, dtype=np.float32)
    solid = np.asarray(solid, dtype=bool)
    if field.shape != solid.shape:
        raise ValueError(f"field {field.shape} and solid {solid.shape} must have the same shape")
    pore = ~solid
    fill = min(float(field[pore].min()) if pore.any() else level, level) - 1.0
    vertices, faces = marching_cubes(np.where(solid, fill, field), level)
    vertices, faces, dropped = drop_small_components(vertices, faces, min_component_vertices)
    near_solid = near_solid_vertices(vertices, solid)
    fixed = box_face_components(vertices, field.shape)
    smoothed = taubin_smooth(vertices, faces, iterations=iterations, lam=lam, mu=mu, fixed=fixed)
    return InterfaceMesh(smoothed, faces, near_solid, dropped)
