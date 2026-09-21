"""Layer 1: physical descriptors of a single field.

A descriptor maps `(field, mask, **params)` to one value per sample. `field` is
`(B, C, *spatial)`, `mask` is `(B, 1, *spatial)` bool with True at voxels that are pore
space; solid is `~mask` and is never a fluid. Scalar descriptors need a single channel.
Spatial rank is free — the same code scores 2-D and 3-D fields — and the last spatial
axis is the solver's flow axis (inlet at index 0, outlet at -1).

Every descriptor that measures a fluid takes `phase`, defaulting to the non-wetting
(invading) phase; `area_interface` is the same quantity from either side so it has none.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import combinations
from typing import Any, Literal

import torch
from torch import Tensor

from ..registry import DESCRIPTORS
from .trace import record

# the solver writes phi as a colour field on pore space: phi > 0 is non-wetting, phi < 0 wetting.
PHASE_THRESHOLD = 0.0

Phase = Literal["nw", "w"]

# What a descriptor returns, and what an error must `accept` to read it. These are the
# kinds, not descriptor names: `VOXELS_KIND` is the kind of the `voxels` descriptor
# (`spec.VOXELS`), and a later identity-like descriptor could share the kind.
SCALAR_KIND = "scalar"
DISTRIBUTION_KIND = "distribution"
VOXELS_KIND = "voxels"


def descriptor(name: str, *, kind: str = SCALAR_KIND, params: Iterable[str] = ()) -> Callable:
    """Register a descriptor and stamp it with its output kind and accepted parameters.

    `kind` is "scalar" (returns `(B,)`), "distribution" (returns `(B, K)`), or "voxels"
    (the identity, scored directly by voxel errors). `params` names the keyword
    arguments a spec may set; `resolve` rejects anything else up front.
    """

    def decorate(fn):
        fn.kind = kind
        fn.params = frozenset(params)
        return DESCRIPTORS.register(name)(fn)

    return decorate


def check_field(field: Tensor, mask: Tensor) -> None:
    """Raise unless `field` is `(B, C, *spatial)` and `mask` is bool `(B, 1, *spatial)`."""
    if field.ndim < 3:
        raise ValueError(f"field must be (batch, channel, *spatial); got shape {tuple(field.shape)}")
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be bool; got {mask.dtype}")
    if mask.ndim != field.ndim or mask.shape[1] != 1 or mask.shape[0] != field.shape[0] or mask.shape[2:] != field.shape[2:]:
        raise ValueError(f"mask must be (batch, 1, *spatial) matching field {tuple(field.shape)}; got {tuple(mask.shape)}")


def _check_single_channel(field: Tensor, mask: Tensor) -> None:
    check_field(field, mask)
    if field.shape[1] != 1:
        raise ValueError(f"scalar descriptors need a single channel; got {field.shape[1]} — set `channel` in the metric spec")


def _sum_per_sample(x: Tensor) -> Tensor:
    return x.sum(dim=tuple(range(1, x.ndim))).to(torch.float32)


def phase_mask(field: Tensor, mask: Tensor, phase: Phase) -> Tensor:
    """Bool `(B, 1, *spatial)`: voxels of `phase` inside the pore space."""
    nw = field > PHASE_THRESHOLD
    return (nw if phase == "nw" else ~nw) & mask


def _face_pairs(a: Tensor, b: Tensor) -> Tensor:
    """Per-sample count of 6-connected faces between an `a` voxel and a `b` voxel.

    Domain-boundary faces are never counted: there is no neighbour on the other side.
    Both directions of each axis are summed so a face is counted once regardless of
    which side `a` is on.
    """
    counts = torch.zeros(a.shape[0], dtype=torch.float32, device=a.device)
    for axis in range(2, a.ndim):
        n = a.shape[axis]
        lo, hi = a.narrow(axis, 0, n - 1), a.narrow(axis, 1, n - 1)
        blo, bhi = b.narrow(axis, 0, n - 1), b.narrow(axis, 1, n - 1)
        counts += _sum_per_sample((lo & bhi) | (hi & blo))
    return counts


@descriptor("voxels", kind=VOXELS_KIND)
def voxels(field: Tensor, mask: Tensor) -> Tensor:
    """The identity: voxel errors (mae, rmse, iou) score the field itself under the mask."""
    check_field(field, mask)
    return field


@descriptor("volume", params=("phase",))
def volume(field: Tensor, mask: Tensor, *, phase: Phase = "nw") -> Tensor:
    """Voxel count of `phase` in the pore space."""
    _check_single_channel(field, mask)
    return _sum_per_sample(phase_mask(field, mask, phase))


@descriptor("saturation", params=("phase",))
def saturation(field: Tensor, mask: Tensor, *, phase: Phase = "nw") -> Tensor:
    """`volume(phase) / pore volume`; NaN for a sample with no pore voxel."""
    _check_single_channel(field, mask)
    pore = _sum_per_sample(mask)
    phase_voxels = _sum_per_sample(phase_mask(field, mask, phase))
    record("pore_voxels", pore)
    record("phase_voxels", phase_voxels)
    return phase_voxels / pore  # 0/0 -> NaN


@descriptor("area_interface")
def area_interface(field: Tensor, mask: Tensor) -> Tensor:
    """Fluid–fluid interfacial area: faces between a non-wetting and a wetting voxel."""
    _check_single_channel(field, mask)
    return _face_pairs(phase_mask(field, mask, "nw"), phase_mask(field, mask, "w"))


@descriptor("area_contact", params=("phase",))
def area_contact(field: Tensor, mask: Tensor, *, phase: Phase = "nw") -> Tensor:
    """Fluid–solid contact area: faces between a `phase` voxel and a solid voxel."""
    _check_single_channel(field, mask)
    return _face_pairs(phase_mask(field, mask, phase), ~mask)


@descriptor("area", params=("phase",))
def area(field: Tensor, mask: Tensor, *, phase: Phase = "nw") -> Tensor:
    """Total surface area of `phase`: faces to the other fluid plus faces to solid.

    `area_interface + area_contact(phase)` in one number — the boundary of the phase
    region. Domain-boundary faces are not counted (there is no neighbour there).
    """
    _check_single_channel(field, mask)
    fluid = phase_mask(field, mask, phase)
    return _face_pairs(fluid, ~fluid)


def _shift(x: Tensor, axis: int, step: int) -> Tensor:
    """`x` moved by `step` (+1 or -1) along `axis`, zero-filled at the vacated edge."""
    n = x.shape[axis]
    out = torch.zeros_like(x)
    if step > 0:
        out.narrow(axis, 1, n - 1).copy_(x.narrow(axis, 0, n - 1))
    else:
        out.narrow(axis, 0, n - 1).copy_(x.narrow(axis, 1, n - 1))
    return out


def reach_from_ends(phase: Tensor) -> Tensor:
    """Voxels of `phase` (bool `(B, 1, *spatial)`) 6-connected to the inlet or outlet slice.

    A batched flood fill: seed with the phase voxels on the first and last slice of the
    flow axis, then grow along face neighbours while staying inside `phase` until nothing
    changes. Iterations scale with the longest path, not with voxel count, and every
    iteration is a handful of shifted boolean ops — fine on GPU for 128^3 domains.
    """
    reached = torch.zeros_like(phase)
    reached[..., 0] = phase[..., 0]
    reached[..., -1] = phase[..., -1]
    while True:
        grown = reached.clone()
        for axis in range(2, phase.ndim):
            grown |= _shift(reached, axis, +1) | _shift(reached, axis, -1)
        grown &= phase
        if torch.equal(grown, reached):
            return reached
        reached = grown


@descriptor("trapped_volume", params=("phase",))
def trapped_volume(field: Tensor, mask: Tensor, *, phase: Phase = "nw") -> Tensor:
    """Voxels of `phase` in 6-connected clusters touching neither the inlet nor the outlet.

    Seeding uses the first and last slice of the tensor exactly as given. the solver's domain is
    larger than the geometry — an inlet buffer and an outlet plate slab, whose spans
    `RunRef.regions` carries — so on an uncropped run the last slice may be solid and
    there is then no outlet seed at all: fluid that has broken through reads as trapped.
    Crop to the rock span before scoring, or accept that reading.
    """
    _check_single_channel(field, mask)
    fluid = phase_mask(field, mask, phase)
    return _sum_per_sample(fluid & ~reach_from_ends(fluid))


@descriptor("euler", params=("phase",))
def euler(field: Tensor, mask: Tensor, *, phase: Phase = "nw") -> Tensor:
    """Euler characteristic of the `phase` region: components − loops (+ cavities in 3-D).

    Counted on the cubical complex whose m-cells are the elementary 2^m voxel blocks
    (2 adjacent along each of m axes) lying entirely inside the phase, so connectivity
    is 6-connected like `trapped_volume` and the solver's ganglia — a diagonal touch joins
    nothing. `chi = sum_m (-1)^m N_m`; 0 for an empty phase.
    """
    _check_single_channel(field, mask)
    x = phase_mask(field, mask, phase)
    chi = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
    for m in range(x.ndim - 1):
        for axes in combinations(range(2, x.ndim), m):
            cells = x
            for axis in axes:
                cells = cells.narrow(axis, 0, cells.shape[axis] - 1) & cells.narrow(axis, 1, cells.shape[axis] - 1)
            chi += (-1) ** m * _sum_per_sample(cells)
    return chi


def _curvature_sign(phase: Phase) -> float:
    return 1.0 if phase == "nw" else -1.0


# One interface mesh per (field, side) however many descriptors read it. `compute` opens
# the context; outside it every descriptor call builds its own mesh, as before.
_MESHES: ContextVar[dict | None] = ContextVar("poreml_shared_meshes", default=None)


@contextmanager
def shared_meshes() -> Iterator[None]:
    """Share interface meshes and curvatures between descriptor calls made inside."""
    token = _MESHES.set({})
    try:
        yield
    finally:
        _MESHES.reset(token)


def _mesh_key(field: Tensor, phase: Phase, params: dict[str, Any]) -> tuple:
    return (field.data_ptr(), tuple(field.shape), str(field.device), phase, tuple(sorted(params.items())))


def _curvatures(field: Tensor, mask: Tensor, phase: Phase, params: dict[str, Any]) -> list:
    """`cv.batch_interface_curvature` through the shared cache, upgrading a cached mesh in place."""
    from . import curvature as cv

    cache = _MESHES.get()
    key = _mesh_key(field, phase, params)
    if cache is not None and ("curvature", key) in cache:
        out = cache[("curvature", key)]
        _record_mesh_stats(out)
        return out
    meshes = cache.get(("mesh", key)) if cache is not None else None
    if meshes is None:
        out = cv.batch_interface_curvature(field, mask, sign=_curvature_sign(phase), **params)
    else:
        fit = {"radius": params["radius"]} if "radius" in params else {}
        out = [cv.curvature_of_mesh(m, **fit) for m in meshes]
    if cache is not None:
        cache[("curvature", key)] = out
    _record_mesh_stats(out)
    return out


def _meshes(field: Tensor, mask: Tensor, phase: Phase, params: dict[str, Any]) -> list:
    """`cv.batch_interface_mesh` through the shared cache; a cached curvature object is a mesh too."""
    from . import curvature as cv

    cache = _MESHES.get()
    key = _mesh_key(field, phase, params)
    if cache is not None:
        for kind in ("curvature", "mesh"):
            if (kind, key) in cache:
                out = cache[(kind, key)]
                _record_mesh_stats(out)
                return out
    out = cv.batch_interface_mesh(field, mask, sign=_curvature_sign(phase), **params)
    if cache is not None:
        cache[("mesh", key)] = out
    _record_mesh_stats(out)
    return out


def _record_mesh_stats(meshes: list) -> None:
    """What every mesh descriptor has in hand: vertex counts and the three areas, plus the
    scored vertices and area when curvatures were fitted."""
    from . import curvature as cv

    record("n_vertices", torch.tensor([float(len(m.vertices)) for m in meshes]))
    record("n_components_dropped", torch.tensor([float(m.dropped_components) for m in meshes]))
    for part, name in (("all", "area"), ("fluid", "area_fluid"), ("solid", "area_solid")):
        record(name, cv.area_of(meshes, part))
    if meshes and isinstance(meshes[0], cv.InterfaceCurvature):
        record("n_scored", torch.tensor([float(m.valid.sum()) for m in meshes]))
        record("scored_area", torch.tensor([float(m.vertex_areas[m.valid].sum()) for m in meshes], dtype=torch.float32))


# Smoothing iterations and fit radius are not spec parameters: they are calibrated on
# analytic surfaces (tests/test_curvature.py) and changing them changes what the metric
# computes — that is a METRICS_VERSION bump, not a config knob. They stay as keyword
# arguments so the calibration can vary them.
CURVATURE_PARAMS = ("phase",)


def _curvature(field: Tensor, mask: Tensor, quantity: str, phase: Phase, iterations: int | None, radius: int | None) -> Tensor:
    from . import curvature as cv

    _check_single_channel(field, mask)
    params = {k: v for k, v in dict(iterations=iterations, radius=radius).items() if v is not None}
    return cv.mean_of(_curvatures(field, mask, phase, params), quantity)


@descriptor("curvature_mean", params=CURVATURE_PARAMS)
def curvature_mean(
    field: Tensor, mask: Tensor, *, phase: Phase = "nw", iterations: int | None = None, radius: int | None = None
) -> Tensor:
    """Mean curvature `H = (k1 + k2) / 2` [1/voxel] averaged over the fluid–fluid interface.

    Marching cubes → Taubin smoothing → libigl quadric fit (`metrics.surface`,
    `metrics.curvature`). Positive when
    `phase` is the droplet: `Pc = 2σH`. Vertices on the contact patch and contact line
    are excluded; NaN when there is no fluid–fluid interface. Needs the `curvature`
    dependency group.
    """
    return _curvature(field, mask, "mean", phase, iterations, radius)


@descriptor("curvature_hist", kind=DISTRIBUTION_KIND, params=CURVATURE_PARAMS + ("bins", "range"))
def curvature_hist(
    field: Tensor,
    mask: Tensor,
    *,
    bins: int,
    range: tuple[float, float],
    phase: Phase = "nw",
    iterations: int | None = None,
    radius: int | None = None,
) -> Tensor:
    """Normalised histogram `(B, bins)` of mean curvature `H` [1/voxel] over the fluid–fluid interface.

    The distribution behind `curvature_mean`: on a drainage front every meniscus has its
    own `H`, and the spread carries the pore-size sampling that a mean hides. Score with
    `w1` (1/voxel) and `w1_norm` (W1 / σ_target), or record it with `pred` / `target`.
    Same vertex selection and sign as `curvature_mean`; NaN row with no interface.
    """
    from . import curvature as cv

    _check_single_channel(field, mask)
    params = {k: v for k, v in dict(iterations=iterations, radius=radius).items() if v is not None}
    return cv.histogram_of(_curvatures(field, mask, phase, params), "mean", bins=bins, range=range)


@descriptor("curvature_gauss", params=CURVATURE_PARAMS)
def curvature_gauss(
    field: Tensor, mask: Tensor, *, phase: Phase = "nw", iterations: int | None = None, radius: int | None = None
) -> Tensor:
    """Gaussian curvature `K = k1·k2` [1/voxel²] averaged over the fluid–fluid interface; see `curvature_mean`."""
    return _curvature(field, mask, "gaussian", phase, iterations, radius)


# --- Minkowski functionals on the mesh ---------------------------------------------
# M1 = ∫ dS, M2 = ∫ H dS, M3 = ∫ K dS (Mecke; Armstrong et al. 2019 for two-phase flow).
# M1 comes from the smoothed mesh, so it has none of the 3/2 staircase inflation of the
# voxel-face `area` trio it mirrors; M2 and M3 are the per-vertex curvatures weighted by
# barycentric vertex areas over the fluid–fluid interface that `curvature_mean` scores.


def _mesh_area(field: Tensor, mask: Tensor, part: str, phase: Phase, iterations: int | None) -> Tensor:
    from . import curvature as cv

    _check_single_channel(field, mask)
    params = {} if iterations is None else {"iterations": iterations}
    return cv.area_of(_meshes(field, mask, phase, params), part)


@descriptor("mesh_area", params=("phase",))
def mesh_area(field: Tensor, mask: Tensor, *, phase: Phase = "nw", iterations: int | None = None) -> Tensor:
    """Surface area [voxel²] of `phase` on the smoothed interface mesh — Minkowski M1.

    `mesh_area_interface + mesh_area_contact(phase)`: the boundary of the phase region as
    marching cubes → Taubin smoothing draws it (`metrics.surface`), so a sphere reads
    4πr² where the voxel-face `area` reads 3/2 of that. Closed against solid, open where
    the phase leaves the domain box: inlet, outlet and buffer planes never count. 0 with
    no surface. 3-D only; needs the `curvature` group (scikit-image).
    """
    return _mesh_area(field, mask, "all", phase, iterations)


@descriptor("mesh_area_interface")
def mesh_area_interface(field: Tensor, mask: Tensor, *, iterations: int | None = None) -> Tensor:
    """Fluid–fluid interfacial area [voxel²] on the smoothed mesh; the same from either side (see `mesh_area`)."""
    return _mesh_area(field, mask, "fluid", "nw", iterations)


@descriptor("mesh_area_contact", params=("phase",))
def mesh_area_contact(field: Tensor, mask: Tensor, *, phase: Phase = "nw", iterations: int | None = None) -> Tensor:
    """Fluid–solid contact area [voxel²] of `phase` on the smoothed mesh (see `mesh_area`)."""
    return _mesh_area(field, mask, "solid", phase, iterations)


def _curvature_integral(
    field: Tensor, mask: Tensor, quantity: str, phase: Phase, iterations: int | None, radius: int | None, normalised: bool
) -> Tensor:
    from . import curvature as cv

    _check_single_channel(field, mask)
    params = {k: v for k, v in dict(iterations=iterations, radius=radius).items() if v is not None}
    curvatures = _curvatures(field, mask, phase, params)
    integral = cv.integral_of(curvatures, quantity)
    record(f"integral_{quantity}", integral)
    return cv.weighted_mean_of(curvatures, quantity) if normalised else integral


@descriptor("curvature_mean_integral", params=CURVATURE_PARAMS)
def curvature_mean_integral(
    field: Tensor, mask: Tensor, *, phase: Phase = "nw", iterations: int | None = None, radius: int | None = None
) -> Tensor:
    """`∫ H dS` [voxel] over the fluid–fluid interface — Minkowski M2.

    Per-vertex mean curvature (`curvature_mean`'s pipeline and sign: a droplet of `phase`
    is positive) times barycentric vertex areas, summed over the vertices the fit trusts —
    on the fluid–fluid interface and outside the contact-line margin, so on a phase
    touching solid the integral covers slightly less than `mesh_area_interface`. A sphere
    reads 4πr. NaN with no fluid–fluid interface.
    """
    return _curvature_integral(field, mask, "mean", phase, iterations, radius, normalised=False)


@descriptor("curvature_mean_integral_norm", params=CURVATURE_PARAMS)
def curvature_mean_integral_norm(
    field: Tensor, mask: Tensor, *, phase: Phase = "nw", iterations: int | None = None, radius: int | None = None
) -> Tensor:
    """`∫ H dS / ∫ dS` [1/voxel] over the scored fluid–fluid interface — M2 per unit area.

    The area-weighted mean curvature: `curvature_mean` weights every vertex equally, this
    weights by the surface each vertex represents, so it is what `Pc = 2σ⟨H⟩` wants of a
    front made of menisci of different sizes. Both integrals run over the same vertices,
    so a sphere reads exactly 1/r whatever the margin removes. NaN with no interface.
    """
    return _curvature_integral(field, mask, "mean", phase, iterations, radius, normalised=True)


@descriptor("curvature_gauss_integral", params=CURVATURE_PARAMS)
def curvature_gauss_integral(
    field: Tensor, mask: Tensor, *, phase: Phase = "nw", iterations: int | None = None, radius: int | None = None
) -> Tensor:
    """`∫ K dS` [dimensionless] over the fluid–fluid interface — Minkowski M3.

    Gauss–Bonnet makes it 2πχ on a closed surface (4π per isolated droplet, 0 for a
    torus), the mesh counterpart of the voxel `euler` descriptor — on a surface cut by
    the domain box or the contact-line margin the geodesic boundary term is missing, so
    read it as a topology proxy, not a count. Same vertices as `curvature_mean_integral`.
    """
    return _curvature_integral(field, mask, "gaussian", phase, iterations, radius, normalised=False)
