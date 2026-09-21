"""Interfacial curvature from a voxel phase field.

Marching cubes → Taubin smoothing (`metrics.surface`) → a local quadric fit at every
vertex, libigl's `principal_curvature`: the same fit as `ax² + by² + cz² + 2exy + 2fyz +
2gzx + 2lx + 2my + 2nz + d = 0` in the CT-curvature literature, whose eigen-decomposition
gives the principal curvatures `k1 ≥ k2`. Mean curvature is `H = (k1 + k2) / 2`,
Gaussian `K = k1·k2`, both per voxel.

Sign convention: a droplet of `phase` has `H = +1/r`, so `Pc = P_phase − P_other = 2σH`.

Statistics use fluid–fluid vertices more than a fit radius away from the contact line.
On the contact patch and the contact line a fluid–fluid curvature is neither defined nor
recoverable at voxel resolution; that is where the CT best-practice studies place their
largest errors.
"""

import logging
import struct
import subprocess
import sys
from functools import cached_property

import numpy as np
import torch
from torch import Tensor

from . import surface

log = logging.getLogger("poreml")

# The quadric fit runs in a worker subprocess: libigl is C++ and a degenerate mesh once
# segfaulted it — a metric must cost at worst a NaN, never the training process. The
# worker imports only numpy and igl, is started lazily, restarted after a crash, and
# speaks a fixed-size binary protocol over its pipes.
_WORKER_SOURCE = """
import struct, sys
import numpy as np
import igl

inp, out = sys.stdin.buffer, sys.stdout.buffer
while True:
    header = inp.read(24)
    if len(header) < 24:
        break
    radius, nv, nf = struct.unpack("<qqq", header)
    v = np.frombuffer(inp.read(nv * 24), np.float64).reshape(nv, 3).copy()
    f = np.frombuffer(inp.read(nf * 24), np.int64).reshape(nf, 3).copy()
    _, _, k1, k2, bad = igl.principal_curvature(v, f, radius, True)
    k1 = np.asarray(k1, dtype=np.float64).copy()
    k2 = np.asarray(k2, dtype=np.float64).copy()
    if len(bad):
        k1[bad] = np.nan
        k2[bad] = np.nan
    out.write(k1.tobytes())
    out.write(k2.tobytes())
    out.flush()
"""


class _IglWorker:
    """`igl.principal_curvature` behind a pipe; a crash in it returns None instead of a signal."""

    def __init__(self, source: str = _WORKER_SOURCE) -> None:
        self.source = source
        self.proc: subprocess.Popen | None = None

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", self.source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def __call__(self, vertices: np.ndarray, faces: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        n = len(vertices)
        try:
            self.proc.stdin.write(struct.pack("<qqq", radius, n, len(faces)))
            self.proc.stdin.write(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
            self.proc.stdin.write(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
            self.proc.stdin.flush()
            payload = self.proc.stdout.read(2 * n * 8)
        except (BrokenPipeError, OSError):
            payload = b""
        if len(payload) < 2 * n * 8:  # the worker died mid-job
            self.proc.kill()
            self.proc.wait()
            self.proc = None
            return None
        flat = np.frombuffer(payload, dtype=np.float64)
        return flat[:n].copy(), flat[n:].copy()


_WORKER = _IglWorker()

# Rings of neighbouring vertices feeding each quadric fit (libigl `radius` with k-ring
# on); the CT workflow this follows used 4 neighbouring triangles.
DEFAULT_RADIUS = 4


class InterfaceCurvature(surface.InterfaceMesh):
    """An `InterfaceMesh` with principal curvatures per vertex (NaN where libigl could not fit).

    `valid` marks the vertices a statistic should use: finite, on the fluid–fluid
    interface, and more than `radius` rings from the contact line — closer in, the fit
    straddles the corner onto the contact patch and reads the wall's curvature.
    """

    def __init__(self, mesh: surface.InterfaceMesh, k1: np.ndarray, k2: np.ndarray, radius: int):
        super().__init__(mesh.vertices, mesh.faces, mesh.near_solid, mesh.dropped_components)
        self.k1 = k1
        self.k2 = k2
        self.radius = radius

    @property
    def valid(self) -> np.ndarray:
        margin = surface.within_rings(self.faces, self.near_solid, self.radius)
        return ~margin & np.isfinite(self.k1) & np.isfinite(self.k2)

    @property
    def mean(self) -> np.ndarray:
        return 0.5 * (self.k1 + self.k2)

    @property
    def gaussian(self) -> np.ndarray:
        return self.k1 * self.k2

    @cached_property
    def vertex_areas(self) -> np.ndarray:
        """`(N,)` barycentric vertex areas (`surface.vertex_areas`), cached: the mesh is immutable."""
        return surface.vertex_areas(self.vertices, self.faces)


def principal_curvatures(
    vertices: np.ndarray, faces: np.ndarray, *, radius: int = DEFAULT_RADIUS
) -> tuple[np.ndarray, np.ndarray]:
    """`(k1, k2)` per vertex from libigl's quadric fit over the `radius`-ring neighbourhood.

    Signed with respect to the mesh normals from `surface.marching_cubes` (toward the
    negative side); `interface_curvature` applies the phase convention. The fit runs in a
    worker subprocess (`_IglWorker`): if libigl crashes, this returns all-NaN — counted as
    `n_invalid` downstream — instead of killing the caller.
    """
    if len(faces) == 0:
        empty = np.zeros(len(vertices), dtype=np.float64)
        return empty, empty
    try:
        import igl  # noqa: F401  (fail here with the install hint, not silently in the worker)
    except ImportError as exc:  # pragma: no cover - only without the group
        raise ImportError(surface._MISSING) from exc
    result = _WORKER(vertices, faces, radius)
    if result is None:
        log.warning("libigl principal_curvature crashed on a %d-vertex mesh; curvature is NaN for this sample", len(vertices))
        nan = np.full(len(vertices), np.nan, dtype=np.float64)
        return nan, nan.copy()
    return result


def curvature_of_mesh(mesh: surface.InterfaceMesh, *, radius: int = DEFAULT_RADIUS) -> InterfaceCurvature:
    """Fit principal curvatures on an existing interface mesh."""
    k1, k2 = principal_curvatures(mesh.vertices, mesh.faces, radius=radius)
    # libigl measures along the marching-cubes normal, which points into the negative
    # side; negate so a droplet of the positive phase reads +1/r, restoring k1 >= k2.
    return InterfaceCurvature(mesh, -k2, -k1, radius)


def interface_curvature(
    field: np.ndarray, solid: np.ndarray, *, radius: int = DEFAULT_RADIUS, **mesh_params
) -> InterfaceCurvature:
    """The whole pipeline for one `(D, H, W)` field whose positive side is the droplet phase.

    `mesh_params` go to `surface.interface_mesh` (`level`, `iterations`, `lam`, `mu`).
    """
    return curvature_of_mesh(surface.interface_mesh(field, solid, **mesh_params), radius=radius)


def _batch_numpy(field: Tensor, mask: Tensor, sign: float) -> tuple[np.ndarray, np.ndarray]:
    if field.ndim != 5 or field.shape[1] != 1:
        raise ValueError(f"interface meshes need a (batch, 1, D, H, W) field; got shape {tuple(field.shape)}")
    phi = (field * sign).detach().cpu().numpy()[:, 0]
    solid = ~mask.detach().cpu().numpy()[:, 0]
    return phi, solid


def batch_interface_curvature(field: Tensor, mask: Tensor, *, sign: float = 1.0, **params) -> list[InterfaceCurvature]:
    """`interface_curvature` per sample of a `(B, 1, D, H, W)` field with a `(B, 1, D, H, W)` pore mask.

    `sign` multiplies the field first: -1 makes the wetting phase the droplet.
    """
    phi, solid = _batch_numpy(field, mask, sign)
    return [interface_curvature(p, s, **params) for p, s in zip(phi, solid, strict=True)]


def batch_interface_mesh(field: Tensor, mask: Tensor, *, sign: float = 1.0, **params) -> list[surface.InterfaceMesh]:
    """`surface.interface_mesh` per sample — the geometry without the quadric fit (no libigl)."""
    phi, solid = _batch_numpy(field, mask, sign)
    return [surface.interface_mesh(p, s, **params) for p, s in zip(phi, solid, strict=True)]


def area_of(meshes: list[surface.InterfaceMesh], part: str = "all") -> Tensor:
    """`(B,)` float32: `InterfaceMesh.area(part)` per sample; 0 with no surface."""
    return torch.tensor([m.area(part) for m in meshes], dtype=torch.float32)


def integral_of(curvatures: list[InterfaceCurvature], quantity: str) -> Tensor:
    """`(B,)` float32: `Σ q_v A_v` of `quantity` ("mean" | "gaussian") over valid vertices; NaN with none.

    The discrete `∫ q dS` over the scored fluid–fluid interface: with `q = H` the second
    Minkowski functional M2, with `q = K` the third, M3 = 2πχ by Gauss–Bonnet on a closed
    surface. Vertices in the contact-line margin are excluded, as in `mean_of`, so on a
    phase touching solid the integral covers a little less than `area("fluid")`.
    """
    out = []
    for c in curvatures:
        keep = c.valid
        out.append(float((getattr(c, quantity)[keep] * c.vertex_areas[keep]).sum()) if keep.any() else float("nan"))
    return torch.tensor(out, dtype=torch.float32)


def weighted_mean_of(curvatures: list[InterfaceCurvature], quantity: str) -> Tensor:
    """`(B,)` float32: `Σ q_v A_v / Σ A_v` over valid vertices — the integral per unit scored area; NaN with none."""
    out = []
    for c in curvatures:
        keep = c.valid
        weights = c.vertex_areas[keep]
        total = weights.sum()
        out.append(float((getattr(c, quantity)[keep] * weights).sum() / total) if total > 0 else float("nan"))
    return torch.tensor(out, dtype=torch.float32)


def histogram_of(curvatures: list[InterfaceCurvature], quantity: str, *, bins: int, range: tuple[float, float]) -> Tensor:
    """`(B, bins)` float32: normalised histogram of `quantity` over each sample's valid vertices; NaN row with none.

    Same binning contract as the `hist` descriptor: values outside `range` land in the
    end bins so nothing is dropped when a prediction overshoots.
    """
    lo, hi = range
    if not hi > lo:
        raise ValueError(f"histogram range must satisfy lo < hi; got {range}")
    edges = np.linspace(lo, hi, bins + 1)
    out = np.full((len(curvatures), bins), np.nan, dtype=np.float32)
    for i, c in enumerate(curvatures):
        values = getattr(c, quantity)[c.valid]
        if len(values):
            counts, _ = np.histogram(np.clip(values, lo, hi), bins=edges)
            out[i] = counts / counts.sum()
    return torch.from_numpy(out)


def mean_of(curvatures: list[InterfaceCurvature], quantity: str) -> Tensor:
    """`(B,)` float32: mean of `quantity` ("mean" | "gaussian") over fluid–fluid vertices; NaN with none."""
    out = []
    for c in curvatures:
        values = getattr(c, quantity)[c.valid]
        out.append(float(values.mean()) if len(values) else float("nan"))
    return torch.tensor(out, dtype=torch.float32)
