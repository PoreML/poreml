"""Shared 3-D rendering for the visual validation figures (topology, curvature).

matplotlib only Lambert-shades a 3-D collection when the colors go through the shaded
parameters — `facecolors=` on `voxels`, `color=` on `plot_trisurf`. A bare `facecolor=`
kwarg falls through to `Poly3DCollection` unshaded and renders every face the same flat
fill. These helpers route everything through the shaded path, add a short-focal-length
perspective as a second depth cue, and shade value-colored meshes by hand — there the
colorbar must keep the unshaded values, because sequential maps carry magnitude in
lightness and full shading would rewrite them.
"""

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

VOXEL_COLOR = "#8ab0d6"
MESH_COLOR = "#c9ced6"
EDGE_COLOR = (0.10, 0.16, 0.26, 0.30)  # darkened fill, not black: outlines that don't flatten
LIGHT = LightSource(azdeg=225, altdeg=20)
# For half-cut solids viewed facing the section plane (elev 22, azim -20): light from the
# viewer's quadrant, or the section sits in shadow and the interior reads as a dark mass.
SECTION_LIGHT = LightSource(azdeg=0, altdeg=35)
FOCAL_LENGTH = 0.3


def _style(ax, spans) -> None:
    ax.set_proj_type("persp", focal_length=FOCAL_LENGTH)
    ax.set_box_aspect(spans)
    ax.set_axis_off()


def voxel_panel(
    ax, blob: np.ndarray, *, color: str = VOXEL_COLOR, linewidth: float = 0.2, light: LightSource | None = None
) -> None:
    """Shaded voxel solid for a boolean `(z, y, x)` array, axes pinned to the full array
    span — `voxels` alone tightens limits to the occupied bounding box, which stretches
    a slab into a cube under a (1, 1, 1) box aspect."""
    ax.voxels(
        np.transpose(blob, (2, 1, 0)),
        facecolors=color,
        edgecolors=EDGE_COLOR,
        linewidth=linewidth,
        shade=True,
        lightsource=light or LIGHT,
    )
    n = max(blob.shape)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    _style(ax, (1, 1, 1))


def mesh_panel(ax, verts: np.ndarray, faces: np.ndarray, *, color: str = MESH_COLOR, cube_aspect: bool = False) -> None:
    """Shaded triangle mesh for `(z, y, x)`-ordered vertices (the array axis order)."""
    zz, yy, xx = verts.T
    ax.plot_trisurf(xx, yy, zz, triangles=faces, color=color, edgecolor="none", shade=True, lightsource=LIGHT)
    _style(ax, (1, 1, 1) if cube_aspect else (np.ptp(xx), np.ptp(yy), np.ptp(zz)))


def value_mesh_panel(
    ax, verts: np.ndarray, faces: np.ndarray, values: np.ndarray, *, cmap: str, clim: tuple[float, float], ambient: float = 0.55
) -> ScalarMappable:
    """Triangle mesh colored per-face by `values` with gentle Lambert relief on top.

    `ambient` floors the shading factor so the relief reads as shape, not as data.
    Returns the mappable of the unshaded values for the caller's colorbar.
    """
    norm = Normalize(*clim)
    rgba = colormaps[cmap](norm(values))  # NaN faces map to the colormap's transparent bad color
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    intensity = np.abs(normals[:, ::-1] @ LIGHT.direction)  # (z, y, x) -> (x, y, z); either orientation lit
    rgba[:, :3] *= (ambient + (1.0 - ambient) * intensity)[:, None]
    ax.add_collection3d(Poly3DCollection(tri[:, :, ::-1], facecolors=rgba, edgecolors="none"))
    zz, yy, xx = verts.T
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_zlim(zz.min(), zz.max())
    _style(ax, (np.ptp(xx), np.ptp(yy), np.ptp(zz)))
    return ScalarMappable(norm=norm, cmap=cmap)
