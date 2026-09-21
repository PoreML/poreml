"""Renders in the solver's own style, so a prediction can sit beside the simulation it imitates.

Red non-wetting phase opaque, rock translucent grey, flow left → right, voxel threshold
with no smoothing (chunky but honest — nothing fakes a thin interface). Rendering runs
in a **subprocess**: this process usually holds a CUDA context, and a headless EGL
context created beside one renders black. One worker process renders a whole batch of
frames, so the ~5 s of VTK start-up is paid once per sequence, not once per frame.

Only `render_jobs`, `write_gif`, `plot_rollout_curves`, `plot_training_curves`,
`plot_eval_curves` and `plot_progress` need the optional `viz` dependency group; importing this module
without it still works, calling them raises with the install hint. The worker entry point
at the bottom imports nothing from `poreml`, so the subprocess never touches torch.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RED = "#d62728"
ROCK = "#8a8a8a"
INSTALL_HINT = "rendering needs the optional dependencies: uv sync --group viz"


@dataclass
class Panel:
    """One rendered view: which voxels are red, and the label drawn top-left."""

    red: np.ndarray  # bool (D, H, W); True where the non-wetting phase is
    label: str = ""


@dataclass
class RenderJob:
    """One output image made of `panels` side by side, sharing the job's solid mask."""

    path: Path
    panels: list[Panel] = field(default_factory=list)


def flow_camera(shape: Sequence[int], side: float = 0.75, dist: float = 2.1, height: float = 0.85) -> list:
    """the solver's camera for flow along the last axis: the invasion front marches left → right
    on screen, seen from a 3/4 look-down angle. `shape` is (D, H, W) with W the flow axis."""
    nz, ny, nx = shape
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    span = max(nx, ny, nz)
    pos = (center[0] + side * span, center[1] - dist * span, center[2] + height * span)
    return [list(pos), list(center), [0.0, 0.0, 1.0]]


def render_jobs(
    jobs: Sequence[RenderJob], solid: np.ndarray, window: tuple[int, int] = (640, 480), rock_opacity: float = 0.06
) -> None:
    """Render every job to its `path` in one worker subprocess."""
    _require_viz()
    if not jobs:
        return
    with tempfile.TemporaryDirectory() as td:
        payload = Path(td) / "payload.npz"
        arrays = {"solid": np.asarray(solid, dtype=bool)}
        spec = []
        for j, job in enumerate(jobs):
            names = []
            for k, panel in enumerate(job.panels):
                name = f"red_{j}_{k}"
                arrays[name] = np.asarray(panel.red, dtype=bool)
                names.append(name)
            Path(job.path).parent.mkdir(parents=True, exist_ok=True)
            spec.append({"path": str(job.path), "arrays": names, "labels": [p.label for p in job.panels]})
        np.savez_compressed(
            payload, jobs=json.dumps({"jobs": spec, "window": list(window), "rock_opacity": rock_opacity}), **arrays
        )
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(payload)],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"render worker failed:\n{proc.stderr[-4000:]}")


def write_gif(frame_paths: Sequence[Path], out: Path, fps: int = 10, max_frames: int = 300) -> Path:
    """Animate PNG frames into a GIF (subsampled evenly past `max_frames`, like the solver)."""
    _require_viz()
    from PIL import Image

    paths = list(frame_paths)
    if len(paths) > max_frames:
        idx = np.unique(np.linspace(0, len(paths) - 1, max_frames).round().astype(int))
        paths = [paths[i] for i in idx]
    frames = [Image.open(p).convert("RGB") for p in paths]
    out = Path(out)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return out


def _metric_rows(names: Sequence[str], max_cols: int = 4) -> list[list[str]]:
    """Panels arranged as a matrix: one row per metric group, wrapped at `max_cols`.

    A group is the channel after '@' (ux/uy/uz collapse to one u group), split into
    plain voxel errors and descriptor metrics ('/' in the name) — so `mae@phi,
    rel_mae@phi, iou@phi` is one row, the phi descriptors the next, `mae@p` and the
    `mae@u*` components their own. Names without '@' (train_loss, fixture metrics)
    share one group. Groups keep the order the metrics were configured in.
    """
    groups: dict[tuple[str, bool], list[str]] = {}
    for name in names:
        base, _, channel = name.partition("@")
        channel = "u" if channel in {"ux", "uy", "uz"} else channel
        groups.setdefault((channel, "/" in base), []).append(name)
    rows: list[list[str]] = []
    for members in groups.values():
        rows += [members[i : i + max_cols] for i in range(0, len(members), max_cols)]
    return rows


def _panel_matrix(plt, names: Sequence[str], panel_w: float, panel_h: float) -> tuple:
    """A figure with one axis per name, laid out by `_metric_rows`; unused cells hidden.

    Returns `(fig, {name: axis})` in the order of `names`' groups.
    """
    rows = _metric_rows(names)
    ncols = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(panel_w * ncols, panel_h * len(rows)), squeeze=False)
    panels: dict[str, object] = {}
    for r, row in enumerate(rows):
        for c, ax in enumerate(axes[r]):
            if c < len(row):
                panels[row[c]] = ax
            else:
                ax.set_visible(False)
    return fig, panels


def plot_rollout_curves(sources: dict[str, Path | dict], metrics: Sequence[str], out: Path) -> Path:
    """Metric versus horizon step, one panel per metric, one line per model.

    `sources` maps a legend label to a `rollout_<split>.json` path (or an already-loaded
    payload); the curves are the `summary` means over runs. Panels form a matrix with
    one row per metric group (see `_metric_rows`), not one long row.
    """
    _require_viz()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded = {
        label: (json.loads(Path(src).read_text()) if not isinstance(src, dict) else src) for label, src in sources.items()
    }
    fig, panels = _panel_matrix(plt, list(metrics), 4.2, 3.4)
    for metric, ax in panels.items():
        for label, payload in loaded.items():
            curve = payload["summary"][metric]
            ax.plot(range(1, len(curve) + 1), curve, label=label, lw=1.6)
        ax.set_xlabel("rollout step (saved frames)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    next(iter(panels.values())).legend(frameon=False)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_metric_curve(sources: dict[str, tuple[Sequence[int], Sequence[float]]], metric: str, out: Path) -> Path:
    """One metric against rollout step, one line per label, markers at the stored steps."""
    _require_viz()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for label, (steps, values) in sources.items():
        ax.plot(list(steps), [float("nan") if v is None else v for v in values], marker="o", ms=3, lw=1.6, label=label)
    ax.set_xlabel("rollout step (saved frames)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_training_curves(metrics_csv: Path, out: Path, columns: Sequence[str] | None = None) -> Path:
    """Training loss and validation metrics versus epoch from a run's `metrics.csv`.

    `columns` picks the validation metrics to show (default: every numeric column that is
    not bookkeeping); `train_loss` always gets the first panel.
    """
    _require_viz()
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(metrics_csv).open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{metrics_csv} has no rows")
    skip = {"epoch", "step", "epoch_seconds", "peak_memory_mb", "lr", "train_loss"}
    if columns is None:
        columns = [c for c in rows[0] if c not in skip and "[" not in c and _any_row_numeric(rows, c)]
    epochs = [int(r["epoch"]) for r in rows]
    fig, panels = _panel_matrix(plt, ["train_loss", *columns], 3.8, 3.2)
    for name, ax in panels.items():
        ys = [float(r[name]) if r[name] != "" else float("nan") for r in rows]
        ax.plot(epochs, ys, marker="o", lw=1.4)
        ax.set_xlabel("epoch")
        ax.set_ylabel(name if name != "train_loss" else "train loss (masked MSE)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_eval_curves(metrics_csv: Path, out: Path, columns: Sequence[str] | None = None) -> Path:
    """Periodic-evaluation metrics versus epoch, from a run's `eval/metrics.csv`.

    One panel per scalar metric: the model solid and the rollout value at the horizon
    (`rollout/<m>`) dashed, so a single figure says whether training is improving
    single-step *and* rollout accuracy.
    """
    _require_viz()
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(metrics_csv).open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{metrics_csv} has no rows")
    skip = {"epoch", "step"}
    if columns is None:
        columns = [
            c for c in rows[0] if c not in skip and "[" not in c and not c.startswith("rollout/") and _any_row_numeric(rows, c)
        ]
    if not columns:
        raise ValueError(f"{metrics_csv} has no scalar metric columns to plot")
    epochs = [int(r["epoch"]) for r in rows]

    def series(name: str) -> list[float] | None:
        if name not in rows[0]:
            return None
        return [float(r[name]) if r[name] != "" else float("nan") for r in rows]

    fig, panels = _panel_matrix(plt, list(columns), 3.8, 3.2)
    for name, ax in panels.items():
        for key, label, style in (
            (name, "model", dict(color="C0", ls="-")),
            (f"rollout/{name}", "model, rollout @h", dict(color="C0", ls="--")),
        ):
            ys = series(key)
            if ys is not None:
                ax.plot(epochs, ys, marker="o", lw=1.4, label=label, **style)
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    next(iter(panels.values())).legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_progress(metrics_csv: Path, eval_csv: Path, out: Path, metric: str) -> Path:
    """`progress.svg` of a run: the one-step `metric` per epoch (`metrics.csv`, left) and its
    rollout twin `rollout/<metric>` at the horizon per periodic eval (`eval/metrics.csv`, right).

    Redrawn every epoch during training, so the rollout panel is empty until the first
    periodic eval and `eval_csv` may not exist yet; the lowest rollout point — the epoch
    `finalise.py` would pick now — is marked.
    """
    _require_viz()
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def read(path: Path, column: str) -> tuple[list[int], list[float]]:
        path = Path(path)
        if not path.is_file():
            return [], []
        with path.open() as f:
            rows = [r for r in csv.DictReader(f) if _numeric(r.get("epoch")) and _numeric(r.get(column))]
        return [int(float(r["epoch"])) for r in rows], [float(r[column]) for r in rows]

    one_x, one_y = read(metrics_csv, metric)
    roll_x, roll_y = read(eval_csv, f"rollout/{metric}")
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    left.plot(one_x, one_y, marker="o", lw=1.4, color="C0")
    left.set_title(f"one-step {metric} (val, per epoch)", fontsize=9)
    right.plot(roll_x, roll_y, marker="o", lw=1.4, color="C0", ls="--")
    if roll_y:
        i = min(range(len(roll_y)), key=roll_y.__getitem__)
        right.plot([roll_x[i]], [roll_y[i]], marker="*", ms=12, color="C3", ls="none", label=f"rollout-best: epoch {roll_x[i]}")
        right.legend(frameon=False, fontsize=8)
    else:
        right.text(0.5, 0.5, "no periodic rollout yet", ha="center", va="center", transform=right.transAxes, fontsize=9)
    right.set_title(f"rollout/{metric} at the horizon (val, per eval)", fontsize=9)
    for ax in (left, right):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _numeric(text: str | None) -> bool:
    """`False` for `None` too: a `DictReader` row shorter than the header pads with `None`,
    not `""`, and `float(None)` raises `TypeError`, not `ValueError`."""
    if text is None:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _any_row_numeric(rows: list[dict], name: str) -> bool:
    """A column counts as scalar if *any* row parses as numeric, not just the first: a
    metric that is NaN (blank) at the very first periodic evaluation — e.g. `w1_norm`
    with a zero-variance target — must not vanish from the plot for the whole run."""
    return any(_numeric(r.get(name)) for r in rows)


def _require_viz() -> None:
    try:
        import matplotlib  # noqa: F401
        import PIL  # noqa: F401
        import pyvista  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(INSTALL_HINT) from e


def require_viz() -> None:
    """Public check for callers outside this module (`train.py`'s fail-fast guard) that
    do not otherwise import anything from the viz group and should not reach into a
    private name to ask whether it is installed."""
    _require_viz()


# ----------------------------------------------------------------------------------------
# Worker: `python viz.py payload.npz` — imports only numpy, pyvista and PIL.


def _grid(red: np.ndarray, solid: np.ndarray):
    import pyvista as pv

    nz, ny, nx = red.shape
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))  # cell data ravels x fastest: C-order of (z, y, x)
    grid.cell_data["red"] = (red & ~solid).astype(np.float32).ravel()
    grid.cell_data["solid"] = solid.astype(np.float32).ravel()
    return grid


def _render_panel(
    red: np.ndarray, solid: np.ndarray, label: str, window: list[int], rock_opacity: float, camera: list
) -> np.ndarray:
    import pyvista as pv

    grid = _grid(red, solid)
    pl = pv.Plotter(off_screen=True, window_size=list(window))
    pl.add_mesh(grid.outline(), color="#444444")
    rock = grid.threshold(0.5, scalars="solid")
    if rock.n_cells:
        pl.add_mesh(rock, color=ROCK, opacity=rock_opacity, show_scalar_bar=False)
    phase = grid.threshold(0.5, scalars="red")
    if phase.n_cells:
        pl.add_mesh(phase, color=RED, show_scalar_bar=False)
    if label:
        pl.add_text(label, position="upper_left", font_size=11, color="#222222")
    pl.camera_position = [tuple(v) for v in camera]
    pl.reset_camera()
    pl.camera.zoom(1.25)
    shot = pl.screenshot(return_img=True)
    pl.close()
    return shot


def _worker(payload_path: str) -> None:
    from PIL import Image

    data = np.load(payload_path)
    spec = json.loads(str(data["jobs"]))
    solid = data["solid"]
    camera = flow_camera(solid.shape)
    for job in spec["jobs"]:
        shots = [
            _render_panel(data[name], solid, label, spec["window"], spec["rock_opacity"], camera)
            for name, label in zip(job["arrays"], job["labels"], strict=True)
        ]
        Image.fromarray(np.hstack(shots)).save(job["path"])


if __name__ == "__main__":  # pragma: no cover - exercised through render_jobs
    _worker(sys.argv[1])
