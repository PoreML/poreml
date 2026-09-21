"""Evaluation-only transfer studies: Geo-Shift (`case/shift`) and Scale-Up (`case/scale`).

A study scores trained checkpoints under a frozen protocol without retraining, in two
stages: the GPU inference stage stores keyframes of every rollout window
(`evaluate.inference_split`), the CPU metric stage scores them and aggregates
(`evaluate.metric_split`). Its configuration is three layers under `configs/<study>/`,
each shared along a different axis:

    configs/<study>/scheme.yaml                    one per study: the window protocol
                                                   (horizon, stride, keyframes) and the
                                                   metric list (criterion first)
    configs/<study>/<campaign>/data.yaml           one per campaign: root, campaign, frozen split
    configs/<study>/<campaign>/<model>_<kind>.yaml one per cell: the training case whose
                                                   checkpoint is scored, the two files above,
                                                   the out_dir under case/<study>/

`Cell.load` reads the three; `resolve_checkpoint` finds `ckpts/best.pt` of the newest
*finished* run of the training case; `derive` composes the scored config: the checkpoint's
own saved `config.yaml` (task, fields, conditions, model — exactly what the weights were
trained on; the migration-note rule) with the cell's data, the scheme's horizon, keyframes
and metrics, and the cell's name and out_dir swapped in through `Config.updated`. The
window stride is not a `Config` field: it is passed separately to `inference_split` as
`cell.scheme.window_stride`. Nothing else of the training config changes, so a study
number never depends on the current `configs/<campaign>/` files.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from .config import Config, DataConfig, _Strict
from .metrics import METRICS_VERSION
from .metrics.spec import MetricSpec

# Geometry sources, as splits/draw.py draws the training splits: generated vs micro-CT.
SOURCES = {
    "gen": ("blob", "poly", "sphere", "fiber"),
    "uCT": ("bentheimer", "buffberea", "castlegate", "gdl_ct", "gdl_ct_20", "gdl_ct_40"),
}
CAMPAIGN_FOLDERS = {"drainage": "drainage", "GDL": "gdl", "trapping": "trapping"}  # data.campaign -> folder name


class SchemeConfig(_Strict):
    """The study's protocol — shared by every campaign and model of the study."""

    horizon: int = Field(default=64, ge=1, description="Steps per rollout window (saved frames)")
    stride: int | None = Field(
        default=None, ge=1, description="Frames between window starts; None = horizon, so windows tile the run"
    )
    keyframes: int = Field(
        default=12, ge=2, description="Stored (and scored) steps per window: first, last, evenly spaced between"
    )
    render: int = Field(
        default=3,
        ge=0,
        description="Runs rendered per cell as truth | prediction GIFs by the GPU stage (one per rock family; 0 disables)",
    )
    metrics: list[MetricSpec] = Field(
        min_length=1, description="What the metric stage scores on every stored frame; first = criterion"
    )

    @property
    def window_stride(self) -> int:
        return self.horizon if self.stride is None else self.stride


class CheckpointSource(_Strict):
    train_case: Path = Field(description="Training case dir (case/train/<campaign>/<model>_<kind>); newest finished run")
    run: Path | None = Field(default=None, description="Pin a run directory or a .pt file instead (bypasses the finished rule)")
    which: Literal["best", "last", "best_rollout"] = Field(
        default="best", description="Which checkpoint of the run: ckpts/best.pt, last.pt or best_rollout.pt (finalise.py)"
    )


class CellConfig(_Strict):
    """One (study, campaign, model) cell: what is scored, on what, under which protocol, where to."""

    name: str
    data: Path = Field(description="The campaign's shared data file (DataConfig)")
    scheme: Path = Field(description="The study's shared scheme file (SchemeConfig)")
    checkpoint: CheckpointSource
    out_dir: Path = Field(description="Where results_test.json, rollout_test.json, frames and GIFs land")

    @classmethod
    def from_yaml(cls, path: str | Path) -> CellConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


@dataclass(frozen=True)
class Cell:
    path: Path
    config: CellConfig
    data: DataConfig
    scheme: SchemeConfig

    @classmethod
    def load(cls, path: str | Path) -> Cell:
        path = Path(path)
        config = CellConfig.from_yaml(path)
        data = DataConfig.model_validate(yaml.safe_load(Path(config.data).read_text()) or {})
        scheme = SchemeConfig.model_validate(yaml.safe_load(Path(config.scheme).read_text()) or {})
        return cls(path=path, config=config, data=data, scheme=scheme)

    @property
    def campaign_dir(self) -> Path:
        """`case/<study>/<campaign>/` — where the per-campaign report lands."""
        return self.config.out_dir.parent


def _status(meta: Path) -> str:
    return str(json.loads(meta.read_text()).get("status", "unknown"))


def resolve_checkpoint(cell: Cell) -> Path:
    """The checkpoint a cell scores.

    A pinned `checkpoint.run` wins (a run dir -> its `ckpts/<which>.pt`, a file -> itself).
    Otherwise the newest run under `<train_case>/ckpts/` whose `run_meta.json` says
    `finished`: a running or preempted run's `best.pt` still moves, so it is never picked.
    """
    source = cell.config.checkpoint
    if source.run is not None:
        return source.run if source.run.is_file() else source.run / "ckpts" / f"{source.which}.pt"
    ckpts = source.train_case / "ckpts"
    metas = sorted(ckpts.glob("*/run_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for meta in metas:
        if _status(meta) == "finished":
            return meta.parent / "ckpts" / f"{source.which}.pt"
    raise FileNotFoundError(f"{cell.path}: no finished run under {ckpts} ({len(metas)} runs); pin checkpoint.run to override")


def saved_config(ckpt: Path) -> Config:
    """The config the checkpoint was trained with: its run directory's config.yaml."""
    path = Path(ckpt).parent.parent / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; a checkpoint must be scored with the config.yaml saved beside it")
    return Config.from_yaml(path)


def derive(cell: Cell, ckpt: Path) -> Config:
    """The scored config: the checkpoint's saved config with the cell's data, the scheme's
    horizon, keyframes and metrics, and the cell's name and out_dir swapped in. The window
    stride is not a Config field: `run.py` passes `cell.scheme.window_stride` to the inference."""
    scheme = cell.scheme
    return saved_config(ckpt).updated(
        name=cell.config.name,
        data=cell.data.model_dump(mode="json"),
        rollout={"horizon": scheme.horizon, "keyframes": scheme.keyframes, "start": None, "start_fraction": 0.25},
        metrics=[m.model_dump(mode="json") for m in scheme.metrics],
        out_dir=str(cell.config.out_dir),
    )


# --- cells on disk ---------------------------------------------------------------------


def cell_paths(study: str, campaigns: Iterable[str] | None = None, repo: Path = Path(".")) -> list[Path]:
    """Every cell config of a study: configs/<study>/<campaign>/*.yaml (data.yaml is shared, not a cell)."""
    root = repo / "configs" / study
    folders = [root / c for c in campaigns] if campaigns else sorted(p for p in root.iterdir() if p.is_dir())
    return [p for folder in folders for p in sorted(folder.glob("*.yaml")) if p.name != "data.yaml"]


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def classify(cell: Cell) -> str:
    """'finished' | 'metric' (frames stored, CPU stage pending) | 'pending' (inference) | 'waiting' (no finished checkpoint).

    A cell is finished only when its metrics were scored for the stored frames' checkpoint
    *and* under the current `METRICS_VERSION`: bumping the version (a descriptor changed what
    it computes) puts every scored cell back in the CPU queue, so the published numbers never
    silently mix evaluator versions.
    """
    out = cell.config.out_dir
    inference = _json(out / "inference_test.json")
    if inference is not None:
        scored = _json(out / "metrics_test.json")
        metric = scored.get("metric", {}) if scored is not None else {}
        same_ckpt = metric.get("checkpoint_sha256") == inference.get("checkpoint_sha256")
        if same_ckpt and metric.get("metrics_version") == METRICS_VERSION:
            return "finished"
        return "metric"
    try:
        resolve_checkpoint(cell)
    except FileNotFoundError:
        return "waiting"
    return "pending"


# --- test splits -----------------------------------------------------------------------


def render_runs(refs: Iterable, n: int) -> list[str]:
    """The `n` runs a cell renders: one per rock family, families and ids sorted, round-robin
    until `n` — so every model of a campaign renders the same runs and the GIFs compare."""
    by_family: dict[str, list[str]] = {}
    for ref in refs:
        by_family.setdefault(str(ref.geometry.get("family")), []).append(ref.run_id)
    queues = [sorted(ids) for _, ids in sorted(by_family.items())]
    picked: list[str] = []
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop(0))
    return picked


def render_gifs(out_dir: Path | str) -> list[Path]:
    """The GIFs a cell has rendered, sorted by run id (a failed render leaves no GIF)."""
    return sorted((Path(out_dir) / "render").glob("*/rollout_*.gif"))


def render_cell(cfg: Config, ckpt: Path | str | None, out_dir: Path | str, n: int, force: bool = False) -> list[Path]:
    """Render `n` runs of the cell's test split as truth | prediction GIFs, exactly training's
    periodic render (`evaluate.render_rollout`: a quarter into the run, `rollout.horizon` steps),
    into `<out_dir>/render/<run_id>/`. A run whose GIF exists is skipped unless `force`; a
    render that fails (the pyvista worker aborting, say) is reported with `!!` and skipped, so
    a cell's inference result never depends on the renderer. Returns the GIFs that exist."""
    from . import evaluate
    from .scoring import load_runs

    # The rollout scores every step in-process for the frame labels; the mesh/curvature family
    # (`eval_only`: 30-60 s per side at 256) would turn a 1-minute render into hours. The metric
    # stage scores those on the stored frames; the labels only need iou / saturation.
    if cfg.metrics:
        cfg = cfg.updated(metrics=[m for m in cfg.metrics if not m.eval_only])
    refs = load_runs(cfg)["test"]
    gifs: list[Path] = []
    for run_id in render_runs(refs, n):
        target = Path(out_dir) / "render" / run_id
        gif = target / f"rollout_{run_id}.gif"
        if gif.exists() and not force:
            print(f"  render {run_id}: {gif.name} exists, skipped")
        else:
            try:
                evaluate.render_rollout(cfg, ckpt=ckpt, run_id=run_id, out_dir=target, split="test")
                print(f"  render {run_id}: {gif}")
            except RuntimeError as e:  # the render worker died; the frames of the other runs are still worth having
                lines = [line for line in str(e).splitlines() if line.strip()] or ["?"]
                target.mkdir(parents=True, exist_ok=True)
                (target / "render_failed.txt").write_text(str(e))
                where = target / "render_failed.txt"
                print(f"  !! render {run_id} failed: {lines[0]} … {lines[-1][:160]} (full text in {where}); skipped")
                continue
        if gif.exists():
            gifs.append(gif)
    return gifs


def rock_max_axis(ref) -> int | None:
    """The size class of a run: the max axis of its *rock*, not of the solver's padded domain.

    `RunRef.shape` is the domain array, which carries inlet/outlet buffer and plate slabs
    along the flow axis (a 128^3 rock run's domain is e.g. (128, 128, 161)); crop the flow
    axis to `regions["rock"]` first. None when either is unrecorded.
    """
    shape, rock = getattr(ref, "shape", ()), getattr(ref, "regions", {}).get("rock")
    if not shape or not rock:
        return None
    return max(*shape[:-1], rock[1] - rock[0])


def pool_test_runs(refs: Iterable, size_class: int, families: Sequence[str] | None) -> list[str]:
    """Every finished run of the size class (and families, None = all), in run-id order.

    The studies take the whole pool — no draw, no quota — so the list is deterministic
    without a seed and a split file can be re-derived and checked at any time.
    """
    return sorted(
        r.run_id
        for r in refs
        if r.status == "finished" and rock_max_axis(r) == size_class and (families is None or r.geometry["family"] in families)
    )


def write_test_split(path: Path, test: list[str], header: str) -> None:
    body = yaml.safe_dump({"train": [], "val": [], "test": test}, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"# {line}\n" for line in header.strip().splitlines()) + body)


# --- report ----------------------------------------------------------------------------


def _fmt(v) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def _cells(campaign_dir: Path) -> dict[str, dict]:
    campaign_dir = Path(campaign_dir)
    return {
        p.name: json.loads((p / "metrics_test.json").read_text())
        for p in sorted(campaign_dir.iterdir())
        if (p / "metrics_test.json").is_file()
    }


def _scalar_metrics(payload: dict) -> list[str]:
    return [name for name, curve in payload["summary"].items() if curve and not isinstance(curve[0], list)]


def slug(metric: str) -> str:
    return metric.replace("/", "_")


def study_curves(campaign_dir: Path) -> list[Path]:
    """Per scalar metric: `curves/<slug>.svg` (mean vs step, one line per cell) and `curves/<slug>.csv` (the plotted numbers).

    CSV columns: `step`, then `<cell>_mean`, `<cell>_n_runs`, `<cell>_n_windows` per cell in name order.
    Every cell must carry the same `steps` (the protocol); a mismatch raises."""
    from . import viz  # optional dependency group

    cells = _cells(campaign_dir)
    if not cells:
        raise FileNotFoundError(f"no <cell>/metrics_test.json under {campaign_dir}")
    steps = next(iter(cells.values()))["steps"]
    for name, payload in cells.items():
        if payload["steps"] != steps:
            raise ValueError(f"{name}: steps {payload['steps']} differ from {steps}; the cells are not on one protocol")
    metrics = list(dict.fromkeys(m for p in cells.values() for m in _scalar_metrics(p)))
    out_dir = Path(campaign_dir) / "curves"
    out_dir.mkdir(exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        header = ["step"] + [f"{c}_{col}" for c in cells for col in ("mean", "n_runs", "n_windows")]
        rows = []
        for i, step in enumerate(steps):
            row = [step]
            for payload in cells.values():
                row += [
                    payload["summary"].get(metric, [None] * len(steps))[i],
                    payload["n_runs_per_step"].get(metric, [0] * len(steps))[i],
                    payload["n_windows_per_step"].get(metric, [0] * len(steps))[i],
                ]
            rows.append(row)
        csv_path = out_dir / f"{slug(metric)}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(
                [["" if v is None or (isinstance(v, float) and math.isnan(v)) else v for v in row] for row in rows]
            )
        svg = viz.plot_metric_curve(
            {c: (steps, p["summary"].get(metric, [])) for c, p in cells.items() if metric in p["summary"]},
            metric,
            out_dir / f"{slug(metric)}.svg",
        )
        written += [csv_path, svg]
    return written


def study_report(campaign_dir: Path) -> str:
    """`report.md`: per group (all runs, then gen / uCT when present) one table, rows metrics, columns cells,
    each entry `value at step 1 / value at the last step`; then counts and provenance per cell."""
    cells = _cells(campaign_dir)
    if not cells:
        raise FileNotFoundError(f"no <cell>/metrics_test.json under {campaign_dir}")
    first = next(iter(cells.values()))
    steps = first["steps"]
    metrics = list(dict.fromkeys(m for p in cells.values() for m in _scalar_metrics(p)))
    campaign_dir = Path(campaign_dir)
    lines = [
        f"# {campaign_dir.parent.name}: {campaign_dir.name}",
        "",
        f"Rollout windows of {steps[-1]} steps; entries are `step {steps[0]} / step {steps[-1]}`, "
        "mean over runs of per-run means.",
        "",
    ]

    def table(title: str, pick) -> None:
        lines.extend(["", f"## {title}", "", "| metric | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)])
        for m in metrics:
            row = []
            for payload in cells.values():
                curve = pick(payload).get("summary", {}).get(m)
                row.append(f"{_fmt(curve[0])} / {_fmt(curve[-1])}" if curve else "-")
            lines.append(f"| {m} | " + " | ".join(row) + " |")

    table("All runs", lambda p: p)
    groups = list(dict.fromkeys(g for p in cells.values() for g in p.get("groups", {})))
    for g in groups:
        n = len(first.get("groups", {}).get(g, {}).get("runs", []))
        table(f"{g} runs only ({n})", lambda p, g=g: p.get("groups", {}).get(g, {}))
    lines += ["", "| cell | runs | windows | checkpoint sha256 | commit |", "|---|---|---|---|---|"]
    for c, p in cells.items():
        meta = p.get("metric", {})
        sha = str(meta.get("checkpoint_sha256"))[:12]
        lines.append(f"| {c} | {p.get('n_runs', '-')} | {p.get('n_windows', '-')} | {sha} | {meta.get('poreml_commit')} |")
    text = "\n".join(lines) + "\n"
    (campaign_dir / "report.md").write_text(text)
    return text


# --- SLURM -----------------------------------------------------------------------------


def queued_configs(squeue_lines: Iterable[str], manifests: Path) -> dict[str, str]:
    """{cell config path: SLURM state} for every array task the queue still holds, resolved
    through the `job_<arrayjob>.tsv` symlink `submit` leaves beside each manifest."""
    found: dict[str, str] = {}
    for line in squeue_lines:
        parts = line.split()
        if len(parts) < 2 or "_" not in parts[0]:
            continue
        array_id, task_id = parts[0].split("_", 1)
        link = manifests / f"job_{array_id}.tsv"
        if not link.exists():
            print(f"  warning: queued {parts[0]} ({parts[1]}) has no manifest under {manifests}; ignoring it")
            continue
        for row in link.read_text().splitlines():
            idx, cfg = row.split("\t", 1)
            if idx == task_id:
                found[cfg] = parts[1]
    return found


def squeue_lines(job_name: str) -> list[str]:
    cmd = ["squeue", "-r", "-h", "-u", os.environ.get("USER", ""), "-n", job_name, "-o", "%i %T"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"squeue failed: {result.stderr.strip()} — not submitting blind")
    return result.stdout.splitlines()
