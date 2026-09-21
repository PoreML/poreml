"""Case reports: the tables and figures every case regenerates from its `output/` folder.

`output/<model>/results_test.json` and `rollout_test.json` are the inputs. A case script
calls `write_report` and `rollout_curves`; the logic lives here so every case shares one
implementation.
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def table(names, columns: dict[str, dict]) -> str:
    rows = ["| metric | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    rows += [f"| {n} | " + " | ".join(_fmt(values.get(n)) for values in columns.values()) + " |" for n in names]
    return "\n".join(rows)


def write_report(output_dir: Path, title: str = "Report") -> str:
    """Single-step and rollout-at-horizon tables for every `<output_dir>/<model>/`; writes `report.md`.

    `title` names the report — the transfer studies pass their study and campaign so a
    `report.md` says which cell it tabulates; the default keeps the training cases' output.
    """
    output_dir = Path(output_dir)
    models = sorted(p.name for p in output_dir.iterdir() if (p / "results_test.json").exists())
    if not models:
        raise FileNotFoundError(f"no <model>/results_test.json under {output_dir}")
    single = {m: json.loads((output_dir / m / "results_test.json").read_text()) for m in models}
    rolls = {m: json.loads(p.read_text()) for m in models if (p := output_dir / m / "rollout_test.json").exists()}
    first = next(iter(single.values()))
    names = [k for k, v in first["metrics"].items() if isinstance(v, float)]
    single_cols = {m: single[m]["metrics"] for m in models}
    lines = [f"# {title}", "", f"Test split: {first['n_runs']} runs, {first['n_samples']} single-step windows.", ""]
    lines += ["## Single-step (test, mean of per-run means)", "", table(names, single_cols)]
    if rolls:
        roll_first = next(iter(rolls.values()))
        roll_cols = {m: rolls[m]["at_horizon"] for m in rolls}
        h, frac = roll_first["horizon"], roll_first["start_fraction"]
        lines += ["", f"## Autonomous rollout at h = {h} frames from {frac:.0%} into each run (test, mean over runs)", ""]
        lines += [table(names, roll_cols)]
    prov = "; ".join(
        f"{m}: commit {single[m]['provenance']['poreml_commit']}, {single[m]['provenance']['device_name']}" for m in models
    )
    lines += ["", "Provenance: " + prov, ""]
    text = "\n".join(lines)
    (output_dir / "report.md").write_text(text)
    return text


def rollout_curves(output_dir: Path, ckpt: Path | None = None) -> list[Path]:
    """`rollout_curves.svg` over every model under `output_dir`, and `training_curves.svg`
    from the run's `metrics.csv` when `ckpt` points into a run."""
    from . import viz

    output_dir = Path(output_dir)
    payloads = {p.parent.name: json.loads(p.read_text()) for p in sorted(output_dir.glob("*/rollout_test.json"))}
    if not payloads:
        raise FileNotFoundError(f"no <model>/rollout_test.json under {output_dir}")
    sources: dict[str, dict] = dict(payloads)
    first = next(iter(payloads.values()))["summary"]
    metrics = [m for m, v in first.items() if v and not isinstance(v[0], list)]  # rollout curves of scalar metrics only
    written = [viz.plot_rollout_curves(sources, metrics, output_dir / "rollout_curves.svg")]
    if ckpt is not None and (Path(ckpt).parent.parent / "metrics.csv").exists():
        written.append(viz.plot_training_curves(Path(ckpt).parent.parent / "metrics.csv", output_dir / "training_curves.svg"))
    return written
