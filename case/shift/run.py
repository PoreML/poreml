"""The Geo-Shift study: the `<model>_gen` checkpoints (trained on generated rocks only at the
128 class), without retraining, on every uCT rock of the same campaign at the same class.
The geometry source is the only thing that changes. See README.md.

    uv run python case/shift/run.py split --check                  # verify splits/shift/<campaign>.yaml
    uv run python case/shift/run.py inference --campaign gdl --model abupt   # GPU stage, one cell
    uv run python case/shift/run.py metric --cell configs/shift/gdl/abupt_gen.yaml --workers 16   # CPU stage
    uv run python case/shift/run.py report                         # curves + report.md per campaign
    uv run python case/shift/run.py all                            # inference, metric, report over every cell

Every cell is a config under configs/shift/<campaign>/<model>_gen.yaml (poreml.study): the
training case whose newest finished checkpoint is scored, the campaign's shared data.yaml and
the study's shared scheme.yaml (horizon, stride, keyframes, metric list). The scored config is
the checkpoint's own saved config.yaml with those swapped in — never the current
configs/<campaign>/ files. Outputs: case/shift/<campaign>/<model>_gen/ and, per campaign,
case/shift/<campaign>/report.md + curves/.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = "shift"
CAMPAIGNS = ("drainage", "gdl", "trapping")
MODELS = ("unet", "fno", "p3d", "transolver", "abupt")
STAGES = ("split", "inference", "metric", "report")

# Split rule: every finished 128-class uCT run of the campaign — the whole uCT counterpart,
# unseen by any `gen` checkpoint. That is train + val of splits/<campaign>/uCT.yaml, which
# `split --check` verifies so the two frozen files can never drift apart.
SIZE_CLASS = 128
UCT_FAMILIES = ("bentheimer", "buffberea", "castlegate", "gdl_ct", "gdl_ct_20", "gdl_ct_40")  # splits/draw.py: SOURCES["uCT"]
POOL_FAMILIES: dict[str, tuple[str, ...] | None] = {"drainage": UCT_FAMILIES, "gdl": UCT_FAMILIES, "trapping": UCT_FAMILIES}


def cells(args):
    from poreml.study import Cell, cell_paths

    paths = [Path(p) for p in args.cells] if args.cells else cell_paths(STUDY, args.campaigns, repo=REPO)
    for path in paths:
        if args.models and path.stem.split("_")[0] not in args.models:
            continue
        yield Cell.load(path)


def scored(args):
    """(cell, cfg, ckpt) for every selected cell that has a finished checkpoint; the rest are reported."""
    from poreml.study import derive, resolve_checkpoint

    for cell in cells(args):
        try:
            ckpt = resolve_checkpoint(cell)
        except FileNotFoundError as e:
            print(f"!! waiting  {cell.path}: {e}")
            continue
        yield cell, derive(cell, ckpt), ckpt


def stage_split(args) -> None:
    """Draw (or --check) the frozen eval-only test split of each campaign from the pool rule."""
    import yaml

    from poreml.config import DataConfig
    from poreml.data import discover
    from poreml.study import pool_test_runs, write_test_split

    for campaign in args.campaigns:
        data = DataConfig.model_validate(yaml.safe_load((REPO / "configs" / STUDY / campaign / "data.yaml").read_text()))
        root = Path(args.root) if args.root else data.root
        families = POOL_FAMILIES[campaign]
        refs = list(discover(root, data.campaign).values())
        test = pool_test_runs(refs, SIZE_CLASS, families)
        path = REPO / data.split
        uct = yaml.safe_load((REPO / "splits" / campaign / "uCT.yaml").read_text())
        uct_runs = sorted(uct["train"] + uct["val"])
        if test != uct_runs:
            print(f"!! {campaign}: pool ({len(test)}) != train + val of splits/{campaign}/uCT.yaml ({len(uct_runs)}); skipped")
            continue
        if args.check:
            current = yaml.safe_load(path.read_text())["test"] if path.exists() else None
            verdict = "matches" if current == test else "DIFFERS FROM"
            print(f"{path.relative_to(REPO)}: {verdict} train + val of the uCT split ({len(test)} runs)")
            continue
        if path.exists() and not args.force:
            print(f"{path.relative_to(REPO)} exists; splits are frozen — --force redraws (invalidating every number)")
            continue
        if not test:
            print(f"{campaign}: no finished {SIZE_CLASS}-class runs under {root} — skipped")
            continue
        header = (
            f"{STUDY} test split, campaign {data.campaign} — drawn by case/{STUDY}/run.py split on {date.today()}\n"
            f"from {root}.\n"
            f"Every finished run whose rock max axis is {SIZE_CLASS}, families {', '.join(families)}: "
            f"{len(test)} of {len(refs)} discovered\n= train + val of splits/{campaign}/uCT.yaml (checked). "
            f"No draw, no quota. Eval-only: train and val are empty."
        )
        write_test_split(path, test, header)
        print(f"wrote {path.relative_to(REPO)}: test {len(test)}")


def stage_inference(args) -> None:
    """GPU stage: store the keyframes of every 64-step window of every run; nothing scored."""
    from poreml.evaluate import inference_split
    from poreml.rollout import Interrupted

    for cell, cfg, ckpt in scored(args):
        state = _state(cell)
        if state in ("finished", "metric") and not args.force:
            print(f"-- inference {cell.config.name}: inference_test.json exists, skipped (--force redoes it)")
            continue
        stride = cell.scheme.window_stride
        print(
            f"-- inference {cell.config.name} ({ckpt}) horizon {cfg.rollout.horizon}, "
            f"stride {stride}, keyframes {cfg.rollout.keyframes}"
        )
        try:
            payload = inference_split(cfg, ckpt=ckpt, split="test", out_dir=cell.config.out_dir, stride=stride)
        except Interrupted as e:
            print(f"!! interrupted {cell.config.name}: {e}")
            raise SystemExit(75) from e
        print(
            f"  {payload['n_windows']} windows, {payload['n_frames']} frames stored, "
            f"{len(payload['runs_without_windows'])} run(s) too short"
        )


def stage_metric(args) -> None:
    """CPU stage: score every metric on the stored frames and write metrics_test.csv / .json."""
    from poreml.evaluate import metric_split
    from poreml.metric import default_workers

    workers = args.workers or default_workers()
    for cell, cfg, ckpt in scored(args):
        state = _state(cell)
        if state == "pending":
            print(f"!! no frames  {cell.path}: run the inference stage first")
            continue
        if state == "finished" and not args.force:
            print(f"-- metric {cell.config.name}: metrics_test.json exists, skipped (--force redoes it)")
            continue
        print(f"-- metric {cell.config.name} on {workers} workers")
        payload = metric_split(cfg, ckpt=ckpt, split="test", out_dir=cell.config.out_dir, workers=workers)
        print(f"  {payload['n_frames']} frames in {payload['metric']['seconds']} s; at h={payload['steps'][-1]}:")
        _print_metrics(cfg, payload["at_horizon"])


def stage_report(args) -> None:
    from poreml.study import study_curves, study_report

    root = (Path(args.root) if args.root else REPO / "case" / STUDY).resolve()
    for campaign in args.campaigns:
        campaign_dir = root / campaign
        if not any(campaign_dir.glob("*/metrics_test.json")):
            print(f"!! {campaign}: no metrics under {campaign_dir} — skipped")
            continue
        print(study_report(campaign_dir))
        for p in study_curves(campaign_dir):
            print(f"wrote {p.relative_to(REPO)}")


def _state(cell) -> str:
    from poreml.study import classify

    return classify(cell)


def _print_metrics(cfg, values: dict) -> None:
    print(f"  {'':<40} {cfg.name:>28}")
    for name, value in values.items():
        if isinstance(value, float):
            print(f"  {name:<40} {value:>28.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=STAGES + ("all",))
    ap.add_argument("--cell", dest="cells", action="append", help=f"repeatable cell config; default: configs/{STUDY}/*/*.yaml")
    ap.add_argument("--campaign", dest="campaigns", action="append", choices=CAMPAIGNS, help="repeatable filter; default: all")
    ap.add_argument("--model", dest="models", action="append", choices=MODELS, help="repeatable filter; default: all")
    ap.add_argument("--workers", type=int, default=None, help="metric: processes (default $SLURM_CPUS_PER_TASK)")
    ap.add_argument(
        "--root",
        default=None,
        help=f"split: discover runs under another data root (e.g. the solver's raw output); "
        f"report: read cells under another results root (e.g. case_dummy/{STUDY}) instead of case/{STUDY}",
    )
    ap.add_argument("--check", action="store_true", help="split: verify the existing files against the pool rule")
    ap.add_argument(
        "--force", action="store_true", help="split: overwrite an existing file; inference/metric: redo an existing artifact"
    )
    args = ap.parse_args()
    args.campaigns = tuple(args.campaigns or CAMPAIGNS)
    args.models = tuple(args.models or ())
    os.chdir(REPO)  # configs use repo-relative paths
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    stages = list(STAGES[1:]) if "all" in args.stages else [s for s in STAGES if s in args.stages]
    for stage in stages:
        print(f"== {stage}")
        globals()[f"stage_{stage}"](args)


if __name__ == "__main__":
    main()
