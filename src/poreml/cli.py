"""Command-line entry point."""

import json
from pathlib import Path
from typing import Annotated

import typer

from .config import Config
from .data import discover, load_split
from .evaluate import evaluate as run_evaluate
from .evaluate import render_rollout, rollout_split
from .metrics import json_safe
from .registry import DESCRIPTORS, ERRORS, MODELS, TASKS
from .rollout import Interrupted
from .train import Preempted, find_resumable
from .train import train as run_train

app = typer.Typer(add_completion=False, help="Benchmark for pore-scale multiphase flow.")
convert_app = typer.Typer(help="Re-encode the solver's gzip runs to zstd under poreml/data (see poreml.convert).")
app.add_typer(convert_app, name="convert")

REGISTRIES = {"tasks": TASKS, "models": MODELS}


@app.command()
def download(
    root: Annotated[Path, typer.Option("--root", help="Where the dataset goes: the configs' data.root")] = Path("data/case"),
    campaign: Annotated[
        list[str] | None, typer.Option("--campaign", help="drainage | trapping | GDL | underfill; repeatable")
    ] = None,
    family: Annotated[
        list[str] | None, typer.Option("--family", help="Geometry family, e.g. bentheimer, sphere, fiber; repeatable")
    ] = None,
    size: Annotated[list[str] | None, typer.Option("--size", help="Domain size, e.g. 128, 256, 128x128x64; repeatable")] = None,
    run: Annotated[list[str] | None, typer.Option("--run", help="One run ID; repeatable")] = None,
    split: Annotated[
        list[Path] | None, typer.Option("--split", exists=True, help="Every run a split file names; repeatable")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before downloading")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be fetched and its size; fetch nothing")] = False,
    workers: Annotated[int, typer.Option("--workers", min=1, help="Parallel downloads")] = 8,
    repo: Annotated[str | None, typer.Option("--repo", help="HuggingFace dataset repository")] = None,
) -> None:
    """Download the dataset from HuggingFace, whole or in part, into the layout the configs expect.

    Without a filter the whole dataset is selected (3.3 TB). The filters intersect; `--split`
    adds the runs of a split file to `--run`. The size of the selection is shown first and
    nothing is fetched until you agree; files already on disk are skipped, so rerunning resumes.
    """
    from . import download as dl

    run_ids = [*(run or []), *(dl.split_runs(split) if split else [])]
    try:
        result = dl.download(
            root,
            repo_id=repo or dl.REPO_ID,
            campaigns=campaign or (),
            families=family or (),
            sizes=size or (),
            runs=run_ids,
            workers=workers,
            dry_run=dry_run,
            confirm=None if yes else lambda prompt: typer.confirm(prompt, default=False),
            echo=typer.echo,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    if result["state"] == "declined":
        typer.echo("nothing downloaded")
        raise typer.Exit(code=1)
    if result["state"] == "present":
        typer.echo("everything selected is already on disk")
    elif result["state"] == "downloaded":
        typer.echo(f"downloaded {result['n_files']} file(s) into {result['root']}")


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Path to a run config YAML")],
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help="Continue a run that stopped short: a run directory, or 'auto' for the newest unfinished run of "
            "this exact config under out_dir (a fresh run starts when there is none)",
        ),
    ] = None,
) -> None:
    """Train a model and write artifacts to a fresh run directory (or continue one with --resume).

    Exit code 75 (EX_TEMPFAIL) means SIGTERM stopped the run with its resume state saved —
    the SLURM worker requeues on it.
    """
    cfg = Config.from_yaml(config)
    run_dir = find_resumable(cfg) if resume == "auto" else (Path(resume) if resume is not None else None)
    try:
        typer.echo(str(run_train(cfg, resume=run_dir)))
    except Preempted as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=75) from e


@app.command("eval")
def eval_(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Path to a run config YAML")],
    ckpt: Annotated[Path | None, typer.Option("--ckpt", help="Checkpoint to score; omit to score an untrained model")] = None,
    split: Annotated[str, typer.Option("--split", help="Split section to score")] = "test",
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", help="Where to write results_<split>.json; defaults beside the checkpoint")
    ] = None,
) -> None:
    """Score a checkpoint on a split and write results_<split>.json."""
    results = run_evaluate(Config.from_yaml(config), ckpt=ckpt, split=split, out_dir=out_dir)
    for name, value in results.summary.items():
        shown = f"{value:.6f}" if isinstance(value, float) else json.dumps(json_safe(value))
        typer.echo(f"{name:<24} {shown}")


@app.command()
def rollout(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Path to a run config YAML")],
    ckpt: Annotated[Path | None, typer.Option("--ckpt", help="Checkpoint to roll out; omit for an untrained model")] = None,
    split: Annotated[str, typer.Option("--split", help="Split section to roll out on")] = "test",
    horizon: Annotated[int | None, typer.Option("--horizon", min=1, help="Override rollout.horizon")] = None,
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", help="Where rollout_<split>.json goes; defaults beside the checkpoint")
    ] = None,
) -> None:
    """Roll a checkpoint out autonomously on every run of a split, every step scored: rollout_<split>.json.

    One rollout per run from `rollout.start_fraction`. The windowed protocol of the paper's
    tables is `poreml inference` (GPU) followed by `poreml metric` (CPU).
    """
    cfg = Config.from_yaml(config)
    if horizon is not None:
        cfg = cfg.updated(rollout={"horizon": horizon})
    summary = rollout_split(cfg, ckpt=ckpt, split=split, out_dir=out_dir)
    typer.echo(f"horizon {summary.horizon} over {len(summary.runs)} run(s); mean over runs at h=1, h=mid, h={summary.horizon}:")
    mid = (summary.horizon - 1) // 2
    for name, curve in summary.summary.items():
        picks = [curve[0], curve[mid], curve[-1]]
        shown = "  ".join(f"{v:.6f}" if isinstance(v, float) else json.dumps(json_safe(v)) for v in picks)
        typer.echo(f"{name:<24} {shown}")


@app.command()
def render(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Path to a run config YAML")],
    run: Annotated[str, typer.Option("--run", help="Run ID (must be in the split) to roll out and render")],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Where frames/, the GIF and the JSON go")],
    ckpt: Annotated[Path | None, typer.Option("--ckpt", help="Checkpoint to roll out; omit for an untrained model")] = None,
    split: Annotated[str, typer.Option("--split", help="Split section holding the run")] = "test",
    horizon: Annotated[int | None, typer.Option("--horizon", min=1, help="Override rollout.horizon")] = None,
    fps: Annotated[int, typer.Option("--fps", min=1, help="GIF frame rate")] = 6,
) -> None:
    """Roll one run out and render truth | prediction frames and a GIF in the solver's style."""
    out = render_rollout(
        Config.from_yaml(config), ckpt=ckpt, run_id=run, out_dir=out_dir, split=split, horizon=horizon, fps=fps
    )
    typer.echo(str(out))


@app.command()
def inference(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="The run's own config.yaml")],
    ckpt: Annotated[Path | None, typer.Option("--ckpt", help="Checkpoint to roll out; omit for an untrained model")] = None,
    split: Annotated[str, typer.Option("--split", help="Split section to roll out on")] = "val",
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", help="Where inference.csv and inference/ go; default: the checkpoint's run dir")
    ] = None,
    horizon: Annotated[int | None, typer.Option("--horizon", min=1, help="Override rollout.horizon")] = None,
    stride: Annotated[
        int | None, typer.Option("--stride", min=1, help="Frames between window starts (default: the horizon)")
    ] = None,
    keyframes: Annotated[int | None, typer.Option("--keyframes", min=1, help="Override rollout.keyframes")] = None,
    fields: Annotated[str, typer.Option("--fields", help="Fields scored with mae / rel_mae, comma-separated")] = "phi,p",
    force: Annotated[bool, typer.Option("--force", help="Redo a finished stage for the same checkpoint")] = False,
) -> None:
    """Timed rollouts of every window of a split: mae / rel_mae per field at every step, forward seconds, keyframes.

    Windows start at frame 0 and every `--stride` frames (the studies' protocol on the training's
    own split). Writes `<out_dir>/inference.csv` and `<out_dir>/inference/` (frames,
    inference.json). SIGTERM stops after the window in hand (exit 75); rerunning resumes.
    `poreml metric` then scores the stored keyframes with the config's whole metric list;
    `util/inference/inference.py` drives both over every finished training.
    """
    from .inference import rollout_inference

    cfg = Config.from_yaml(config)
    try:
        payload = rollout_inference(
            cfg,
            ckpt=ckpt,
            split=split,
            out_dir=out_dir,
            horizon=horizon,
            stride=stride,
            keyframes=keyframes,
            fields=tuple(f.strip() for f in fields.split(",") if f.strip()),
            force=force,
        )
    except Interrupted as e:
        typer.echo(f"interrupted: {e}", err=True)
        raise typer.Exit(code=75) from e
    fwd = payload["timing"]["forward_s"]
    first = payload["metrics_spec"][0]["name"]
    typer.echo(
        f"{payload['n_runs']} runs, {payload['n_windows']} windows, {fwd.get('n', 0)} steps, "
        f"forward {fwd.get('mean', float('nan')):.3f} s/step; {first} at horizon {payload['at_horizon'][first]}"
    )


@app.command()
def metric(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="The run's own config.yaml")],
    ckpt: Annotated[Path | None, typer.Option("--ckpt", help="The checkpoint `poreml inference` rolled out")] = None,
    split: Annotated[str, typer.Option("--split", help="Split section the frames were stored for")] = "val",
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", help="The --out-dir given to `poreml inference`; default: the run directory")
    ] = None,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, help="Processes; default $SLURM_CPUS_PER_TASK or the CPU count")
    ] = None,
) -> None:
    """Score the keyframes `poreml inference` stored with the config's whole metric list, on CPU.

    Writes `<out_dir>/inference/metrics_<split>.csv` (one row per run, window and step) and
    `.json` (per step: mean per run first, then over runs). No GPU; the mesh descriptors
    (curvature, Minkowski functionals) are the reason this is its own stage.
    """
    from .inference import metric_inference
    from .metric import default_workers

    payload = metric_inference(
        Config.from_yaml(config), ckpt=ckpt, split=split, out_dir=out_dir, workers=workers or default_workers()
    )
    typer.echo(f"metric: {payload['n_frames']} frames on {payload['metric']['workers']} workers; at h={payload['steps'][-1]}:")
    for name, value in payload["at_horizon"].items():
        if isinstance(value, float):
            typer.echo(f"{name:<40} {value:.6f}")


@app.command()
def finalise(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="A training run directory")],
    render: Annotated[
        bool | None, typer.Option("--render/--no-render", help="Override the config's eval.render for the final stage")
    ] = None,
) -> None:
    """Mark the rollout-best epoch of a finished run as ckpts/best_rollout.pt and evaluate it fully under final/.

    `train` does this when it ends; run it by hand for a run that finished before 2026-09-10 (it
    chooses between best.pt and last.pt) or whose final stage failed. Reruns replace final/.
    """
    from .finalise import finalise as run_finalise

    record = run_finalise(run_dir, render=render)
    if record is None:
        typer.echo("no periodic rollout rows to select on; nothing marked")
        raise typer.Exit(code=1)
    meta = json.loads((Path(run_dir) / "run_meta.json").read_text())
    chosen = meta["progress"]["best_rollout"]
    criterion = next(k for k in chosen if k.startswith("rollout/"))
    typer.echo(
        f"epoch {chosen['epoch']} marked as {record['checkpoint']}: "
        f"{criterion} {chosen[criterion]:.6g} of {chosen['candidates']} candidates"
    )

    if record.get("error"):
        typer.echo(f"final stage failed: {record['error']}", err=True)
        raise typer.Exit(code=1)
    typer.echo(record["dir"])


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "nan" if value != value else f"{value:g}"
    return str(value)


@app.command()
def runs(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, help="Path to a run config YAML")],
    split: Annotated[str | None, typer.Option("--split", help="Only runs in this split section (train, val, test)")] = None,
) -> None:
    """List the runs the solver has produced for the config's campaign, with status and conditions.

    Every discovered run is shown, whether or not the split names it, so a split can be
    drawn from what is actually there — and so a run still `running` is visible before it
    ends up in one.
    """
    cfg = Config.from_yaml(config)
    available = discover(cfg.data.root, cfg.data.campaign)
    section_of = {run_id: section for section, ids in load_split(cfg.data.split).items() for run_id in ids}
    header = ["run_id", "split", "status", "finish", "family", "geometry", "M", "theta", "ca"]
    rows = []
    for run_id, ref in sorted(available.items()):
        section = section_of.get(run_id)
        if split is not None and section != split:
            continue
        geo, params = ref.geometry, ref.params
        rows.append(
            [run_id, section, ref.status, ref.finish_type, geo["family"], geo["id"], params["M"], params["theta"], params["ca"]]
        )
    table = [header, *([_fmt(v) for v in row] for row in rows)]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in table:
        typer.echo("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())


@app.command()
def ls(kind: Annotated[str, typer.Argument(help="One of: tasks, models, metrics")]) -> None:
    """List the registered extension points."""
    if kind == "metrics":
        typer.echo("descriptors (a metric is {descriptor: ..., error: ...}; descriptor defaults to voxels):")
        for name in DESCRIPTORS.names():
            typer.echo(f"  {name}")
        typer.echo("errors:")
        for name in ERRORS.names():
            typer.echo(f"  {name}")
        return
    if kind not in REGISTRIES:
        typer.echo(f"unknown kind {kind!r}; expected one of: tasks, models, metrics", err=True)
        raise typer.Exit(code=1)
    for name in REGISTRIES[kind].names():
        typer.echo(name)


# --- convert ----------------------------------------------------------------------------

SRC_DEFAULT = Path("../solver/case")
DST_DEFAULT = Path("data/case")


@convert_app.command("scan")
def convert_scan(
    src: Annotated[Path, typer.Option("--src", help="the solver's case root (<campaign>/runs/...)")] = SRC_DEFAULT,
    dst: Annotated[Path, typer.Option("--dst", help="poreml's mirrored data root")] = DST_DEFAULT,
) -> None:
    """Classify every source run: converted, pending, not finished, out of scope."""
    from .convert import scan

    entries = scan(src, dst)
    for e in entries:
        typer.echo(
            f"{e['state']:16s} {e.get('campaign', '?'):10s} {e.get('family', '?')}/{e.get('size', '?'):12s} "
            f"{e['run_id']}  {e['reason']}"
        )
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["state"]] = counts.get(e["state"], 0) + 1
    typer.echo(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


@convert_app.command("run")
def convert_run_(
    src_run: Annotated[Path, typer.Argument(exists=True, help="one source run directory")],
    dst_run: Annotated[Path, typer.Argument(help="its mirrored destination directory")],
) -> None:
    """Convert one run (what a SLURM array task does); prints the verification record."""
    from .convert import convert_run

    record = convert_run(src_run, dst_run)
    typer.echo(json.dumps(record, indent=2))


@convert_app.command("ledger")
def convert_ledger(
    src: Annotated[Path, typer.Option("--src")] = SRC_DEFAULT,
    dst: Annotated[Path, typer.Option("--dst")] = DST_DEFAULT,
    out: Annotated[Path | None, typer.Option("--out", help="defaults to <src>/h5_todo.md")] = None,
) -> None:
    """Write the h5_todo.md ledger: what is not converted and why."""
    from .convert import scan, write_ledger

    path = out or src / "h5_todo.md"
    write_ledger(scan(src, dst), path, dst.resolve())
    typer.echo(str(path))


@convert_app.command("anonymise")
def convert_anonymise(
    root: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="the mirrored data root to edit in place")],
    replace: Annotated[
        list[str] | None,
        typer.Option("--replace", help="OLD=NEW, repeatable; default <your login name>=poreml_author"),
    ] = None,
    json_files: Annotated[bool, typer.Option("--json/--no-json", help="rewrite the JSON sidecars")] = True,
    h5: Annotated[bool, typer.Option("--h5/--no-h5", help="rewrite the HDF5 root attributes (no reader may be open)")] = True,
    rehash: Annotated[bool, typer.Option("--rehash/--no-rehash", help="recompute the conversion record's sha256")] = True,
    deep: Annotated[bool, typer.Option("--deep", help="also walk every group and dataset's attributes (slow on NFS)")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="count what would change; write nothing")] = False,
    workers: Annotated[int, typer.Option("--workers", help="parallel run directories")] = 1,
    ledger: Annotated[Path | None, typer.Option("--ledger", help="write the per-run records here as JSON")] = None,
) -> None:
    """Anonymise the mirror in place: usernames in every path, the SLURM job id set to null.

    The JSON pass is safe beside readers (atomic replace); the HDF5 pass is not — run it when
    nothing reads the tree. Idempotent: a second pass reports zero changes.
    """
    from .convert import DEFAULT_REPLACEMENTS, anonymise_tree

    replacements = dict(DEFAULT_REPLACEMENTS)
    if replace:
        replacements = {}
        for item in replace:
            old, sep, new = item.partition("=")
            if not sep or not old:
                raise typer.BadParameter(f"--replace wants OLD=NEW, got {item!r}")
            replacements[old] = new
    summary = anonymise_tree(
        root, replacements, json_files=json_files, h5=h5, rehash=rehash, dry_run=dry_run, deep=deep, workers=workers
    )
    if ledger is not None:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps(summary, indent=2))
    changed = [r for r in summary["per_run"] if r["json_replacements"] or r["h5_attrs_changed"]]
    typer.echo(
        f"{'dry run: ' if dry_run else ''}{summary['runs']} run{'s' if summary['runs'] != 1 else ''}, "
        f"{len(changed)} with changes: {summary['json_replacements']} JSON replacements, "
        f"{summary['h5_attrs_changed']} HDF5 attributes, {summary['rehashed']} files rehashed"
    )


@convert_app.command("repack")
def convert_repack(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="one mirrored run directory")],
    drop: Annotated[list[str] | None, typer.Option("--drop", help="dataset names to leave out; default rho, umag")] = None,
    replace: Annotated[
        list[str] | None, typer.Option("--replace", help="OLD=NEW for the root attributes; default <login name>=poreml_author")
    ] = None,
    drop_npy: Annotated[bool, typer.Option("--drop-npy", help="also delete a .npy sidecar identical to /rock")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="check derivability and count; write nothing")] = False,
) -> None:
    """Repack one run's HDF5 without its derivable fields (what a SLURM array task does); prints the record.

    Raw chunks are copied as they are, so no decode or re-encode happens; the root attributes are
    anonymised on the way and the conversion record gets the new size and hash. No reader may have
    the file open. Idempotent.
    """
    from .convert import DEFAULT_REPLACEMENTS, DROPPED_FIELDS, repack_run

    replacements = dict(DEFAULT_REPLACEMENTS)
    if replace:
        replacements = {}
        for item in replace:
            old, sep, new = item.partition("=")
            if not sep or not old:
                raise typer.BadParameter(f"--replace wants OLD=NEW, got {item!r}")
            replacements[old] = new
    record = repack_run(
        run_dir, drop=tuple(drop) if drop else DROPPED_FIELDS, replacements=replacements, drop_npy=drop_npy, dry_run=dry_run
    )
    typer.echo(json.dumps(record, indent=2))


if __name__ == "__main__":
    app()
