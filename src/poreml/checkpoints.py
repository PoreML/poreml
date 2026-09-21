"""Fetch the benchmark's trained checkpoints from the HuggingFace Hub into `case/`.

The Hub repository's root *is* `case/`: `<phase>/<campaign>/<model>_<kind>/ckpts/<run>/` holding
the run's `config.yaml`, `run_meta.json`, `metrics.csv`, `train_log.csv` and the weights
`ckpts/{best,last,best_rollout}.pt`. Everything that looks a finished run up — the push configs'
`train.init_from`, the studies, `util/inference`, the submitters — globs exactly that layout, so
downloading into `case/` leaves nothing to move and makes the cell read as finished.

A selection is any combination of phases (`train`, `train_push`), campaigns, models, split kinds
(`gen`, `all`) and weight files; a run's small text files always come with it, because a
checkpoint is scored with the config saved beside it. As with the dataset, the cost is shown
before anything is fetched, and a file already on disk with the Hub's byte size is skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .download import _free_bytes, _match, human, pending

REPO_ID = "PoreML/PoreML_checkpoint"
DEFAULT_ROOT = Path("case")
PHASES = ("train", "train_push")


@dataclass(frozen=True)
class RemoteFile:
    """One file of the Hub repository. Run files sit under `<phase>/<campaign>/<model>_<kind>/ckpts/<run>/`."""

    path: str
    size: int

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.path.split("/"))

    @property
    def is_run_file(self) -> bool:
        p = self.parts
        return len(p) in (6, 7) and p[0] in PHASES and p[3] == "ckpts" and "_" in p[2]

    @property
    def phase(self) -> str | None:
        return self.parts[0] if self.is_run_file else None

    @property
    def campaign(self) -> str | None:
        return self.parts[1] if self.is_run_file else None

    @property
    def model(self) -> str | None:
        return self.parts[2].rsplit("_", 1)[0] if self.is_run_file else None

    @property
    def kind(self) -> str | None:
        return self.parts[2].rsplit("_", 1)[1] if self.is_run_file else None

    @property
    def run_dir(self) -> str | None:
        return "/".join(self.parts[:5]) if self.is_run_file else None

    @property
    def weight(self) -> str | None:
        """`best`, `last` or `best_rollout` for a weight file; None for the run's text files."""
        p = self.parts
        is_weight = self.is_run_file and len(p) == 7 and p[5] == "ckpts" and p[6].endswith(".pt")
        return p[6].removesuffix(".pt") if is_weight else None


def list_files(repo_id: str = REPO_ID) -> list[RemoteFile]:
    """Every file of the repository with its size: one metadata request, nothing downloaded."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    return [RemoteFile(s.rfilename, int(s.size or 0)) for s in info.siblings]


def select(
    files: Iterable[RemoteFile],
    *,
    phases: Sequence[str] = (),
    campaigns: Sequence[str] = (),
    models: Sequence[str] = (),
    kinds: Sequence[str] = (),
    which: Sequence[str] = (),
) -> list[RemoteFile]:
    """The run files passing every given filter; an empty filter passes everything.

    `which` narrows the weight files only. A filter value that matches nothing is an error, not
    an empty download: a typo must not look like success.
    """
    run_files = [f for f in files if f.is_run_file]
    for label, wanted, attr in (
        ("phase", phases, "phase"),
        ("campaign", campaigns, "campaign"),
        ("model", models, "model"),
        ("kind", kinds, "kind"),
        ("which", which, "weight"),
    ):
        known = {getattr(f, attr) for f in run_files} - {None}
        unknown = [w for w in wanted if w.lower() not in {k.lower() for k in known}]
        if unknown:
            raise ValueError(f"unknown {label}: {', '.join(unknown)} (available: {', '.join(sorted(known))})")
    runs = [
        f
        for f in run_files
        if _match(f.phase, phases) and _match(f.campaign, campaigns) and _match(f.model, models) and _match(f.kind, kinds)
    ]
    chosen = [f for f in runs if f.weight is None or _match(f.weight, which)]
    if not any(f.weight for f in chosen):
        raise ValueError("the filters select no checkpoint: each one matches something, but no weight file passes all of them")
    # A run none of whose weights were asked for (a base run under `--which best_rollout`) is not selected at all.
    kept = {f.run_dir for f in chosen if f.weight}
    return [f for f in chosen if f.run_dir in kept]


def summarise(files: Iterable[RemoteFile]) -> list[dict]:
    """One row per (phase, campaign): runs, weight files, bytes — and a `total` row last."""
    rows: dict[tuple[str, str], dict] = {}
    for f in files:
        if not f.is_run_file:
            continue
        row = rows.setdefault(
            (f.phase, f.campaign), {"phase": f.phase, "campaign": f.campaign, "runs": set(), "weights": 0, "bytes": 0}
        )
        row["runs"].add(f.run_dir)
        row["weights"] += f.weight is not None
        row["bytes"] += f.size
    table = [{**r, "runs": len(r["runs"])} for _, r in sorted(rows.items())]
    table.append({"phase": "total", "campaign": "", **{k: sum(r[k] for r in table) for k in ("runs", "weights", "bytes")}})
    return table


def format_plan(
    repo_id: str, root: Path, everything: list[RemoteFile], chosen: list[RemoteFile], todo: list[RemoteFile]
) -> str:
    """What the user reads before agreeing: the repository's size, the selection's, what is left to fetch."""
    whole = summarise(everything)[-1]
    need, free = sum(f.size for f in todo), _free_bytes(root)
    lines = [
        f"checkpoints  https://huggingface.co/{repo_id}",
        f"             {whole['runs']} runs, {whole['weights']} weight files, {human(whole['bytes'])} in full",
        f"into         {root}",
        "",
        f"{'phase':<12}{'campaign':<12}{'runs':>6}{'weights':>9}{'size':>12}",
    ]
    lines += [
        f"{r['phase']:<12}{r['campaign']:<12}{r['runs']:>6}{r['weights']:>9}{human(r['bytes']):>12}" for r in summarise(chosen)
    ]
    have = sum(f.size for f in chosen) - need
    lines += ["", f"to fetch     {len(todo)} file(s), {human(need)} ({human(have)} already on disk)"]
    lines.append(f"free         {human(free)} on the target filesystem" + ("  <-- NOT ENOUGH" if need > free else ""))
    return "\n".join(lines)


def _fetch(repo_id: str, root: Path, todo: list[RemoteFile], workers: int) -> None:
    from huggingface_hub import snapshot_download

    # Exact paths: a few hundred small patterns, and a glob would re-verify the weights already there.
    allow = [f.path for f in todo]
    snapshot_download(repo_id, repo_type="model", local_dir=str(root), allow_patterns=allow, max_workers=workers)


def download(
    root: Path = DEFAULT_ROOT,
    *,
    repo_id: str = REPO_ID,
    phases: Sequence[str] = (),
    campaigns: Sequence[str] = (),
    models: Sequence[str] = (),
    kinds: Sequence[str] = (),
    which: Sequence[str] = (),
    workers: int = 8,
    dry_run: bool = False,
    confirm: Callable[[str], bool] | None = None,
    echo: Callable[[str], None] = print,
    files: list[RemoteFile] | None = None,
    fetch: Callable[[str, Path, list[RemoteFile], int], None] | None = None,
) -> dict:
    """Show what the selection costs, ask, then fetch what is missing under `root`.

    `confirm(prompt) -> bool` decides; `None` means the caller has already agreed. No filter
    selects every checkpoint. Returns `{state, n_files, bytes, root}` with `state` one of
    `present` (nothing to fetch), `dry-run`, `declined`, `downloaded`.
    """
    root = Path(root)
    everything = files if files is not None else list_files(repo_id)
    chosen = select(everything, phases=phases, campaigns=campaigns, models=models, kinds=kinds, which=which)
    todo = pending(chosen, root)
    result = {"n_files": len(todo), "bytes": sum(f.size for f in todo), "root": str(root)}
    echo(format_plan(repo_id, root, everything, chosen, todo))
    if not todo:
        return {**result, "state": "present"}
    if dry_run:
        return {**result, "state": "dry-run"}
    if confirm is not None and not confirm(f"Download {human(result['bytes'])} into {root}?"):
        return {**result, "state": "declined"}
    root.mkdir(parents=True, exist_ok=True)
    (fetch or _fetch)(repo_id, root, todo, workers)
    left = pending(chosen, root)
    if left:
        raise RuntimeError(f"{len(left)} file(s) still missing after the download, e.g. {left[0].path}; rerun to resume")
    return {**result, "state": "downloaded"}
