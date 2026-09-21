"""Fetch the PoreML dataset from the HuggingFace Hub into the layout the configs expect.

The Hub repository's root *is* `data/case`: `<campaign>/runs/<family>/<size>/<run_id>/<files>`,
plus the dataset card (`README.md`, `assets/`). Downloading into `data.root` therefore leaves
nothing to move. The whole dataset is 3.3 TB, so nothing is fetched before the caller has seen
what the selection costs and agreed to it; a selection is any combination of campaigns, geometry
families, domain sizes and run IDs (a split file is just a list of run IDs). A file already on
disk with the Hub's byte size is never fetched again, which makes a rerun the resume path.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "PoreML/PoreML_data"
DEFAULT_ROOT = Path("data/case")
CARD_FILES = ("README.md", "assets")  # the dataset card: a few MB, fetched with every selection


@dataclass(frozen=True)
class RemoteFile:
    """One file of the Hub repository. Run files sit at `<campaign>/runs/<family>/<size>/<run_id>/<name>`."""

    path: str
    size: int

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.path.split("/"))

    @property
    def is_run_file(self) -> bool:
        return len(self.parts) == 6 and self.parts[1] == "runs"

    @property
    def is_card(self) -> bool:
        return self.parts[0] in CARD_FILES

    @property
    def campaign(self) -> str | None:
        return self.parts[0] if self.is_run_file else None

    @property
    def family(self) -> str | None:
        return self.parts[2] if self.is_run_file else None

    @property
    def domain(self) -> str | None:
        return self.parts[3] if self.is_run_file else None

    @property
    def run_id(self) -> str | None:
        return self.parts[4] if self.is_run_file else None

    @property
    def run_dir(self) -> str | None:
        return "/".join(self.parts[:5]) if self.is_run_file else None


def list_files(repo_id: str = REPO_ID) -> list[RemoteFile]:
    """Every file of the dataset with its size: one metadata request, nothing downloaded."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, files_metadata=True)
    return [RemoteFile(s.rfilename, int(s.size or 0)) for s in info.siblings]


def _match(value: str | None, wanted: Sequence[str]) -> bool:
    return not wanted or (value is not None and value.lower() in {w.lower() for w in wanted})


def select(
    files: Iterable[RemoteFile],
    *,
    campaigns: Sequence[str] = (),
    families: Sequence[str] = (),
    sizes: Sequence[str] = (),
    runs: Sequence[str] = (),
) -> list[RemoteFile]:
    """The run files passing every given filter (an empty filter passes everything), plus the card.

    A filter value that matches no run is an error, not an empty download: a typo must not
    look like success.
    """
    files = list(files)
    run_files = [f for f in files if f.is_run_file]
    for label, wanted, attr in (
        ("campaign", campaigns, "campaign"),
        ("family", families, "family"),
        ("size", sizes, "domain"),
        ("run", runs, "run_id"),
    ):
        known = {getattr(f, attr).lower() for f in run_files}
        unknown = [w for w in wanted if w.lower() not in known]
        if unknown:
            shown = ", ".join(sorted({getattr(f, attr) for f in run_files})) if label != "run" else "see `README.md` on the Hub"
            raise ValueError(f"unknown {label}: {', '.join(unknown)} (available: {shown})")
    chosen = [
        f
        for f in run_files
        if _match(f.campaign, campaigns) and _match(f.family, families) and _match(f.domain, sizes) and _match(f.run_id, runs)
    ]
    if not chosen:
        raise ValueError("the filters select no run: each one matches something, but no run passes all of them")
    return chosen + [f for f in files if f.is_card]


def pending(files: Iterable[RemoteFile], root: Path) -> list[RemoteFile]:
    """The files not yet under `root` with the Hub's byte size."""
    out = []
    for f in files:
        local = Path(root) / f.path
        if not local.is_file() or local.stat().st_size != f.size:
            out.append(f)
    return out


def summarise(files: Iterable[RemoteFile]) -> list[dict]:
    """One row per campaign: runs, files, bytes — and a `total` row last."""
    rows: dict[str, dict] = {}
    for f in files:
        if not f.is_run_file:
            continue
        row = rows.setdefault(f.campaign, {"campaign": f.campaign, "runs": set(), "files": 0, "bytes": 0})
        row["runs"].add(f.run_dir)
        row["files"] += 1
        row["bytes"] += f.size
    table = [{**r, "runs": len(r["runs"])} for r in rows.values()]
    table.append(
        {
            "campaign": "total",
            "runs": sum(r["runs"] for r in table),
            "files": sum(r["files"] for r in table),
            "bytes": sum(r["bytes"] for r in table),
        }
    )
    return table


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1000:
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1000
    return f"{n_bytes:.2f} TB"


def _free_bytes(root: Path) -> int:
    probe = Path(root).resolve()
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def format_plan(
    repo_id: str, root: Path, everything: list[RemoteFile], chosen: list[RemoteFile], todo: list[RemoteFile]
) -> str:
    """What the user reads before agreeing: the dataset's size, the selection's, what is left to fetch."""
    whole, picked = summarise(everything)[-1], summarise(chosen)
    need, free = sum(f.size for f in todo), _free_bytes(root)
    lines = [
        f"dataset   https://huggingface.co/datasets/{repo_id}",
        f"          {whole['runs']} runs, {whole['files']} files, {human(whole['bytes'])} in full",
        f"into      {root}",
        "",
        f"{'campaign':<12}{'runs':>6}{'files':>8}{'size':>12}",
    ]
    lines += [f"{r['campaign']:<12}{r['runs']:>6}{r['files']:>8}{human(r['bytes']):>12}" for r in picked]
    lines += ["", f"to fetch  {len(todo)} file(s), {human(need)} ({human(sum(f.size for f in chosen) - need)} already on disk)"]
    lines.append(f"free      {human(free)} on the target filesystem" + ("  <-- NOT ENOUGH" if need > free else ""))
    return "\n".join(lines)


def patterns(todo: list[RemoteFile], chosen: list[RemoteFile]) -> list[str]:
    """The Hub's `allow_patterns` for `todo`: one glob per run directory that is missing whole, exact paths otherwise.

    A full download is ~4 000 files; one pattern per file would make the Hub client match every
    file against every pattern, and a glob over a partly present directory would re-verify the
    multi-GB file that is already there.
    """
    missing, total = Counter(f.run_dir for f in todo), Counter(f.run_dir for f in chosen)
    whole = {d for d, n in missing.items() if d is not None and n == total[d]}
    return sorted(f"{d}/*" for d in whole) + [f.path for f in todo if f.run_dir not in whole]


def _fetch(repo_id: str, root: Path, todo: list[RemoteFile], allow: list[str], workers: int) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(root), allow_patterns=allow, max_workers=workers)


def download(
    root: Path = DEFAULT_ROOT,
    *,
    repo_id: str = REPO_ID,
    campaigns: Sequence[str] = (),
    families: Sequence[str] = (),
    sizes: Sequence[str] = (),
    runs: Sequence[str] = (),
    workers: int = 8,
    dry_run: bool = False,
    confirm: Callable[[str], bool] | None = None,
    echo: Callable[[str], None] = print,
    files: list[RemoteFile] | None = None,
    fetch: Callable[[str, Path, list[RemoteFile], list[str], int], None] | None = None,
) -> dict:
    """Show what the selection costs, ask, then fetch what is missing under `root`.

    `confirm(prompt) -> bool` decides; `None` means the caller has already agreed. No filter
    selects the whole dataset. Returns `{state, n_files, bytes, root}` with `state` one of
    `present` (nothing to fetch), `dry-run`, `declined`, `downloaded`.
    """
    root = Path(root)
    everything = files if files is not None else list_files(repo_id)
    chosen = select(everything, campaigns=campaigns, families=families, sizes=sizes, runs=runs)
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
    (fetch or _fetch)(repo_id, root, todo, patterns(todo, chosen), workers)
    left = pending(chosen, root)
    if left:
        raise RuntimeError(f"{len(left)} file(s) still missing after the download, e.g. {left[0].path}; rerun to resume")
    return {**result, "state": "downloaded"}


def split_runs(paths: Iterable[Path]) -> list[str]:
    """Every run ID named by the given split files, all sections."""
    from .data import load_split

    ids: list[str] = []
    for path in paths:
        for section in load_split(path).values():
            ids.extend(section)
    return sorted(set(ids))
