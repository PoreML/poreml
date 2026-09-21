"""poreml.download: nothing is fetched before the size is shown and agreed to, and a selection is exact."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from poreml import download as dl
from poreml.cli import app
from poreml.download import RemoteFile


def _run(campaign: str, family: str, size: str, run_id: str, h5_bytes: int) -> list[RemoteFile]:
    base = f"{campaign}/runs/{family}/{size}/{run_id}"
    return [RemoteFile(f"{base}/{run_id}.h5", h5_bytes), RemoteFile(f"{base}/run_meta.json", 100)]


@pytest.fixture
def hub() -> list[RemoteFile]:
    return [
        RemoteFile(".gitattributes", 10),
        RemoteFile("README.md", 1000),
        RemoteFile("assets/drainage.png", 5000),
        *_run("drainage", "bentheimer", "128", "drain_a", 3_000_000_000),
        *_run("drainage", "sphere", "256", "drain_b", 20_000_000_000),
        *_run("GDL", "fiber", "128x128x64", "gdl_a", 2_000_000_000),
    ]


def _write(root: Path, files: list[RemoteFile]) -> None:
    for f in files:
        path = root / f.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            fh.truncate(f.size)  # sparse: the size check is all `pending` looks at


def test_remote_file_parses_the_hub_layout(hub):
    h5 = hub[3]
    assert (h5.campaign, h5.family, h5.domain, h5.run_id) == ("drainage", "bentheimer", "128", "drain_a")
    assert h5.run_dir == "drainage/runs/bentheimer/128/drain_a"
    assert hub[1].is_card and hub[2].is_card and not hub[0].is_card and not hub[1].is_run_file


def test_no_filter_selects_every_run_and_the_card(hub):
    chosen = dl.select(hub)
    assert {f.path for f in chosen} == {f.path for f in hub} - {".gitattributes"}


def test_filters_intersect_and_ignore_case(hub):
    chosen = dl.select(hub, campaigns=["gdl"])
    assert {f.run_id for f in chosen if f.is_run_file} == {"gdl_a"}
    chosen = dl.select(hub, campaigns=["drainage"], sizes=["256"])
    assert {f.run_id for f in chosen if f.is_run_file} == {"drain_b"}
    assert any(f.path == "README.md" for f in chosen)  # the card rides along with every selection


def test_a_filter_matching_nothing_is_an_error(hub):
    with pytest.raises(ValueError, match="unknown campaign: drainge"):
        dl.select(hub, campaigns=["drainge"])
    with pytest.raises(ValueError, match="unknown run"):
        dl.select(hub, runs=["drain_zzz"])
    with pytest.raises(ValueError, match="select no run"):
        dl.select(hub, campaigns=["GDL"], families=["sphere"])


def test_summarise_counts_runs_files_and_bytes(hub):
    rows = {r["campaign"]: r for r in dl.summarise(dl.select(hub))}
    assert rows["drainage"] == {"campaign": "drainage", "runs": 2, "files": 4, "bytes": 23_000_000_200}
    assert rows["total"]["runs"] == 3 and rows["total"]["bytes"] == 25_000_000_300


def test_pending_skips_files_already_there_with_the_right_size(hub, tmp_path):
    chosen = dl.select(hub, runs=["drain_a"])
    _write(tmp_path, [f for f in chosen if f.path.endswith(".json")])
    (tmp_path / "README.md").write_text("short")  # wrong size: fetched again
    left = {f.path for f in dl.pending(chosen, tmp_path)}
    assert left == {"drainage/runs/bentheimer/128/drain_a/drain_a.h5", "README.md", "assets/drainage.png"}


def test_download_shows_the_size_then_asks_and_a_no_fetches_nothing(hub, tmp_path):
    shown, asked, fetched = [], [], []

    def confirm(prompt: str) -> bool:
        asked.append(prompt)
        return False

    out = dl.download(tmp_path, files=hub, confirm=confirm, echo=shown.append, fetch=lambda *a: fetched.append(a))
    assert out["state"] == "declined" and not fetched
    assert "25.0 GB in full" in shown[0] and "to fetch  8 file(s), 25.0 GB" in shown[0]
    assert asked == [f"Download 25.0 GB into {tmp_path}?"]


def test_download_fetches_only_what_is_missing(hub, tmp_path):
    _write(tmp_path, [f for f in hub if f.run_id == "drain_a" or f.is_card])
    calls = []

    def fetch(repo_id, root, todo, allow, workers):
        calls.append(allow)
        _write(root, todo)

    out = dl.download(tmp_path, files=hub, campaigns=["drainage"], confirm=None, echo=lambda _: None, fetch=fetch)
    assert out["state"] == "downloaded" and out["bytes"] == 20_000_000_100
    assert calls == [["drainage/runs/sphere/256/drain_b/*"]]  # a run missing whole is one glob
    again = dl.download(tmp_path, files=hub, campaigns=["drainage"], confirm=None, echo=lambda _: None, fetch=fetch)
    assert again["state"] == "present" and len(calls) == 1


def test_patterns_name_exact_files_in_a_partly_present_run(hub, tmp_path):
    chosen = dl.select(hub, runs=["drain_a"])
    _write(tmp_path, [f for f in chosen if f.path.endswith(".h5")])
    todo = dl.pending(chosen, tmp_path)
    expected = ["drainage/runs/bentheimer/128/drain_a/run_meta.json", "README.md", "assets/drainage.png"]
    assert dl.patterns(todo, chosen) == expected


def test_dry_run_never_asks_or_fetches(hub, tmp_path):
    never = lambda _: pytest.fail("asked")  # noqa: E731
    out = dl.download(tmp_path, files=hub, dry_run=True, confirm=never, echo=lambda _: None)
    assert out["state"] == "dry-run" and out["n_files"] == 8


def test_a_fetch_that_leaves_files_missing_is_an_error(hub, tmp_path):
    with pytest.raises(RuntimeError, match="still missing"):
        dl.download(tmp_path, files=hub, confirm=None, echo=lambda _: None, fetch=lambda *a: None)


def test_split_runs_reads_every_section(tmp_path):
    split = tmp_path / "s.yaml"
    split.write_text("train: [b, a]\nval: [c]\ntest: [d]\n")
    assert dl.split_runs([split]) == ["a", "b", "c", "d"]


def test_cli_declined_download_exits_1_and_yes_skips_the_prompt(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "list_files", lambda repo_id=dl.REPO_ID: hub)
    fetched = []

    def fake_fetch(repo_id, root, todo, allow, workers):
        fetched.append(len(todo))
        _write(root, todo)

    monkeypatch.setattr(dl, "_fetch", fake_fetch)
    runner = CliRunner()
    declined = runner.invoke(app, ["download", "--root", str(tmp_path), "--run", "drain_a"], input="n\n")
    assert declined.exit_code == 1 and not fetched and "3.0 GB" in declined.output
    agreed = runner.invoke(app, ["download", "--root", str(tmp_path), "--run", "drain_a", "--yes"])
    assert agreed.exit_code == 0 and fetched == [4], agreed.output
    typo = runner.invoke(app, ["download", "--root", str(tmp_path), "--campaign", "drainge", "--yes"])
    assert typo.exit_code == 2 and "unknown campaign" in typo.output
