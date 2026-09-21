"""poreml.checkpoints: a selection is exact, lands in the run-directory layout, and nothing is fetched unasked."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from poreml import checkpoints as ck
from poreml.checkpoints import RemoteFile
from poreml.cli import app


def _run(phase: str, campaign: str, cell: str, run: str, weights: tuple[str, ...], pt_bytes: int) -> list[RemoteFile]:
    base = f"{phase}/{campaign}/{cell}/ckpts/{run}"
    text = [RemoteFile(f"{base}/{name}", 100) for name in ("config.yaml", "run_meta.json", "metrics.csv", "train_log.csv")]
    return text + [RemoteFile(f"{base}/ckpts/{w}.pt", pt_bytes) for w in weights]


@pytest.fixture
def hub() -> list[RemoteFile]:
    return [
        RemoteFile(".gitattributes", 10),
        RemoteFile("README.md", 1000),
        RemoteFile("checkpoints.csv", 5000),
        *_run("train", "drainage", "unet_gen", "unet_drainage_gen_1", ("best", "last"), 40_000_000),
        *_run("train", "gdl", "abupt_all", "abupt_gdl_all_1", ("best", "last"), 42_000_000),
        *_run("train_push", "drainage", "unet_gen", "unet_drainage_gen_push_1", ("best", "last", "best_rollout"), 40_000_000),
    ]


def _write(root: Path, files: list[RemoteFile]) -> None:
    for f in files:
        path = root / f.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            fh.truncate(f.size)


def test_remote_file_parses_the_case_layout(hub):
    text, weight = hub[3], hub[-1]
    assert (weight.phase, weight.campaign, weight.model, weight.kind) == ("train_push", "drainage", "unet", "gen")
    assert weight.weight == "best_rollout" and text.weight is None and text.is_run_file
    assert weight.run_dir == "train_push/drainage/unet_gen/ckpts/unet_drainage_gen_push_1"
    assert not hub[1].is_run_file and not hub[2].is_run_file  # the card and the index stay on the Hub


def test_no_filter_selects_every_run_and_nothing_else(hub):
    assert {f.path for f in ck.select(hub)} == {f.path for f in hub if f.is_run_file}


def test_filters_intersect_and_ignore_case(hub):
    chosen = ck.select(hub, campaigns=["Drainage"], phases=["train"])
    assert {f.run_dir for f in chosen} == {"train/drainage/unet_gen/ckpts/unet_drainage_gen_1"}
    assert {f.model for f in ck.select(hub, models=["abupt"], kinds=["all"])} == {"abupt"}


def test_which_narrows_the_weights_and_keeps_the_config_beside_them(hub):
    chosen = ck.select(hub, which=["best_rollout"])
    # only the push run has one; the base runs are not selected at all, not even their text files
    assert {f.phase for f in chosen} == {"train_push"}
    assert [f.weight for f in chosen if f.weight] == ["best_rollout"]
    assert any(f.path.endswith("/config.yaml") for f in chosen)


def test_a_filter_matching_nothing_is_an_error(hub):
    with pytest.raises(ValueError, match="unknown model: unte"):
        ck.select(hub, models=["unte"])
    with pytest.raises(ValueError, match="select no checkpoint"):
        ck.select(hub, campaigns=["gdl"], phases=["train_push"])


def test_download_shows_the_size_then_asks_and_a_no_fetches_nothing(hub, tmp_path):
    shown, asked, fetched = [], [], []

    def confirm(prompt: str) -> bool:
        asked.append(prompt)
        return False

    out = ck.download(tmp_path, files=hub, confirm=confirm, echo=shown.append, fetch=lambda *a: fetched.append(a))
    assert out["state"] == "declined" and not fetched
    assert "3 runs, 7 weight files, 284.0 MB in full" in shown[0]
    assert asked == [f"Download 284.0 MB into {tmp_path}?"]


def test_download_fetches_only_what_is_missing_into_the_run_directories(hub, tmp_path):
    _write(tmp_path, [f for f in hub if f.phase == "train"])
    calls = []

    def fetch(repo_id, root, todo, workers):
        calls.append([f.path for f in todo])
        _write(root, todo)

    out = ck.download(tmp_path, files=hub, campaigns=["drainage"], confirm=None, echo=lambda _: None, fetch=fetch)
    assert out["state"] == "downloaded" and all(p.startswith("train_push/") for p in calls[0])
    # what the submitters and studies glob for: <cell>/ckpts/<run>/run_meta.json with its weights beside it
    (meta,) = (tmp_path / "train_push/drainage/unet_gen/ckpts").glob("*/run_meta.json")
    assert (meta.parent / "config.yaml").is_file() and (meta.parent / "ckpts/best_rollout.pt").is_file()
    again = ck.download(tmp_path, files=hub, campaigns=["drainage"], confirm=None, echo=lambda _: None, fetch=fetch)
    assert again["state"] == "present" and len(calls) == 1


def test_dry_run_never_asks_or_fetches(hub, tmp_path):
    never = lambda _: pytest.fail("asked")  # noqa: E731
    out = ck.download(tmp_path, files=hub, dry_run=True, confirm=never, echo=lambda _: None)
    assert out["state"] == "dry-run" and out["n_files"] == 19


def test_a_fetch_that_leaves_files_missing_is_an_error(hub, tmp_path):
    with pytest.raises(RuntimeError, match="still missing"):
        ck.download(tmp_path, files=hub, confirm=None, echo=lambda _: None, fetch=lambda *a: None)


def test_cli_declined_download_exits_1_and_yes_skips_the_prompt(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(ck, "list_files", lambda repo_id=ck.REPO_ID: hub)
    fetched = []

    def fake_fetch(repo_id, root, todo, workers):
        fetched.append(len(todo))
        _write(root, todo)

    monkeypatch.setattr(ck, "_fetch", fake_fetch)
    runner = CliRunner()
    args = ["checkpoints", "--root", str(tmp_path), "--phase", "train_push", "--which", "best_rollout"]
    declined = runner.invoke(app, args, input="n\n")
    assert declined.exit_code == 1 and not fetched and "40.0 MB" in declined.output
    agreed = runner.invoke(app, [*args, "--yes"])
    assert agreed.exit_code == 0 and fetched == [5], agreed.output
    typo = runner.invoke(app, ["checkpoints", "--root", str(tmp_path), "--model", "unte", "--yes"])
    assert typo.exit_code == 2 and "unknown model" in typo.output
