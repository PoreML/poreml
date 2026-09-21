"""The rollout-selected checkpoint: selection over `eval/metrics.csv`, the mark, the final stage,
and the CLI that applies it to a run finished under the old protocol."""

import csv
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from poreml import finalise
from poreml.cli import app
from poreml.config import Config
from poreml.train import train


def _eval_rows(run_dir: Path, rows: list[dict]) -> None:
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    with (run_dir / "eval" / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fake_run(tmp_path: Path, epochs_with_files: list[int], best_epoch: int, last_epoch: int) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "ckpts").mkdir(parents=True)
    for k in epochs_with_files:
        (run_dir / "ckpts" / f"epoch_{k:03d}.pt").write_bytes(f"epoch {k}".encode())
    (run_dir / "ckpts" / "best.pt").write_bytes(f"epoch {best_epoch}".encode())
    (run_dir / "ckpts" / "last.pt").write_bytes(f"epoch {last_epoch}".encode())
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "status": "finished",
                "progress": {
                    "best": {"epoch": best_epoch, "mae@phi": 0.1},
                    "last": {"epoch": last_epoch},
                    "epochs_total": last_epoch + 1,
                },
            }
        )
    )
    return run_dir


def test_candidates_are_the_epochs_with_weights_on_disk(tmp_path):
    run_dir = _fake_run(tmp_path, epochs_with_files=[3, 4], best_epoch=1, last_epoch=4)
    assert finalise.candidate_checkpoints(run_dir) == {
        1: run_dir / "ckpts" / "best.pt",
        3: run_dir / "ckpts" / "epoch_003.pt",
        4: run_dir / "ckpts" / "epoch_004.pt",  # the per-epoch file wins over last.pt for the same epoch
    }


def test_selection_takes_the_lowest_rollout_value_among_candidates_ties_to_the_earlier_epoch(tmp_path):
    run_dir = _fake_run(tmp_path, epochs_with_files=[], best_epoch=0, last_epoch=4)
    _eval_rows(
        run_dir,
        [
            {"epoch": 0, "mae@phi": 0.1, "rollout/mae@phi": 0.30},
            {"epoch": 2, "mae@phi": 0.2, "rollout/mae@phi": 0.10},  # best rollout, but no weights on disk
            {"epoch": 4, "mae@phi": 0.3, "rollout/mae@phi": 0.30},
        ],
    )
    chosen = finalise.select(run_dir, "mae@phi")
    assert chosen == {"epoch": 0, "rollout/mae@phi": 0.30, "checkpoint": str(run_dir / "ckpts" / "best.pt"), "candidates": 2}


def test_selection_skips_blank_and_nan_rows_and_returns_none_without_rollout_rows(tmp_path):
    run_dir = _fake_run(tmp_path, epochs_with_files=[0, 1, 2], best_epoch=0, last_epoch=2)
    _eval_rows(run_dir, [{"epoch": 0, "mae@phi": 0.1}, {"epoch": 1, "mae@phi": 0.1}])
    assert finalise.select(run_dir, "mae@phi") is None
    _eval_rows(
        run_dir,
        [
            {"epoch": 0, "mae@phi": 0.1, "rollout/mae@phi": ""},
            {"epoch": 1, "mae@phi": 0.1, "rollout/mae@phi": "nan"},
            {"epoch": 2, "mae@phi": 0.1, "rollout/mae@phi": 0.5},
        ],
    )
    assert finalise.select(run_dir, "mae@phi")["epoch"] == 2
    assert finalise.select(tmp_path / "nowhere", "mae@phi") is None  # no eval at all


def test_mark_copies_the_winner_to_best_rollout(tmp_path):
    run_dir = _fake_run(tmp_path, epochs_with_files=[0, 1], best_epoch=0, last_epoch=1)
    finalise.mark(run_dir, run_dir / "ckpts" / "epoch_001.pt")
    marked = run_dir / "ckpts" / "best_rollout.pt"
    assert marked.is_file() and not marked.is_symlink() and marked.read_bytes() == b"epoch 1"


def test_cli_finalises_a_run_trained_under_the_old_protocol(fake_root, tmp_path):
    """A run with evals but only best.pt / last.pt (every run before 2026-09-10) gets selected
    between those two and evaluated under final/."""
    split = tmp_path / "split.yaml"
    split.write_text("train: [tiny_0000_M1_theta120]\nval: [tiny_0001_M1_theta130]\ntest: [tiny_0002_M10_theta140]\n")
    cfg = Config.model_validate(
        {
            "name": "old",
            "data": {"root": str(fake_root), "campaign": "drainage", "split": str(split)},
            "task": {"name": "next_frame", "history": 1},
            "model": {"name": "persistence"},
            "train": {
                "epochs": 2,
                "batch_size": 1,
                "device": "cpu",
                "max_batches": 2,
                "eval": {"every": 1, "rollout": True, "render": False},
            },
            "rollout": {"horizon": 4},  # the fixture runs hold 6 frames: the horizon is reachable
            "out_dir": str(tmp_path / "runs"),
        }
    )
    run_dir = train(cfg)
    # strip the run back to the old layout: no per-epoch files, no mark, no final stage
    for p in (run_dir / "ckpts").glob("epoch_*.pt"):
        p.unlink()
    (run_dir / "ckpts" / "best_rollout.pt").unlink()
    shutil.rmtree(run_dir / "final")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    meta["progress"].pop("best_rollout"), meta["progress"].pop("final")
    (run_dir / "run_meta.json").write_text(json.dumps(meta))

    result = CliRunner().invoke(app, ["finalise", str(run_dir)])

    assert result.exit_code == 0, result.output
    meta = json.loads((run_dir / "run_meta.json").read_text())
    chosen = meta["progress"]["best_rollout"]
    assert chosen["epoch"] in (0, 1) and chosen["candidates"] == 2  # best.pt (epoch 0: persistence never improves) and last.pt
    assert (run_dir / "ckpts" / "best_rollout.pt").exists()
    assert (run_dir / "final" / "results_val.json").exists() and (run_dir / "final" / "rollout_val.json").exists()
    assert f"epoch {chosen['epoch']}" in result.output


def test_progress_metric_prefers_rel_mae_phi_then_any_rel_mae_then_the_primary():
    assert finalise.progress_metric(["mae@phi", "rel_mae@phi", "iou@phi"], "mae@phi") == "rel_mae@phi"
    assert finalise.progress_metric(["mae@p", "rel_mae@p"], "mae@p") == "rel_mae@p"
    assert finalise.progress_metric(["mae", "iou"], "mae") == "mae"


def test_progress_plot_is_drawn_from_the_two_csvs(tmp_path):
    pytest.importorskip("matplotlib")
    from poreml import viz

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "rel_mae@phi"])
        w.writeheader()
        w.writerows([{"epoch": 0, "rel_mae@phi": 0.3}, {"epoch": 1, "rel_mae@phi": 0.2}, {"epoch": 2, "rel_mae@phi": 0.25}])
    out = viz.plot_progress(run_dir / "metrics.csv", run_dir / "eval" / "metrics.csv", run_dir / "progress.svg", "rel_mae@phi")
    assert out.exists()  # no eval rows yet: the rollout panel is empty, the one-step panel drawn
    _eval_rows(
        run_dir,
        [
            {"epoch": 1, "rel_mae@phi": 0.2, "rollout/rel_mae@phi": 0.6},
            {"epoch": 2, "rel_mae@phi": 0.25, "rollout/rel_mae@phi": 0.4},
        ],
    )
    assert viz.plot_progress(
        run_dir / "metrics.csv", run_dir / "eval" / "metrics.csv", run_dir / "progress.svg", "rel_mae@phi"
    ).exists()
