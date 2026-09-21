"""Checkpoint / resume: a preempted or crashed run continues where it stopped.

The reference in every test is the same config trained uninterrupted; a resumed run has
to reproduce its per-epoch metrics exactly, which is only possible when model, optimiser,
scheduler, epoch/batch position, running loss and the shuffle order all come back.
"""

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from poreml.config import Config, EvalConfig
from poreml.train import Preempted, epoch_permutation, find_resumable, train


@pytest.fixture
def cfg(fake_root, tmp_path, interruptible):
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"train": ["tiny_0000_M1_theta120"], "val": ["tiny_0001_M1_theta130"]}))
    return Config.model_validate(
        {
            "name": "resume",
            "data": {"root": str(fake_root), "campaign": "drainage", "split": str(split)},
            "task": {"name": "next_frame", "history": 1, "params": {"roi": None}},  # 8x8x10: even, as the UNet needs
            "model": {"name": "_test_interruptible", "params": {"base": 4, "depth": 1}},
            "train": {
                "epochs": 3,
                "batch_size": 1,
                "lr": 1e-3,
                "lr_schedule": "onecycle",
                "device": "cpu",
                "checkpoint_every": 0,  # write the resume state after every batch
            },
            "out_dir": str(tmp_path / "runs"),
        }
    )


def rows(run_dir: Path) -> list[dict]:
    return list(csv.DictReader((run_dir / "metrics.csv").open()))


def assert_same_training(a: list[dict], b: list[dict]) -> None:
    assert [r["epoch"] for r in a] == [r["epoch"] for r in b]
    for ra, rb in zip(a, b, strict=True):
        for key in ra.keys() - {"epoch_seconds", "peak_memory_mb"}:  # everything but the clock
            assert float(ra[key]) == pytest.approx(float(rb[key]), rel=1e-6), key


def only_run(cfg: Config) -> Path:
    (run_dir,) = Path(cfg.out_dir).iterdir()
    return run_dir


def test_epoch_permutation_is_deterministic_per_epoch_and_skippable():
    a, b = epoch_permutation(10, seed=1, epoch=2), epoch_permutation(10, seed=1, epoch=2)
    assert a == b and sorted(a) == list(range(10))
    assert epoch_permutation(10, seed=1, epoch=3) != a
    assert epoch_permutation(10, seed=2, epoch=2) != a
    assert epoch_permutation(10, seed=1, epoch=2, skip=4) == a[4:]


def test_resume_after_a_mid_epoch_crash_reproduces_the_uninterrupted_run(cfg, interruptible, tmp_path):
    reference = rows(train(cfg.model_copy(update={"out_dir": tmp_path / "ref"})))
    assert len(reference) == 3

    interruptible.forwards, interruptible.crash_at = 0, 7  # epoch 1, second window of five
    with pytest.raises(RuntimeError, match="simulated crash"):
        train(cfg)
    run_dir = only_run(cfg)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "failed"
    assert (run_dir / "ckpts" / "resume.pt").is_file()
    assert len(rows(run_dir)) == 1  # epoch 0 finished, epoch 1 did not

    interruptible.crash_at = None
    resumed = train(cfg, resume=run_dir)

    assert resumed == run_dir
    assert_same_training(rows(run_dir), reference)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "finished"
    assert meta["progress"]["epochs_done"] == 3
    assert len(meta["run"]["segments"]) == 2
    assert not list((run_dir / "ckpts").glob("*.tmp"))


def test_sigterm_saves_the_state_and_marks_the_run_preempted(cfg, interruptible, tmp_path):
    reference = rows(train(cfg.model_copy(update={"out_dir": tmp_path / "ref"})))

    interruptible.forwards, interruptible.sigterm_at = 0, 9  # epoch 1, fourth window
    with pytest.raises(Preempted):
        train(cfg)
    run_dir = only_run(cfg)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["status"] == "preempted"
    state = torch.load(run_dir / "ckpts" / "resume.pt", weights_only=False)
    assert (state["epoch"], state["batches_done"]) == (1, 4)

    interruptible.sigterm_at = None
    train(cfg, resume=run_dir)

    assert_same_training(rows(run_dir), reference)
    assert json.loads((run_dir / "run_meta.json").read_text())["status"] == "finished"


def test_resume_refuses_a_run_trained_under_a_different_config(cfg, interruptible):
    interruptible.forwards, interruptible.crash_at = 0, 2
    with pytest.raises(RuntimeError):
        train(cfg)
    run_dir = only_run(cfg)

    other = cfg.model_copy(update={"train": cfg.train.model_copy(update={"lr": 5e-4})})
    with pytest.raises(ValueError, match="config"):
        train(other, resume=run_dir)


def test_find_resumable_returns_the_newest_unfinished_run_or_none(cfg, interruptible):
    assert find_resumable(cfg) is None  # nothing there yet

    interruptible.forwards, interruptible.crash_at = 0, 2
    with pytest.raises(RuntimeError):
        train(cfg)
    run_dir = only_run(cfg)
    assert find_resumable(cfg) == run_dir

    interruptible.crash_at = None
    train(cfg, resume=run_dir)
    assert find_resumable(cfg) is None  # finished runs are never resumed

    other = cfg.model_copy(update={"train": cfg.train.model_copy(update={"lr": 5e-4})})
    interruptible.forwards, interruptible.crash_at = 0, 2
    with pytest.raises(RuntimeError):
        train(other)
    assert find_resumable(cfg) is None  # a run of a different config is not ours to resume


def test_resume_accepts_a_changed_operational_knob(cfg, interruptible):
    """num_workers, device, log/checkpoint cadence change nothing about what is trained,
    so a run may be continued with them adjusted — lowering the loader count when the
    filesystem chokes must not strand every stopped run."""
    interruptible.forwards, interruptible.crash_at = 0, 2
    with pytest.raises(RuntimeError):
        train(cfg)
    run_dir = only_run(cfg)

    adjusted = cfg.model_copy(update={"train": cfg.train.model_copy(update={"num_workers": 1, "checkpoint_every": None})})
    assert find_resumable(adjusted) == run_dir
    assert train(adjusted, resume=run_dir) == run_dir


def test_rng_restore_reproduces_cpu_random_sequence():
    import random

    import numpy as np

    from poreml.train import _rng_state, _set_rng_state

    state = _rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(5))
    _set_rng_state(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(5), expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA checkpoint remapping")
def test_rng_restore_after_checkpoint_remapped_to_cuda(tmp_path):
    from poreml.train import _rng_state, _set_rng_state

    path = tmp_path / "rng.pt"
    torch.save(_rng_state(), path)
    expected_cpu = torch.rand(5)
    expected_cuda = torch.rand(5, device="cuda")
    state = torch.load(path, map_location="cuda", weights_only=False)
    assert state["torch"].is_cuda
    _set_rng_state(state)
    assert torch.equal(torch.rand(5), expected_cpu)
    assert torch.equal(torch.rand(5, device="cuda"), expected_cuda)


def test_resume_accepts_a_changed_periodic_eval(cfg, interruptible):
    """`train.eval` is reporting: it scores, rolls out and renders under a forked RNG and
    never touches what is trained, so a stopped run may be continued with a cheaper (or
    no) periodic eval — an abupt run whose full-resolution eval outgrew its time limit
    must not be stranded on its original eval settings."""
    interruptible.forwards, interruptible.crash_at = 0, 2
    with pytest.raises(RuntimeError):
        train(cfg)
    run_dir = only_run(cfg)

    adjusted = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"eval": EvalConfig(every=1, stride=64, rollout=False, render=False)})}
    )
    assert find_resumable(adjusted) == run_dir
    assert train(adjusted, resume=run_dir) == run_dir

    still_other = cfg.model_copy(update={"train": cfg.train.model_copy(update={"lr": 5e-4})})
    assert find_resumable(still_other) is None
