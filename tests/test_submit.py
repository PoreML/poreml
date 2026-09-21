"""case/train/submit.py: what is in the queue is never submitted again, and a run that
stopped short is resumed rather than restarted."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("submit", Path("case/train/submit.py"))
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


@pytest.fixture
def manifests(tmp_path):
    d = tmp_path / "_manifests"
    d.mkdir()
    (d / "train_20260902-100000.tsv").write_text("0\tconfigs/drainage/unet_gen.yaml\n1\tconfigs/gdl/fno_all.yaml\n")
    (d / "job_31092.tsv").symlink_to("train_20260902-100000.tsv")  # written by submit after sbatch
    return d


def test_queued_configs_maps_running_and_pending_array_tasks_to_their_configs(manifests):
    lines = ["31092_1 PENDING", "31092_0 RUNNING"]

    assert submit.queued_configs(lines, manifests) == {
        "configs/drainage/unet_gen.yaml": "RUNNING",
        "configs/gdl/fno_all.yaml": "PENDING",
    }


def test_queued_configs_ignores_jobs_without_a_manifest(manifests):
    assert submit.queued_configs(["99999_0 RUNNING"], manifests) == {}


def write_run(out_dir: Path, name: str, status: str, resume: bool) -> Path:
    run = out_dir / name
    (run / "ckpts").mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"status": status}))
    if resume:
        (run / "ckpts" / "resume.pt").write_bytes(b"")
    return run


def test_classify_reports_finished_resume_or_fresh(tmp_path):
    assert submit.classify(tmp_path / "missing") == ("fresh", None)

    out = tmp_path / "a"
    write_run(out, "x_1", "failed", resume=False)
    assert submit.classify(out) == ("fresh", None)  # died before any resume state existed

    stopped = write_run(out, "x_2", "preempted", resume=True)
    assert submit.classify(out) == ("resume", stopped)

    write_run(out, "x_3", "finished", resume=True)
    assert submit.classify(out) == ("finished", None)  # a finished run wins over anything resumable


def test_time_limit_is_a_property_of_the_model():
    # The voxel models finish a drainage run in a day and get a day, so the scheduler can
    # backfill them into gaps; AB-UPT two days; Transolver keeps the five-day ceiling.
    assert submit.time_limit("unet3d") == "1-00:00:00"
    assert submit.time_limit("fno3d") == "1-00:00:00"
    assert submit.time_limit("p3d") == "1-00:00:00"
    assert submit.time_limit("abupt") == "2-00:00:00"
    assert submit.time_limit("transolver") == "5-00:00:00"
    assert submit.time_limit("persistence") == "5-00:00:00"  # anything unlisted gets the ceiling


def test_split_throttle_shares_the_total_by_group_size_with_at_least_one_each():
    assert submit.split_throttle(20, [21, 7, 7]) == [12, 4, 4]
    assert submit.split_throttle(20, [35]) == [20]
    assert submit.split_throttle(3, [30, 1, 1]) == [1, 1, 1]
    assert submit.split_throttle(20, [1, 1]) == [10, 10]  # above an array's size is harmless: Slurm caps it there
    assert submit.split_throttle(20, []) == []
    assert submit.split_throttle(0, [21, 7]) == [0, 0]  # 0: no cap on any array


def test_plan_groups_pending_configs_by_time_limit_in_config_order(tmp_path):
    pending = [
        (tmp_path / "drainage/unet_gen.yaml", "unet3d"),
        (tmp_path / "drainage/transolver_gen.yaml", "transolver"),
        (tmp_path / "drainage/abupt_gen.yaml", "abupt"),
        (tmp_path / "gdl/fno_gen.yaml", "fno3d"),
        (tmp_path / "gdl/abupt_gen.yaml", "abupt"),
    ]
    groups = submit.plan(pending, throttle=20)
    assert [(g.time_limit, [p.name for p in g.configs], g.throttle) for g in groups] == [
        ("1-00:00:00", ["unet_gen.yaml", "fno_gen.yaml"], 8),
        ("2-00:00:00", ["abupt_gen.yaml", "abupt_gen.yaml"], 8),
        ("5-00:00:00", ["transolver_gen.yaml"], 4),
    ]
    assert sum(g.throttle for g in groups) == 20
