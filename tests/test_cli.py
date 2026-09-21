import json

import pytest
import yaml
from typer.testing import CliRunner

from poreml.cli import app

runner = CliRunner()


def write_cfg(fake_root, tmp_path, metrics=None) -> str:
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {
                "train": ["tiny_0000_M1_theta120"],
                "val": ["tiny_0001_M1_theta130"],
                "test": ["tiny_0002_M10_theta140"],
            }
        )
    )
    payload = {
        "name": "cli",
        "data": {"root": str(fake_root), "campaign": "drainage", "split": str(split)},
        "task": {"name": "next_frame"},
        "model": {"name": "persistence"},
        "train": {"epochs": 1, "device": "cpu", "max_batches": 1},
        "out_dir": str(tmp_path / "runs"),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(payload))
    return str(cfg_path)


def test_ls_tasks_lists_the_registered_task():
    result = runner.invoke(app, ["ls", "tasks"])

    assert result.exit_code == 0
    assert "next_frame" in result.stdout


def test_ls_models_and_metrics():
    assert "persistence" in runner.invoke(app, ["ls", "models"]).stdout

    listing = runner.invoke(app, ["ls", "metrics"]).stdout
    assert "saturation" in listing
    assert "abs_err" in listing


def test_ls_unknown_kind_exits_nonzero():
    result = runner.invoke(app, ["ls", "widgets"])

    assert result.exit_code != 0


def test_train_command_reports_the_run_directory(fake_root, tmp_path):
    result = runner.invoke(app, ["train", "-c", write_cfg(fake_root, tmp_path)])

    assert result.exit_code == 0, result.output
    assert "cli_" in result.stdout


def test_eval_command_prints_metrics(fake_root, tmp_path):
    cfg_path = write_cfg(fake_root, tmp_path)
    train_result = runner.invoke(app, ["train", "-c", cfg_path])
    run_dir = train_result.stdout.strip().splitlines()[-1]

    result = runner.invoke(app, ["eval", "-c", cfg_path, "--ckpt", f"{run_dir}/ckpts/best.pt"])

    assert result.exit_code == 0, result.output
    assert "mae" in result.stdout


def test_eval_prints_a_distribution_metric_as_a_json_array(fake_root, tmp_path):
    # A `(B, K)` metric means to a list; the printed line has to stay valid JSON rather
    # than crash the float format string.
    cfg_path = write_cfg(
        fake_root,
        tmp_path,
        metrics=[{"error": "mae"}, {"descriptor": "hist", "error": "pred", "bins": 4, "range": [-1.0, 1.0]}],
    )
    run_dir = runner.invoke(app, ["train", "-c", cfg_path]).stdout.strip().splitlines()[-1]

    result = runner.invoke(app, ["eval", "-c", cfg_path, "--ckpt", f"{run_dir}/ckpts/best.pt"])

    assert result.exit_code == 0, result.output
    line = next(line for line in result.stdout.splitlines() if line.startswith("hist/pred"))
    shown = json.loads(line.split(maxsplit=1)[1])
    assert isinstance(shown, list) and len(shown) == 4


def test_eval_out_dir_option_controls_where_results_are_written(fake_root, tmp_path):
    cfg_path = write_cfg(fake_root, tmp_path)
    run_dir = runner.invoke(app, ["train", "-c", cfg_path]).stdout.strip().splitlines()[-1]
    out_dir = tmp_path / "scores"

    result = runner.invoke(app, ["eval", "-c", cfg_path, "--ckpt", f"{run_dir}/ckpts/best.pt", "--out-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert (out_dir / "results_test.json").is_file()


def test_eval_with_a_bad_config_path_exits_nonzero():
    result = runner.invoke(app, ["eval", "-c", "does/not/exist.yaml"])

    assert result.exit_code != 0


def test_runs_command_lists_every_discovered_run_with_status_and_conditions(fake_root, tmp_path):
    from tests.fixtures import write_fake_run

    write_fake_run(fake_root, "drainage", "live_0009_M1_theta120", status="running")
    cfg = write_cfg(fake_root, tmp_path)

    result = runner.invoke(app, ["runs", "-c", cfg])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == ["run_id", "split", "status", "finish", "family", "geometry", "M", "theta", "ca"]
    rows = {line.split()[0]: line.split() for line in lines[1:]}
    assert set(rows) == {"tiny_0000_M1_theta120", "tiny_0001_M1_theta130", "tiny_0002_M10_theta140", "live_0009_M1_theta120"}
    assert rows["tiny_0002_M10_theta140"][1:] == ["test", "finished", "pv_cap", "tiny", "tiny_0002", "10", "140", "1e-05"]
    assert rows["live_0009_M1_theta120"][1:4] == ["-", "running", "-"]


def test_runs_command_can_be_limited_to_one_split_section(fake_root, tmp_path):
    cfg = write_cfg(fake_root, tmp_path)

    result = runner.invoke(app, ["runs", "-c", cfg, "--split", "val"])

    assert result.exit_code == 0, result.output
    assert [line.split()[0] for line in result.output.splitlines()[1:]] == ["tiny_0001_M1_theta130"]


def test_rollout_command_writes_the_artifact_and_prints_the_horizon_values(fake_root, tmp_path):
    cfg = write_cfg(fake_root, tmp_path)
    run_dir = runner.invoke(app, ["train", "-c", cfg]).output.strip().splitlines()[-1]

    result = runner.invoke(app, ["rollout", "-c", cfg, "--ckpt", f"{run_dir}/ckpts/best.pt", "--horizon", "3"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "runs").glob("*/ckpts/rollout_test.json")
    written = json.loads(next((tmp_path / "runs").glob("*/ckpts/rollout_test.json")).read_text())
    assert written["horizon"] == 3
    assert "mae" in result.output and "h=3" in result.output


def test_render_command_writes_frames_gif_and_json(fake_root, tmp_path):
    pytest.importorskip("pyvista")
    cfg = write_cfg(fake_root, tmp_path)
    out = tmp_path / "viz"

    result = runner.invoke(
        app, ["render", "-c", cfg, "--run", "tiny_0002_M10_theta140", "--out-dir", str(out), "--horizon", "2"]
    )

    assert result.exit_code == 0, result.output
    frames = sorted((out / "frames").glob("frame_*.png"))
    # t0 is a quarter of the way in (frame 1 of 6), so the rendered targets are frames 2 and 3: steps 200, 300.
    assert [f.name for f in frames] == ["frame_00000200.png", "frame_00000300.png"]
    assert (out / "rollout_tiny_0002_M10_theta140.gif").is_file()
    payload = json.loads((out / "rollout_tiny_0002_M10_theta140.json").read_text())
    assert payload["horizon"] == 2 and len(payload["values"]["mae"]) == 2


def test_the_unet_case_config_resolves_against_the_task_and_registries():
    from poreml.config import Config
    from poreml.metrics import resolve
    from poreml.tasks import build_task

    cfg = Config.from_yaml("configs/drainage/unet_gen.yaml")
    task = build_task(cfg.task)

    assert cfg.task.history == 1
    assert task.target_channels == ("phi", "ux", "uy", "uz", "p")
    assert task.in_channels == 8 and task.out_channels == 5  # solid | M theta | 5 fields
    metrics = resolve(cfg.metrics, task.target_channels)
    assert metrics[0].name == "mae@phi"
    assert {"curvature_hist/w1@phi", "curvature_hist/w1_norm@phi", "mae@p", "mae@ux"} <= {m.name for m in metrics}
    assert cfg.train.eval is not None and cfg.train.eval.every >= 1


# --- train --resume -------------------------------------------------------------------

from pathlib import Path  # noqa: E402


def write_interruptible_cfg(fake_root, tmp_path) -> str:
    cfg_path = write_cfg(fake_root, tmp_path)
    payload = yaml.safe_load(Path(cfg_path).read_text())
    payload["task"]["params"] = {"roi": None}  # the fixture's 8x8x10 domain is even, as the UNet needs
    payload["model"] = {"name": "_test_interruptible", "params": {"base": 4, "depth": 1}}
    payload["train"] = {"epochs": 2, "device": "cpu", "checkpoint_every": 0}
    Path(cfg_path).write_text(yaml.safe_dump(payload))
    return cfg_path


def test_train_resume_auto_continues_the_run_that_stopped_short(fake_root, tmp_path, interruptible):
    cfg_path = write_interruptible_cfg(fake_root, tmp_path)
    interruptible.forwards, interruptible.crash_at = 0, 3
    crashed = runner.invoke(app, ["train", "-c", cfg_path])
    assert crashed.exit_code != 0
    (run_dir,) = (tmp_path / "runs").iterdir()

    interruptible.crash_at = None
    resumed = runner.invoke(app, ["train", "-c", cfg_path, "--resume", "auto"])

    assert resumed.exit_code == 0, resumed.output
    assert resumed.stdout.strip().splitlines()[-1] == str(run_dir)
    assert json.loads((run_dir / "run_meta.json").read_text())["status"] == "finished"
    assert [p.name for p in (tmp_path / "runs").iterdir()] == [run_dir.name]  # nothing new was started


def test_train_resume_auto_starts_fresh_when_nothing_stopped_short(fake_root, tmp_path):
    cfg_path = write_cfg(fake_root, tmp_path)
    first = runner.invoke(app, ["train", "-c", cfg_path]).stdout.strip().splitlines()[-1]

    second = runner.invoke(app, ["train", "-c", cfg_path, "--resume", "auto"])

    assert second.exit_code == 0, second.output
    assert second.stdout.strip().splitlines()[-1] != first
    assert len(list((tmp_path / "runs").iterdir())) == 2


# --- convert ----------------------------------------------------------------------------

from tests.test_convert import write_source_run  # noqa: E402


def test_convert_run_command_converts_one_run_and_prints_its_record(tmp_path):
    src, dst = tmp_path / "solver_case", tmp_path / "data" / "case"
    src.mkdir()
    run = write_source_run(src)

    result = runner.invoke(app, ["convert", "run", str(run), str(dst / run.relative_to(src))])

    assert result.exit_code == 0, result.output
    assert (dst / run.relative_to(src) / f"{run.name}.h5").is_file()
    assert "verified" in result.stdout


def test_convert_scan_and_ledger_commands(tmp_path):
    src, dst = tmp_path / "solver_case", tmp_path / "data" / "case"
    src.mkdir()
    write_source_run(src)
    write_source_run(src, size="512", run_id="drain_blob512_0000")  # outside SCOPE_SIZES (the 256 classes are in it)

    scan = runner.invoke(app, ["convert", "scan", "--src", str(src), "--dst", str(dst)])
    ledger = runner.invoke(app, ["convert", "ledger", "--src", str(src), "--dst", str(dst)])

    assert scan.exit_code == 0 and "pending" in scan.stdout and "out of scope" in scan.stdout, scan.output
    assert ledger.exit_code == 0 and (src / "h5_todo.md").is_file(), ledger.output


def test_inference_then_metric_commands(fake_root, tmp_path):
    cfg = write_cfg(fake_root, tmp_path)
    run_dir = runner.invoke(app, ["train", "-c", cfg]).output.strip().splitlines()[-1]
    ckpt = f"{run_dir}/ckpts/best.pt"
    out = tmp_path / "staged"

    stage = ["-c", cfg, "--ckpt", ckpt, "--split", "test", "--out-dir", str(out)]
    inf = runner.invoke(app, ["inference", *stage, "--horizon", "2", "--stride", "2", "--fields", "phi"])
    assert inf.exit_code == 0, inf.output
    assert (out / "inference" / "frames").is_dir() and (out / "inference" / "inference.json").is_file()
    assert (out / "inference.csv").is_file() and "windows" in inf.output

    met = runner.invoke(app, ["metric", *stage, "--workers", "1"])
    assert met.exit_code == 0, met.output
    scored = json.loads((out / "inference" / "metrics_test.json").read_text())
    assert scored["metric"]["workers"] == 1 and (out / "inference" / "metrics_test.csv").is_file()


def test_rollout_has_one_job_and_the_stage_flags_are_gone(fake_root, tmp_path):
    cfg = write_cfg(fake_root, tmp_path)
    for flag in ("--inference", "--metric", "--shared"):
        result = runner.invoke(app, ["rollout", "-c", cfg, flag])
        assert result.exit_code == 2 and "No such option" in result.output, flag


def test_inference_exits_75_when_interrupted(fake_root, tmp_path, monkeypatch):
    import poreml.inference as inference_mod
    from poreml.rollout import Interrupted

    def interrupted(*args, **kwargs):
        raise Interrupted("stopped")

    monkeypatch.setattr(inference_mod, "rollout_inference", interrupted)
    cfg = write_cfg(fake_root, tmp_path)
    result = runner.invoke(app, ["inference", "-c", cfg])
    assert result.exit_code == 75


def test_convert_anonymise_command_reports_and_applies(tmp_path):
    from tests.test_convert import _converted_copy_with_identity

    dst, dst_run, h5 = _converted_copy_with_identity((tmp_path / "solver_case", tmp_path / "data" / "case"))
    (tmp_path / "solver_case").mkdir(exist_ok=True)
    dry = runner.invoke(app, ["convert", "anonymise", str(dst), "--replace", "alice=poreml_author", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "dry run" in dry.output and "alice" in (dst_run / "run_meta.json").read_text()
    result = runner.invoke(app, ["convert", "anonymise", str(dst), "--replace", "alice=poreml_author"])
    assert result.exit_code == 0, result.output
    assert "alice" not in (dst_run / "run_meta.json").read_text()
    assert "1 run" in result.output
