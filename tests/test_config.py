from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from poreml.config import Config, resolve_device

MINIMAL = {
    "name": "demo",
    "data": {"root": "tests/_data/tiny", "campaign": "drainage", "split": "splits/smoke.yaml"},
    "task": {"name": "next_frame"},
    "model": {"name": "persistence"},
}


def write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


STUDIES = (
    "scale",
    "shift",
    "scale_push",
    "shift_push",
)  # configs/<study>/ holds poreml.study cell/scheme/data files, not training Configs


def training_configs() -> list[Path]:
    """Every shipped training config: configs/**/*.yaml minus the evaluation-study trees."""
    return sorted(p for p in Path("configs").rglob("*.yaml") if p.parts[1] not in STUDIES)


def test_from_yaml_loads_minimal_config(tmp_path):
    cfg = Config.from_yaml(write(tmp_path, MINIMAL))

    assert cfg.name == "demo"
    assert cfg.data.campaign == "drainage"
    assert cfg.data.root == Path("tests/_data/tiny")
    assert cfg.task.name == "next_frame"
    assert cfg.model.name == "persistence"


def test_defaults_are_applied(tmp_path):
    cfg = Config.from_yaml(write(tmp_path, MINIMAL))

    assert cfg.task.history == 1
    assert cfg.train.epochs == 1
    assert cfg.train.device == "auto"
    assert cfg.out_dir == Path("runs")
    assert cfg.seed == 0


def test_metrics_default_to_none_meaning_task_defaults(tmp_path):
    cfg = Config.from_yaml(write(tmp_path, MINIMAL))

    assert cfg.metrics is None


def test_unknown_key_is_rejected(tmp_path):
    payload = {**MINIMAL, "epochs": 10}  # belongs under train, not top level

    with pytest.raises(ValidationError, match="epochs"):
        Config.from_yaml(write(tmp_path, payload))


def test_unknown_nested_key_is_rejected(tmp_path):
    payload = {**MINIMAL, "train": {"epochz": 3}}

    with pytest.raises(ValidationError, match="epochz"):
        Config.from_yaml(write(tmp_path, payload))


def test_missing_required_section_is_rejected(tmp_path):
    payload = {k: v for k, v in MINIMAL.items() if k != "model"}

    with pytest.raises(ValidationError, match="model"):
        Config.from_yaml(write(tmp_path, payload))


def test_history_must_be_positive(tmp_path):
    payload = {**MINIMAL, "task": {"name": "next_frame", "history": 0}}

    with pytest.raises(ValidationError):
        Config.from_yaml(write(tmp_path, payload))


def test_round_trip_through_to_yaml(tmp_path):
    cfg = Config.from_yaml(write(tmp_path, MINIMAL))
    out = tmp_path / "out.yaml"
    cfg.to_yaml(out)

    assert Config.from_yaml(out) == cfg


def test_resolve_device_passes_explicit_values_through():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_a_real_device():
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_train_config_accepts_grad_clip_and_lr_schedule():
    from poreml.config import TrainConfig

    assert TrainConfig().grad_clip is None and TrainConfig().lr_schedule == "constant"
    t = TrainConfig(grad_clip=1.0, lr_schedule="onecycle")
    assert t.grad_clip == 1.0 and t.lr_schedule == "onecycle"
    with pytest.raises(ValidationError):
        TrainConfig(lr_schedule="cosine")


def test_val_stride_and_rollout_stride_default_to_the_strided_protocol():
    from poreml.config import EvalConfig, TrainConfig

    assert TrainConfig().val_stride == 8  # per-epoch selection on every 8th val window
    assert TrainConfig(val_stride=None).val_stride is None  # None: the task's own stride (full pass)
    assert EvalConfig(every=2).rollout_stride == 64  # periodic rollout scored at the horizon only
    assert EvalConfig(every=2, rollout_stride=1).rollout_stride == 1  # 1: every step, the pre-knob behaviour
    with pytest.raises(ValidationError):
        TrainConfig(val_stride=0)
    with pytest.raises(ValidationError):
        EvalConfig(every=2, rollout_stride=0)


def test_every_shipped_config_conditions_on_m_and_theta_only():
    """Project default: every config conditions the model on the run's M and theta.

    Not Ca: trapping runs record no `ca` at all, and the conditioning set must be the
    same across campaigns for one protocol to span every task. `ca` stays available in
    `ConditionSpec` for one-off studies, but no shipped config uses it.
    """
    from poreml.models import representation_of
    from poreml.tasks import build_task

    paths = training_configs()
    assert len(paths) >= 56  # 6 legacy top-level + the 30-run main matrix + 15 parked uCT + 5 underfill
    for path in paths:
        cfg = Config.from_yaml(path)
        names = [c.name for c in build_task(cfg.task, representation_of(cfg.model)).conditions]
        assert names == ["M", "theta"], f"{path} conditions {names}"


def test_updated_replaces_named_keys_and_keeps_the_rest(tmp_path):
    cfg = Config.from_yaml(write(tmp_path, MINIMAL))

    new = cfg.updated(data={"campaign": "GDL", "split": "splits/shift_GDL.yaml"})

    assert new.data.campaign == "GDL"
    assert new.data.split == Path("splits/shift_GDL.yaml")
    assert new.data.root == cfg.data.root  # untouched keys survive
    assert new.task == cfg.task and new.model == cfg.model
    assert cfg.data.campaign == "drainage"  # the original is not mutated


def test_updated_merges_more_than_one_section(tmp_path):
    cfg = Config.from_yaml(write(tmp_path, MINIMAL))

    new = cfg.updated(data={"campaign": "GDL"}, train={"batch_size": 1})

    assert new.train.batch_size == 1
    assert new.train.device == cfg.train.device


def test_updated_stays_strict(tmp_path):
    cfg = Config.from_yaml(write(tmp_path, MINIMAL))

    with pytest.raises(KeyError):
        cfg.updated(bogus={"x": 1})
    with pytest.raises(ValidationError):
        cfg.updated(data={"bogus": 1})


def test_every_shipped_point_config_enables_the_wall_channel():
    """Project default: shipped point configs feed the binary wall flag as the first feature channel.

    The default lives in the configs, not in code — `wall_channel` is false by default so a
    checkpoint's saved config keeps rebuilding exactly the inputs it was trained on.
    """
    from poreml.models import representation_of
    from poreml.tasks import build_task

    for path in sorted(Path("configs").glob("*.yaml")):
        cfg = Config.from_yaml(path)
        if representation_of(cfg.model) != "points":
            continue
        task = build_task(cfg.task, "points")
        assert task.points.wall_channel, f"{path} lacks points.wall_channel: true"


def test_model_precision_is_validated():
    cfg = Config.model_validate({**MINIMAL, "model": {"name": "persistence", "precision": "bf16"}})
    assert cfg.model.precision == "bf16"
    assert Config.model_validate(MINIMAL).model.precision is None
    with pytest.raises(ValidationError):
        Config.model_validate({**MINIMAL, "model": {"name": "persistence", "precision": "fp16"}})


def test_every_shipped_config_states_its_precision_bf16_for_abupt_tf32_for_the_rest():
    """The class defaults already give AB-UPT bf16 and everything else the project default tf32
    (2026-09-10); the shipped configs say it explicitly so a checkpoint's saved config.yaml records
    what it trained in, as with `wall_channel`. Nothing shipped trains in fp32 any more."""
    for path in training_configs():
        cfg = Config.from_yaml(path)
        expected = "bf16" if cfg.model.name == "abupt" else "tf32"
        assert cfg.model.precision == expected, f"{path} does not state precision: {expected}"


def test_the_training_matrix_trains_twenty_effective_epochs():
    """Equal effective epochs (2026-09-02): 20 epochs on gen and uCT (64 runs), 10 on all (128
    runs — the union of the two, so the same run-passes and windows seen), 20 on underfill
    (its single 25-run split is not a union). The periodic eval runs 5 times per run, FNO's
    StepLR halves 4 times, P3D halves once at two thirds of the run."""
    for path in sorted(p for c in ("drainage", "gdl", "trapping", "underfill") for p in Path("configs", c).rglob("*.yaml")):
        cfg = Config.from_yaml(path)
        epochs = 10 if path.stem.endswith("_all") and path.parts[1] != "underfill" else 20
        assert cfg.train.epochs == epochs, f"{path}: epochs {cfg.train.epochs}, expected {epochs}"
        assert cfg.train.eval is not None and cfg.train.eval.every == epochs // 5, f"{path}: eval.every"
        if cfg.model.name == "fno3d":
            assert cfg.train.lr_step_size == epochs // 5, f"{path}: FNO lr_step_size {cfg.train.lr_step_size}"
        if cfg.model.name == "p3d":
            assert cfg.train.lr_step_size == round(epochs * 2 / 3), f"{path}: P3D lr_step_size {cfg.train.lr_step_size}"


def test_rollout_keyframes_default_none_and_reject_one():
    from pydantic import ValidationError

    from poreml.config import RolloutConfig

    assert RolloutConfig().keyframes is None
    assert RolloutConfig(keyframes=12).keyframes == 12
    with pytest.raises(ValidationError):
        RolloutConfig(keyframes=1)
