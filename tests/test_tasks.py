import pydantic
import pytest

from poreml.config import TaskConfig
from poreml.data import WindowDataset, discover
from poreml.registry import TASKS
from poreml.tasks import Task, build_task


@pytest.fixture
def runs(fake_root):
    return list(discover(fake_root, "drainage").values())


def test_next_frame_is_registered():
    assert "next_frame" in TASKS


def test_build_task_returns_something_satisfying_the_protocol():
    task = build_task(TaskConfig(name="next_frame"))

    assert isinstance(task, Task)
    assert task.name == "next_frame"


def test_build_task_with_unknown_name_lists_options():
    with pytest.raises(KeyError, match="next_frame"):
        build_task(TaskConfig(name="does_not_exist"))


def test_in_channels_is_solid_plus_history():
    assert build_task(TaskConfig(name="next_frame", history=1)).in_channels == 2
    assert build_task(TaskConfig(name="next_frame", history=4)).in_channels == 5


def test_default_metrics_are_declared_as_specs():
    from poreml.metrics import MetricSpec

    task = build_task(TaskConfig(name="next_frame"))
    assert task.metrics == (
        MetricSpec(error="mae"),
        MetricSpec(error="iou"),
        MetricSpec(descriptor="saturation", error="abs_err"),
    )
    assert task.target_channels == ("phi",)


def test_target_channels_follow_the_fields_param():
    assert build_task(TaskConfig(name="next_frame", params={"fields": [{"name": "p"}]})).target_channels == ("p",)


def test_dataset_is_a_window_dataset_with_the_configured_history(runs):
    task = build_task(TaskConfig(name="next_frame", history=2))

    ds = task.dataset(runs)

    assert isinstance(ds, WindowDataset)
    assert ds.history == 2
    inputs, _, _, _ = ds[0]
    assert inputs.shape[0] == task.in_channels


def test_task_params_control_field_and_stride(runs):
    task = build_task(TaskConfig(name="next_frame", params={"fields": [{"name": "p"}], "stride": 2}))

    ds = task.dataset(runs)

    assert ds.channels == ("p",)
    assert len(ds) == 3 * len(runs)


def test_fields_param_defines_channels_and_widths(runs):
    task = build_task(
        TaskConfig(
            name="next_frame",
            history=2,
            params={"fields": [{"name": "phi"}, {"name": "u", "scale": 0.01}, {"name": "p", "offset": 0.33, "scale": 0.01}]},
        )
    )

    assert task.target_channels == ("phi", "ux", "uy", "uz", "p")
    assert task.out_channels == 5
    assert task.in_channels == 1 + 2 * 5
    ds = task.dataset(runs)
    assert ds.channels == task.target_channels
    assert ds.fields[1].scale == 0.01


def test_the_retired_field_shorthand_is_an_error_that_names_its_replacement():
    with pytest.raises(ValueError, match=r"retired; write `fields: \[\{name: p\}\]`"):
        build_task(TaskConfig(name="next_frame", params={"field": "p"}))


def test_unknown_field_spec_key_is_an_error():
    with pytest.raises((ValueError, pydantic.ValidationError)):
        build_task(TaskConfig(name="next_frame", params={"fields": [{"name": "phi", "mean": 0.0}]}))


def test_dataset_stride_override(runs):
    task = build_task(TaskConfig(name="next_frame", params={"stride": 1}))

    assert len(task.dataset(runs, stride=2)) < len(task.dataset(runs))


def test_importing_poreml_populates_the_registries():
    import poreml
    from poreml.registry import DESCRIPTORS, ERRORS, MODELS, TASKS

    assert poreml.__version__
    assert TASKS.names() and MODELS.names()
    assert DESCRIPTORS.names() and ERRORS.names()


def test_task_defaults_to_the_rock_roi_and_can_be_told_otherwise(runs):
    assert build_task(TaskConfig(name="next_frame")).dataset(runs).roi == "rock"
    assert build_task(TaskConfig(name="next_frame", params={"roi": None})).dataset(runs).roi is None


def test_point_in_channels_counts_the_wall_channel():
    on = TaskConfig(name="next_frame", history=2, params={"points": {"geometry_radii": [1, 3], "wall_channel": True}})
    off = TaskConfig(name="next_frame", history=2, params={"points": {"geometry_radii": [1, 3]}})
    assert build_task(on, representation="points").in_channels == 1 + 2 + 2 * 1  # wall + G fractions + 2 phi frames
    assert build_task(off, representation="points").in_channels == 2 + 2 * 1  # default off: pre-wall checkpoints rebuild


def test_task_builds_a_point_dataset_when_the_model_consumes_points(fake_root):
    from poreml.points import PointWindowDataset

    runs = list(discover(fake_root, "drainage").values())
    cfg = TaskConfig(name="next_frame", history=2, params={"points": {"geometry_radii": [1, 3], "train_points": 4}})
    task = build_task(cfg, representation="points")
    assert task.representation == "points"
    assert task.in_channels == 2 + 2 * 1  # G=2 fractions + 2 phi frames (wall_channel defaults off)
    train_ds = task.dataset(runs[:1], training=True)
    eval_ds = task.dataset(runs[:1])
    assert isinstance(train_ds, PointWindowDataset) and train_ds.train_points == 4
    assert isinstance(eval_ds, PointWindowDataset) and eval_ds.train_points is None


def test_task_defaults_to_voxels_and_rejects_point_params_for_voxel_models():
    task = build_task(TaskConfig(name="next_frame"))
    assert task.representation == "voxel" and task.in_channels == 2
    with pytest.raises(ValueError, match="points"):
        build_task(TaskConfig(name="next_frame", params={"points": {"train_points": 4}}))
    with pytest.raises(ValueError, match="representation"):
        build_task(TaskConfig(name="next_frame"), representation="graph")


def test_conditions_in_task_params_extend_in_channels(fake_root):
    conditions = [{"name": "M", "transform": "log10"}, {"name": "theta", "scale": 180.0}]
    task = build_task(TaskConfig(name="next_frame", params={"conditions": conditions}))
    assert task.in_channels == 1 + 2 + 1  # solid + K + phi

    runs = list(discover(fake_root, "drainage").values())
    inputs, _, _, _ = task.dataset(runs)[0]
    assert inputs.shape[0] == task.in_channels

    point_task = build_task(
        TaskConfig(name="next_frame", params={"conditions": conditions, "points": {"geometry_radii": [1, 2]}}), "points"
    )
    assert point_task.in_channels == 2 + 2 + 1  # fractions + conditions + phi (wall_channel defaults off)
    pts, _, _, _ = point_task.dataset(runs)[0]
    assert pts.feats.shape[1] == point_task.in_channels


def test_unknown_condition_name_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="name"):
        build_task(TaskConfig(name="next_frame", params={"conditions": [{"name": "reynolds"}]}))
