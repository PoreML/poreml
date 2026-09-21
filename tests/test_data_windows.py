import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from poreml.data import SOLID_FILL, Trajectory, WindowDataset, discover

FD_DIR = Path("/proc/self/fd")


def count_open_h5() -> int:
    """How many HDF5 files this process currently holds open (Linux only)."""
    total = 0
    for fd in FD_DIR.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:  # the fd backing the directory scan can vanish mid-iteration
            continue
        if target.endswith(".h5"):
            total += 1
    return total


@pytest.fixture
def runs(fake_root):
    return list(discover(fake_root, "drainage").values())


def test_trajectory_lists_steps_in_order(runs):
    traj = Trajectory(runs[0].h5_path)

    assert traj.steps == ("000000000", "000000100", "000000200", "000000300", "000000400", "000000500")


def test_trajectory_rock_is_bool_with_true_meaning_solid(runs):
    traj = Trajectory(runs[0].h5_path)

    assert traj.rock.dtype == np.bool_
    assert traj.rock.shape == (8, 8, 10)
    assert traj.rock.any()


def test_frame_replaces_nan_in_solid_with_fill(runs):
    traj = Trajectory(runs[0].h5_path)

    phi = traj.frame("000000000", "phi")

    assert not np.isnan(phi).any()
    assert phi.dtype == np.float32
    assert np.all(phi[traj.rock] == SOLID_FILL)


def test_frame_preserves_pore_values(runs):
    traj = Trajectory(runs[0].h5_path)
    pore = ~traj.rock

    phi = traj.frame("000000100", "phi")

    # The fixture invades slabs 0..1 by step index 1.
    invaded = phi[:, :, :2][pore[:, :, :2]]
    assert np.all(invaded == 1.0)


def test_trajectory_survives_pickling(runs):
    traj = Trajectory(runs[0].h5_path)
    _ = traj.rock  # force the file open

    revived = pickle.loads(pickle.dumps(traj))

    assert revived.steps == traj.steps
    assert np.array_equal(revived.rock, traj.rock)


def test_window_dataset_length_counts_valid_windows(runs):
    # 6 steps, history 1 -> windows at t = 0..4, i.e. 5 per run.
    ds = WindowDataset(runs[:1], history=1)

    assert len(ds) == 5


def test_window_dataset_length_with_longer_history(runs):
    # history 3 -> the first valid t is 2, so t = 2..4, i.e. 3 per run.
    ds = WindowDataset(runs[:1], history=3)

    assert len(ds) == 3


def test_window_dataset_spans_all_runs(runs):
    ds = WindowDataset(runs, history=1)

    assert len(ds) == 5 * len(runs)


def test_getitem_shapes_and_dtypes(runs):
    ds = WindowDataset(runs[:1], history=2, roi=None)

    inputs, target, mask, meta = ds[0]

    assert inputs.shape == (3, 8, 8, 10)  # 1 solid channel + 2 history frames
    assert target.shape == (1, 8, 8, 10)
    assert mask.shape == (1, 8, 8, 10)
    assert inputs.dtype == torch.float32
    assert target.dtype == torch.float32
    assert mask.dtype == torch.bool


def test_channel_zero_is_the_solid_mask(runs):
    ds = WindowDataset(runs[:1], history=1, roi=None)
    traj = Trajectory(runs[0].h5_path)

    inputs, _, mask, _ = ds[0]

    assert torch.equal(inputs[0] > 0.5, torch.from_numpy(traj.rock))
    assert torch.equal(mask[0], torch.from_numpy(~traj.rock))


def test_history_channels_are_oldest_first_and_target_is_the_next_frame(runs):
    ds = WindowDataset(runs[:1], history=2, roi=None)
    traj = Trajectory(runs[0].h5_path)

    inputs, target, _, _ = ds[0]  # t = 1, so history covers steps 0 and 1, target is step 2

    assert torch.equal(inputs[1], torch.from_numpy(traj.frame(traj.steps[0])))
    assert torch.equal(inputs[2], torch.from_numpy(traj.frame(traj.steps[1])))
    assert torch.equal(target[0], torch.from_numpy(traj.frame(traj.steps[2])))


def test_no_nan_reaches_the_tensors(runs):
    ds = WindowDataset(runs, history=2)

    for i in range(len(ds)):
        inputs, target, _, _ = ds[i]
        assert not torch.isnan(inputs).any()
        assert not torch.isnan(target).any()


def test_stride_subsamples_windows(runs):
    ds = WindowDataset(runs[:1], history=1, stride=2)

    assert len(ds) == 3  # t = 0, 2, 4


def test_trajectory_reopens_when_the_owning_process_changed(runs):
    traj = Trajectory(runs[0].h5_path)
    inherited = traj.file
    expected = traj.frame(traj.steps[0])

    traj._pid = -1  # stand in for "this handle was opened before a fork"

    assert traj.file is not inherited
    assert np.array_equal(traj.frame(traj.steps[0]), expected)


@pytest.mark.skipif(not FD_DIR.is_dir(), reason="counting open files needs /proc")
def test_window_dataset_leaves_no_open_h5_handles(runs):
    # Construction counts steps, which opens every run. Those handles must be closed
    # again: forked DataLoader workers inherit whatever the parent still holds open, and
    # HDF5 is not fork-safe.
    before = count_open_h5()

    ds = WindowDataset(runs, history=1)

    assert count_open_h5() == before
    _ = ds[0]  # and reading reopens lazily rather than failing
    assert count_open_h5() == before + 1


@pytest.mark.skipif(not FD_DIR.is_dir(), reason="counting open files needs /proc")
def test_window_dataset_closes_handles_even_when_construction_fails(runs):
    before = count_open_h5()

    with pytest.raises(ValueError, match="history"):
        WindowDataset(runs, history=99)

    assert count_open_h5() == before


def test_forked_dataloader_workers_read_the_same_data_as_the_parent(runs):
    ds = WindowDataset(runs[:1], history=1)
    expected = [ds[i] for i in range(len(ds))]  # leaves the parent holding an open handle

    loader = DataLoader(ds, batch_size=1, num_workers=2)
    got = list(loader)

    assert len(got) == len(expected)
    for (inputs, target, mask, _), (want_in, want_t, want_m, _) in zip(got, expected, strict=True):
        assert torch.equal(inputs[0], want_in)
        assert torch.equal(target[0], want_t)
        assert torch.equal(mask[0], want_m)


def test_dataset_works_through_a_dataloader(runs):
    ds = WindowDataset(runs[:1], history=1, roi=None)
    loader = DataLoader(ds, batch_size=2)

    inputs, target, mask, meta = next(iter(loader))

    assert inputs.shape == (2, 2, 8, 8, 10)
    assert target.shape == (2, 1, 8, 8, 10)
    assert mask.shape == (2, 1, 8, 8, 10)
    assert meta["run"].tolist() == [0, 0] and meta["t"].tolist() == [0, 1]


def test_history_longer_than_the_trajectory_raises(runs):
    with pytest.raises(ValueError, match="history"):
        WindowDataset(runs[:1], history=99)


def test_default_roi_crops_to_the_rock_span_along_the_flow_axis(runs):
    # The fixture's regions are {"inbuf": [0, 1], "rock": [1, 10]}: the rock ROI drops the
    # first slab, so the last axis shrinks from 10 to 9.
    ds = WindowDataset(runs[:1], history=1)  # roi defaults to "rock"
    traj = Trajectory(runs[0].h5_path)

    inputs, target, mask, _ = ds[0]

    assert inputs.shape == (2, 8, 8, 9)
    assert target.shape == (1, 8, 8, 9)
    assert torch.equal(mask[0], torch.from_numpy(~traj.rock[:, :, 1:]))
    assert torch.equal(target[0], torch.from_numpy(traj.frame(traj.steps[1])[:, :, 1:]))


def test_roi_none_keeps_the_whole_domain(runs):
    ds = WindowDataset(runs[:1], history=1, roi=None)

    inputs, _, _, _ = ds[0]

    assert inputs.shape == (2, 8, 8, 10)


def test_unknown_roi_raises_at_construction(runs):
    with pytest.raises(ValueError, match="plate"):
        WindowDataset(runs[:1], history=1, roi="plate")


def test_meta_identifies_run_and_time_index(runs):
    ds = WindowDataset(runs, history=2)

    _, _, _, first = ds[0]
    _, _, _, later = ds[len(ds) - 1]

    assert first == {"run": 0, "t": 1}
    assert later == {"run": len(runs) - 1, "t": 4}


def test_dataset_solid_and_frame_helpers_are_what_getitem_stacks(runs):
    from poreml.data import stack_inputs

    ds = WindowDataset(runs[:1], history=2)

    inputs, target, mask, meta = ds[0]  # t = 1
    solid = ds.solid(0)
    frames = [ds.frame(0, 0), ds.frame(0, 1)]

    assert solid.dtype == np.bool_ and solid.shape == (8, 8, 9)  # rock ROI
    assert torch.equal(inputs, torch.from_numpy(stack_inputs(solid, frames)))
    assert torch.equal(target[0], torch.from_numpy(ds.frame(0, 2))[0])
    assert torch.equal(mask[0], torch.from_numpy(~solid))
    assert ds.n_steps(0) == 6


def test_stack_inputs_puts_geometry_first_then_frames_oldest_to_newest():
    from poreml.data import stack_inputs

    solid = np.array([[[True, False]]])
    older = np.full((1, 1, 1, 2), -1.0, dtype=np.float32)
    newer = np.full((1, 1, 1, 2), 1.0, dtype=np.float32)

    x = stack_inputs(solid, [older, newer])

    assert x.shape == (3, 1, 1, 2) and x.dtype == np.float32
    assert x[0].tolist() == [[[1.0, 0.0]]]
    assert (x[1] == -1.0).all() and (x[2] == 1.0).all()


def test_field_spec_defaults_fill_by_name():
    from poreml.data import FieldSpec

    assert FieldSpec(name="phi").fill_value == SOLID_FILL
    assert FieldSpec(name="p").fill_value == 0.0
    assert FieldSpec(name="p", fill=7.0).fill_value == 7.0
    assert FieldSpec(name="u").channels == ("ux", "uy", "uz")
    assert FieldSpec(name="p").channels == ("p",)


def test_multi_field_frames_stack_normalised_channels(runs):
    from poreml.data import FieldSpec

    fields = [FieldSpec(name="phi"), FieldSpec(name="u", scale=0.01), FieldSpec(name="p", offset=0.33, scale=0.001)]
    ds = WindowDataset(runs, history=1, fields=fields)

    assert ds.channels == ("phi", "ux", "uy", "uz", "p")
    assert ds.frame_fill.tolist() == [SOLID_FILL, 0.0, 0.0, 0.0, 0.0]
    frame = ds.frame(0, 2)
    solid = ds.solid(0)
    assert frame.shape == (5,) + solid.shape and frame.dtype == np.float32
    # ROI slabs 0..1 (domain slabs 1..2) are invaded at t=2 -> phi 1, uz 0.005 / 0.01 = 0.5
    pore = ~solid
    assert (frame[0][pore][frame[0][pore] > 0] == 1.0).all()
    assert np.allclose(frame[3][pore][frame[0][pore] > 0], 0.5)
    assert np.allclose(frame[3][pore][frame[0][pore] < 0], 0.0)
    assert np.allclose(frame[4][pore], (0.33 + 0.001 * 2 - 0.33) / 0.001, atol=1e-3)
    # solid voxels carry each field's fill, not NaN
    assert (frame[0][solid] == SOLID_FILL).all() and (frame[1:][:, solid] == 0.0).all()
    assert np.isfinite(frame).all()


def test_frame_raises_when_a_field_has_the_wrong_component_count(runs):
    # A 4-D dataset not named "u" (or a "u" that isn't 3-D) must not silently desync
    # `channels`/`frame_fill` from what was actually read.
    import h5py

    from poreml.data import FieldSpec

    with h5py.File(runs[0].h5_path, "a") as f:  # a 2-component vector field the spec would read as one channel
        for step in f["steps"]:
            f["steps"][step].create_dataset("w2", data=np.zeros((8, 8, 10, 2), dtype=np.float32))
    ds = WindowDataset(runs[:1], history=1, fields=[FieldSpec(name="w2")])

    with pytest.raises(ValueError, match="components"):
        ds.frame(0, ds.index[0][1])


def test_multi_field_getitem_shapes(runs):
    from poreml.data import FieldSpec

    ds = WindowDataset(runs, history=2, fields=[FieldSpec(name="phi"), FieldSpec(name="u")])
    inputs, target, mask, _ = ds[0]

    assert inputs.shape[0] == 1 + 2 * 4 and target.shape[0] == 4 and mask.shape[0] == 1
    assert torch.equal(inputs[-4:], torch.from_numpy(ds.frame(0, ds.index[0][1])))  # last F channels = most recent


def test_stack_inputs_with_multichannel_frames_keeps_recent_frame_last():
    from poreml.data import stack_inputs

    solid = np.array([[[True, False]]])
    older = np.stack([np.full((1, 1, 2), -1.0, dtype=np.float32), np.full((1, 1, 2), 2.0, dtype=np.float32)])
    newer = np.stack([np.full((1, 1, 2), 1.0, dtype=np.float32), np.full((1, 1, 2), 3.0, dtype=np.float32)])

    x = stack_inputs(solid, [older, newer])

    assert x.shape == (5, 1, 1, 2)
    assert x[0].tolist() == [[[1.0, 0.0]]]
    assert (x[1] == -1.0).all() and (x[2] == 2.0).all() and (x[3] == 1.0).all() and (x[4] == 3.0).all()


def test_conditions_become_constant_static_channels(runs):
    from poreml.data import ConditionSpec

    plain = WindowDataset(runs)
    ds = WindowDataset(runs, conditions=(ConditionSpec(name="M", transform="log10"), ConditionSpec(name="theta", scale=180.0)))

    inputs, target, mask, meta = ds[0]
    ref = ds.runs[meta["run"]]
    f = len(ds.channels)
    assert inputs.shape[0] == 1 + 2 + ds.history * f  # solid | conditions | history

    m_channel, theta_channel = inputs[1], inputs[2]
    assert torch.unique(m_channel).numel() == 1  # constant over the volume
    assert m_channel[0, 0, 0].item() == pytest.approx(np.log10(ref.params["M"]))
    assert theta_channel[0, 0, 0].item() == pytest.approx(ref.params["theta"] / 180.0)

    # The convention survives: channel 0 is still solid, the last F channels the most recent frame.
    plain_inputs, _, _, _ = plain[0]
    assert torch.equal(inputs[0], plain_inputs[0])
    assert torch.equal(inputs[-f:], plain_inputs[-f:])


def test_encode_carries_condition_channels(runs):
    from poreml.data import ConditionSpec

    ds = WindowDataset(runs, conditions=(ConditionSpec(name="ca", transform="log10"),))
    encoded = ds.encode(0, [ds.frame(0, 0)])
    assert encoded.shape[1] == 1 + 1 + len(ds.channels)
    assert encoded[0, 1, 0, 0, 0].item() == pytest.approx(np.log10(ds.runs[0].params["ca"]))


def test_missing_condition_fails_at_construction(runs):
    from poreml.data import ConditionSpec

    runs[0].meta["solver"].pop("theta")
    with pytest.raises(ValueError, match="theta"):
        WindowDataset(runs, conditions=(ConditionSpec(name="theta"),))


def reference_frame(ds: WindowDataset, run_idx: int, t: int) -> np.ndarray:
    """The original frame pipeline, kept verbatim as the contract the in-place rewrite must
    match bit for bit: read, normalise, fill, move components first, crop, concatenate."""
    traj = ds.trajectories[run_idx]
    step, span = traj.steps[t], ds.spans[run_idx]
    parts = []
    for spec in ds.fields:
        raw = traj.frame(step, spec.name, fill=float("nan"))
        x = (raw - spec.offset) / spec.scale
        x = np.nan_to_num(x, nan=spec.fill_value, posinf=spec.fill_value, neginf=spec.fill_value)
        x = np.moveaxis(x, -1, 0) if x.ndim == 4 else x[None]
        parts.append(x[..., span])
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


@pytest.mark.parametrize("roi", ["rock", None])
def test_frame_is_bit_identical_to_the_reference_pipeline(runs, roi):
    from poreml.data import FieldSpec

    fields = [
        FieldSpec(name="phi"),
        FieldSpec(name="u", scale=0.01),
        FieldSpec(name="p", offset=0.33, scale=0.001),
        FieldSpec(name="rho", fill=2.0),
    ]
    ds = WindowDataset(runs, history=1, fields=fields, roi=roi)
    for run_idx in range(len(runs)):
        for t in range(ds.n_steps(run_idx)):
            got, want = ds.frame(run_idx, t), reference_frame(ds, run_idx, t)
            assert got.dtype == want.dtype and got.shape == want.shape and got.flags.c_contiguous
            np.testing.assert_array_equal(got, want)
            assert not np.isnan(got).any()
