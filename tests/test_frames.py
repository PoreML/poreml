import h5py
import numpy as np
import pytest

from poreml.frames import H5FrameStore, MemoryFrameStore


@pytest.fixture(params=["memory", "h5"])
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryFrameStore(fields=("phi", "p"))
    return H5FrameStore(tmp_path / "frames", fields=("phi", "p"))


def frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((2, 3, 4, 5), dtype=np.float32)
    f[:, 0] = -1.0  # a solid plane, as decode leaves it
    return f


def test_round_trip_keeps_values_steps_and_meta_per_window(store):
    store.open("run_a", 0, horizon=4, steps=[1, 4], checkpoint_sha256="abc")
    store.write("run_a", 0, 4, frame(4))
    store.write("run_a", 0, 1, frame(1))
    store.finish("run_a", 0)
    store.open("run_a", 64, horizon=4, steps=[1, 4], checkpoint_sha256="abc")
    store.write("run_a", 64, 1, frame(11))
    store.write("run_a", 64, 4, frame(14))
    store.finish("run_a", 64)

    assert store.runs() == ["run_a"] and store.windows("run_a") == [0, 64]
    assert store.has("run_a", 64) and not store.has("run_a", 128) and not store.has("run_b", 0)
    assert store.steps("run_a", 0) == [1, 4]
    assert np.array_equal(store.read("run_a", 0, 1), frame(1))
    assert np.array_equal(store.read("run_a", 64, 4), frame(14))
    meta = store.meta("run_a", 64)
    assert meta["horizon"] == 4 and list(meta["steps"]) == [1, 4] and meta["checkpoint_sha256"] == "abc"
    assert list(meta["fields"]) == ["phi", "p"]


def test_meta_keeps_none(store):
    store.open("run_a", 0, horizon=1, steps=[1], checkpoint_sha256=None)
    store.write("run_a", 0, 1, frame(1))
    store.finish("run_a", 0)
    assert store.meta("run_a", 0)["checkpoint_sha256"] is None


def test_write_before_open_and_wrong_channel_count_are_errors(store):
    with pytest.raises(RuntimeError, match="open"):
        store.write("run_a", 0, 1, frame(1))
    store.open("run_a", 0, horizon=1, steps=[1], checkpoint_sha256=None)
    with pytest.raises(ValueError, match="channels"):
        store.write("run_a", 0, 1, frame(1)[:1])


def test_h5_window_is_invisible_until_finished_and_leaves_no_tmp(tmp_path):
    store = H5FrameStore(tmp_path / "frames", fields=("phi", "p"))
    store.open("run_a", 0, horizon=1, steps=[1], checkpoint_sha256=None)
    store.write("run_a", 0, 1, frame(1))
    assert store.windows("run_a") == [] and not store.has("run_a", 0) and store.runs() == []
    store.finish("run_a", 0)
    assert store.windows("run_a") == [0] and not list((tmp_path / "frames").rglob("*.tmp"))
    assert (tmp_path / "frames" / "run_a" / "w000000.h5").is_file()
    with h5py.File(tmp_path / "frames" / "run_a" / "w000000.h5", "r") as f:
        ds = f["steps"]["001"]["phi"]
        assert ds.chunks == (1, 4, 5)  # the frame is (2, 3, 4, 5): F, D, H, W
        assert 32015 in ds.id.get_create_plist().get_filter(0)  # zstd's filter id


def test_h5_reopen_replaces_the_window(tmp_path):
    store = H5FrameStore(tmp_path / "frames", fields=("phi", "p"))
    for steps in ([1, 2, 3], [2]):
        store.open("run_a", 0, horizon=3, steps=steps, checkpoint_sha256=None)
        for h in steps:
            store.write("run_a", 0, h, frame(h))
        store.finish("run_a", 0)
    assert store.steps("run_a", 0) == [2]


def test_h5_store_reads_without_fields_and_needs_them_to_write(tmp_path):
    writer = H5FrameStore(tmp_path / "frames", fields=("phi", "p"))
    writer.open("run_a", 0, horizon=1, steps=[1], checkpoint_sha256=None)
    writer.write("run_a", 0, 1, frame(1))
    writer.finish("run_a", 0)
    reader = H5FrameStore(tmp_path / "frames")
    assert np.array_equal(reader.read("run_a", 0, 1), frame(1)) and reader.meta("run_a", 0)["fields"] == ["phi", "p"]
    with pytest.raises(ValueError, match="fields"):
        reader.open("run_a", 64, horizon=1, steps=[1], checkpoint_sha256=None)
