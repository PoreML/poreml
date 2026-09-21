"""Predicted rollout frames on disk: the interface between the GPU inference stage and the CPU metric stage.

A store holds, per (run, window start `t0`), the decoded prediction `(F, D, H, W)` at each
stored step (solid voxels at each field's fill, exactly what the metrics see) plus the window's
`horizon`, `steps` and the checkpoint's sha256, so the metric stage can refuse frames that
belong to another checkpoint. `H5FrameStore` writes one zstd-3 HDF5 per window —
`<dir>/<run_id>/w<t0:06d>.h5`, datasets `steps/<h:03d>/<field>` chunked `(1, H, W)` like
the solver's files — to a `.tmp` sibling that `finish` renames into place, so a killed inference never
leaves a torn window and a rerun resumes at the first missing one. `MemoryFrameStore` backs
the tests.

Several workers may share one store (`rollout.infer_windows(shared=True)`): before computing a
window a worker `claim`s it — an exclusively created `<window>.claim` beside the `.tmp`, touched
at every keyframe write and removed by `finish` — and skips a window another worker holds. A
claim nobody has touched for `stale_after` seconds belongs to a dead worker and is taken over,
so a crashed worker's window is redone by whoever comes next rather than lost.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import h5py
import hdf5plugin
import numpy as np

ZSTD_LEVEL = 3
STALE_CLAIM_SECONDS = 1800.0  # a window's keyframe writes come minutes apart at most; 30 min without one is a dead worker
_WINDOW = re.compile(r"^w(\d{6})\.h5$")


class FrameStore(Protocol):
    def open(self, run_id: str, t0: int, **meta: Any) -> None: ...
    def write(self, run_id: str, t0: int, h: int, frame: np.ndarray) -> None: ...
    def finish(self, run_id: str, t0: int) -> None: ...
    def has(self, run_id: str, t0: int) -> bool: ...
    def claim(self, run_id: str, t0: int) -> bool: ...
    def read(self, run_id: str, t0: int, h: int) -> np.ndarray: ...
    def steps(self, run_id: str, t0: int) -> list[int]: ...
    def meta(self, run_id: str, t0: int) -> dict[str, Any]: ...
    def windows(self, run_id: str) -> list[int]: ...
    def runs(self) -> list[str]: ...


def _check(frame: np.ndarray, fields: Sequence[str] | None) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 4:
        raise ValueError(f"a frame is (F, D, H, W); got shape {frame.shape}")
    if fields is not None and frame.shape[0] != len(fields):
        raise ValueError(f"the store holds {len(fields)} fields {tuple(fields)}; got a frame with {frame.shape[0]} channels")
    return frame


class MemoryFrameStore:
    """Frames in a dict, keyed by (run_id, t0): the tests."""

    def __init__(self, fields: Sequence[str] | None = None) -> None:
        self.fields = tuple(fields) if fields is not None else None
        self._frames: dict[tuple[str, int], dict[int, np.ndarray]] = {}
        self._meta: dict[tuple[str, int], dict[str, Any]] = {}
        self._done: set[tuple[str, int]] = set()
        self._claimed: set[tuple[str, int]] = set()

    def open(self, run_id: str, t0: int, **meta: Any) -> None:
        key = (run_id, int(t0))
        self._meta[key] = {
            "t0": int(t0),
            **meta,
            "fields": list(self.fields or ()),
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._frames[key] = {}
        self._done.discard(key)

    def write(self, run_id: str, t0: int, h: int, frame: np.ndarray) -> None:
        key = (run_id, int(t0))
        if key not in self._frames or key in self._done:
            raise RuntimeError(f"open({run_id!r}, {t0}) must precede write")
        self._frames[key][int(h)] = _check(frame, self.fields).copy()

    def finish(self, run_id: str, t0: int) -> None:
        self._done.add((run_id, int(t0)))
        self._claimed.discard((run_id, int(t0)))

    def has(self, run_id: str, t0: int) -> bool:
        return (run_id, int(t0)) in self._done

    def claim(self, run_id: str, t0: int) -> bool:
        key = (run_id, int(t0))
        if key in self._claimed:
            return False
        self._claimed.add(key)
        return True

    def read(self, run_id: str, t0: int, h: int) -> np.ndarray:
        return self._frames[(run_id, int(t0))][int(h)]

    def steps(self, run_id: str, t0: int) -> list[int]:
        return sorted(self._frames.get((run_id, int(t0)), {}))

    def meta(self, run_id: str, t0: int) -> dict[str, Any]:
        return dict(self._meta[(run_id, int(t0))])

    def windows(self, run_id: str) -> list[int]:
        return sorted(t0 for r, t0 in self._done if r == run_id)

    def runs(self) -> list[str]:
        return sorted({r for r, _ in self._done})


class H5FrameStore:
    """One zstd HDF5 per window under `directory/<run_id>/`; `fields` names the channels when writing."""

    def __init__(
        self, directory: Path | str, fields: Sequence[str] | None = None, stale_after: float = STALE_CLAIM_SECONDS
    ) -> None:
        self.directory = Path(directory)
        self.fields = tuple(fields) if fields is not None else None
        self.stale_after = float(stale_after)
        self._open: dict[tuple[str, int], h5py.File] = {}  # (run_id, t0) -> the .tmp file being written

    def path(self, run_id: str, t0: int) -> Path:
        return self.directory / run_id / f"w{int(t0):06d}.h5"

    def _tmp(self, run_id: str, t0: int) -> Path:
        return self.path(run_id, t0).with_suffix(".h5.tmp")

    def _claim(self, run_id: str, t0: int) -> Path:
        return self.path(run_id, t0).with_suffix(".h5.claim")

    def claim(self, run_id: str, t0: int) -> bool:
        """Take the window for this worker: True if nobody holds it (or its holder went stale), else False.

        The marker is created exclusively (`O_EXCL`), so two workers racing for one window get one
        winner; it records host, pid and time for the operator. A marker untouched for `stale_after`
        seconds is a dead worker's and is taken over. `write` touches the marker, `finish` removes it.
        """
        marker = self._claim(run_id, t0)
        marker.parent.mkdir(parents=True, exist_ok=True)
        stamp = f"{socket.gethostname()} pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', '-')} at={time.time():.0f}\n"
        for _ in range(2):
            try:
                fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                try:
                    age = time.time() - marker.stat().st_mtime
                except FileNotFoundError:  # released between our two looks: try the exclusive create again
                    continue
                if age <= self.stale_after:
                    return False
                marker.write_text(stamp)  # a dead worker's: take it over (the write refreshes the mtime)
                return True
            with os.fdopen(fd, "w") as f:
                f.write(stamp)
            return True
        return False

    def _touch_claim(self, run_id: str, t0: int) -> None:
        try:
            os.utime(self._claim(run_id, t0))
        except FileNotFoundError:
            pass

    def open(self, run_id: str, t0: int, **meta: Any) -> None:
        if self.fields is None:
            raise ValueError("H5FrameStore needs `fields` to write")
        key = (run_id, int(t0))
        if key in self._open:
            self._open.pop(key).close()
        tmp = self._tmp(run_id, t0)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.unlink(missing_ok=True)
        f = h5py.File(tmp, "w")
        f.attrs["fields"] = np.array(list(self.fields), dtype=h5py.string_dtype())
        f.attrs["written_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        f.attrs["meta"] = json.dumps({"t0": int(t0), **meta})  # JSON keeps None and lists intact
        f.create_group("steps")
        self._open[key] = f

    def write(self, run_id: str, t0: int, h: int, frame: np.ndarray) -> None:
        f = self._open.get((run_id, int(t0)))
        if f is None:
            raise RuntimeError(f"open({run_id!r}, {t0}) must precede write")
        frame = _check(frame, self.fields)
        g = f["steps"].require_group(f"{int(h):03d}")
        for c, name in enumerate(self.fields or ()):
            if name in g:
                del g[name]
            g.create_dataset(name, data=frame[c], chunks=(1,) + frame.shape[2:], **hdf5plugin.Zstd(clevel=ZSTD_LEVEL))
        f.flush()
        self._touch_claim(run_id, t0)

    def finish(self, run_id: str, t0: int) -> None:
        self._open.pop((run_id, int(t0))).close()
        self._tmp(run_id, t0).replace(self.path(run_id, t0))
        self._claim(run_id, t0).unlink(missing_ok=True)

    def has(self, run_id: str, t0: int) -> bool:
        return self.path(run_id, t0).is_file()

    def read(self, run_id: str, t0: int, h: int) -> np.ndarray:
        with h5py.File(self.path(run_id, t0), "r") as f:
            g = f["steps"][f"{int(h):03d}"]
            fields = [str(v) for v in f.attrs["fields"]]
            return np.stack([g[name][()] for name in fields]).astype(np.float32, copy=False)

    def steps(self, run_id: str, t0: int) -> list[int]:
        with h5py.File(self.path(run_id, t0), "r") as f:
            return sorted(int(k) for k in f["steps"])

    def meta(self, run_id: str, t0: int) -> dict[str, Any]:
        with h5py.File(self.path(run_id, t0), "r") as f:
            out = json.loads(f.attrs["meta"])
            out["fields"] = [str(v) for v in f.attrs["fields"]]
            out["written_at"] = str(f.attrs["written_at"])
            return out

    def windows(self, run_id: str) -> list[int]:
        folder = self.directory / run_id
        if not folder.is_dir():
            return []
        return sorted(int(m.group(1)) for p in folder.iterdir() if (m := _WINDOW.match(p.name)))

    def runs(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(p.name for p in self.directory.iterdir() if p.is_dir() and self.windows(p.name))
