"""Reading the solver's run output.

the solver writes one HDF5 file per run alongside a run_meta.json. Nothing here repackages or
copies data: runs are read lazily in place. A run directory is any directory holding a
run_meta.json and exactly one *.h5; its name is the run ID.
"""

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import hdf5plugin  # noqa: F401  registers the zstd filter (32015): poreml/data holds the solver's runs re-encoded with it
import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import Dataset

SPLIT_SECTIONS = ("train", "val", "test")


@dataclass(frozen=True)
class RunRef:
    """A located run: where its data is and what physical conditions produced it."""

    run_id: str
    campaign: str
    h5_path: Path
    meta_path: Path
    geometry_path: Path | None
    meta: dict[str, Any]

    @property
    def params(self) -> dict[str, float]:
        """Physical run conditions: viscosity ratio M, contact angle theta, capillary number."""
        extra = self.meta.get("extra", {})
        solver = self.meta.get("solver", {})
        # the solver's underfill driver predates the extra.M convention: it records the
        # viscosity ratio as extra.m_ratio.
        m = extra.get("M", extra.get("m_ratio", float("nan")))
        return {
            "M": float(m),
            "theta": float(solver.get("theta", float("nan"))),
            "ca": float(extra.get("ca", float("nan"))),
        }

    @property
    def status(self) -> str:
        """the solver's run status: "finished", "running", "diverged", ...; "unknown" if unrecorded."""
        return str(self.meta.get("run", {}).get("status", "unknown"))

    @property
    def finish_type(self) -> str | None:
        """Why a finished run stopped (drainage: "pv_cap" or "dp_cap"); None while running."""
        value = self.meta.get("extra", {}).get("finish_type")
        return None if value is None else str(value)

    @property
    def geometry(self) -> dict[str, Any]:
        """Provenance of the porous geometry: `{"id", "family", "sha256", "porosity"}`.

        the solver stages a geometry from `<...>/data/<family>/<size>/<stem>.npy` and records
        that path as `geometry.source`; `family` (bentheimer, blob, poly, sphere) and
        `id` (the stem) come from it. Every value is None where the solver recorded nothing —
        the split-by-geometry rule needs to know that it cannot be checked, not guess.
        """
        geo = self.meta.get("geometry", {})
        source = geo.get("source")
        family = geometry_id = None
        if source:
            path = Path(source)
            geometry_id = path.stem
            family = path.parent.parent.name or None
        porosity = geo.get("porosity")
        return {
            "id": geometry_id,
            "family": family,
            "sha256": geo.get("sha256"),
            "porosity": None if porosity is None else float(porosity),
        }

    def summary(self) -> dict[str, Any]:
        """The flat per-run record artifacts embed so results can be stratified offline."""
        return {
            "run_id": self.run_id,
            "campaign": self.campaign,
            "status": self.status,
            "finish_type": self.finish_type,
            "params": self.params,
            "geometry": self.geometry,
            "shape": list(self.shape),
        }

    @property
    def regions(self) -> dict[str, list[int]]:
        """Axial spans of the domain, e.g. {"inbuf": [0, 7], "rock": [7, 135]}.

        the solver's domain is larger than the geometry: it carries inlet buffer and outlet
        plate slabs. Tasks that need to score only the rock ROI crop with this.
        """
        return dict(self.meta.get("extra", {}).get("regions", {}))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.meta.get("geometry", {}).get("shape", ()))


def _read_run(run_dir: Path, campaign: str) -> RunRef | None:
    # Trapping runs are two-stage and carry run_meta_flood.json / run_meta_stab.json
    # instead of a plain run_meta.json; the single h5 is the flood trajectory, so the
    # flood meta is the one whose status/finish_type describe it.
    meta_path = next((p for name in ("run_meta.json", "run_meta_flood.json") if (p := run_dir / name).is_file()), None)
    if meta_path is None:
        return None
    h5_files = sorted(run_dir.glob("*.h5"))
    if len(h5_files) != 1:
        return None
    geometry = sorted(run_dir.glob("*_geometry.json"))
    return RunRef(
        run_id=run_dir.name,
        campaign=campaign,
        h5_path=h5_files[0],
        meta_path=meta_path,
        geometry_path=geometry[0] if geometry else None,
        meta=json.loads(meta_path.read_text()),
    )


def discover(root: Path | str, campaign: str) -> dict[str, RunRef]:
    """Index every run of a campaign under <root>/<campaign>/runs/, keyed by run ID."""
    runs_dir = Path(root) / campaign / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"no runs directory for campaign {campaign!r} at {runs_dir}")

    found: dict[str, RunRef] = {}
    run_dirs = {p.parent for name in ("run_meta.json", "run_meta_flood.json") for p in runs_dir.rglob(name)}
    for run_dir in sorted(run_dirs):
        ref = _read_run(run_dir, campaign)
        if ref is not None:
            found[ref.run_id] = ref
    return found


def load_split(path: Path | str) -> dict[str, list[str]]:
    """Read a frozen split file into {"train": [...], "val": [...], "test": [...]}."""
    path = Path(path)
    payload = yaml.safe_load(path.read_text()) or {}

    unknown = set(payload) - set(SPLIT_SECTIONS)
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)} in split {path}; expected {list(SPLIT_SECTIONS)}")

    split = {section: list(payload.get(section) or []) for section in SPLIT_SECTIONS}

    seen: dict[str, str] = {}
    for section, run_ids in split.items():
        for run_id in run_ids:
            if run_id in seen:
                raise ValueError(f"run {run_id!r} appears in both {seen[run_id]!r} and {section!r} in split {path}")
            seen[run_id] = section
    return split


def resolve(
    split: dict[str, list[str]], available: dict[str, RunRef], *, require_finished: bool = True
) -> dict[str, list[RunRef]]:
    """Join a split's run IDs to discovered runs.

    A run ID with no matching directory raises. Nothing is silently dropped: a benchmark
    that quietly skips samples reports a number nobody can reproduce.

    A run that is not `finished` is refused too unless `require_finished` is False: a
    running run gains frames between two invocations, so any number scored against it
    is not reproducible either.
    """
    resolved: dict[str, list[RunRef]] = {}
    unfinished: list[str] = []
    for section, run_ids in split.items():
        refs = []
        for run_id in run_ids:
            if run_id not in available:
                known = ", ".join(sorted(available)[:5]) or "<none>"
                raise KeyError(f"run {run_id!r} from split section {section!r} was not found; discovered e.g.: {known}")
            ref = available[run_id]
            if require_finished and ref.status != "finished":
                unfinished.append(f"{run_id} ({section}: {ref.status})")
            refs.append(ref)
        resolved[section] = refs
    if unfinished:
        raise ValueError(
            "split names run(s) that are not finished, so their frame count can still change: "
            + ", ".join(unfinished)
            + "; set data.require_finished: false to score them anyway"
        )
    return resolved


def draw_split(refs: Sequence[RunRef], n_train: int, n_val: int, n_test: int, *, seed: int) -> dict[str, list[str]]:
    """Draw a family-stratified split of finished runs, deterministic under `seed`.

    Each geometry family contributes to every section in proportion to its share of the
    finished pool (largest remainder, so the section sizes are exact), and a run appears
    once. Only finished runs are eligible. Raises rather than under-filling: a split file
    that quietly holds fewer runs than asked for is the kind of thing nobody notices
    until the numbers are published.
    """
    finished = [r for r in refs if r.status == "finished"]
    wanted = n_train + n_val + n_test
    if wanted > len(finished):
        raise ValueError(f"asked for {wanted} runs but only {len(finished)} finished runs are available")
    by_family: dict[str, list[str]] = {}
    for r in finished:
        family = r.geometry["family"]
        if family is None:
            raise ValueError(f"run {r.run_id!r} has no geometry family; cannot stratify")
        by_family.setdefault(family, []).append(r.run_id)

    rng = np.random.default_rng(seed)
    families = sorted(by_family)
    pools = {f: [by_family[f][i] for i in rng.permutation(len(by_family[f]))] for f in families}
    total = len(finished)
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for section, n in (("test", n_test), ("val", n_val), ("train", n_train)):
        # Largest-remainder apportionment of n across families by pool size.
        quotas = {f: n * len(by_family[f]) / total for f in families}
        counts = {f: int(quotas[f]) for f in families}
        for f in sorted(families, key=lambda f: quotas[f] - counts[f], reverse=True)[: n - sum(counts.values())]:
            counts[f] += 1
        for f in families:
            if counts[f] > len(pools[f]):
                raise ValueError(f"family {f!r} has too few finished runs left for section {section!r}")
            split[section].extend(pools[f][: counts[f]])
            del pools[f][: counts[f]]
    return {section: sorted(ids) for section, ids in split.items()}


# the solver writes NaN at solid voxels. Filling with the wetting-phase value keeps solid out
# of the way of both losses and metrics; nothing downstream ever sees a NaN.
SOLID_FILL = -1.0


class FieldSpec(BaseModel):
    """One the solver field as the model sees it: `(raw - offset) / scale`, solid voxels set to `fill`.

    Constants are fixed in the config rather than estimated from the training split, so a
    config fully determines what a checkpoint was fed. `phi` fills solid with
    `SOLID_FILL`, every other field with 0 — in normalised units, applied after scaling.
    `u` is the solver's `(D, H, W, 3)` velocity and expands to `ux, uy, uz`; anything else is one
    channel named after its dataset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    offset: float = 0.0
    scale: float = Field(default=1.0, gt=0)
    fill: float | None = None

    @property
    def fill_value(self) -> float:
        if self.fill is not None:
            return self.fill
        return SOLID_FILL if self.name == "phi" else 0.0

    @property
    def channels(self) -> tuple[str, ...]:
        return ("ux", "uy", "uz") if self.name == "u" else (self.name,)


DEFAULT_FIELDS: tuple[FieldSpec, ...] = (FieldSpec(name="phi"),)


class ConditionSpec(BaseModel):
    """One run condition as the model sees it: `(transform(raw) - offset) / scale`.

    `name` is a key of `RunRef.params` (the solver's run conditions). Like `FieldSpec`, the
    constants live in the config so a config fully determines what a checkpoint was fed.
    `M` and `ca` span decades — give them `transform: log10`; `theta` is degrees, so
    `scale: 180` puts it at O(1).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["M", "theta", "ca"]
    transform: Literal["log10"] | None = None
    offset: float = 0.0
    scale: float = Field(default=1.0, gt=0)

    def value(self, ref: "RunRef") -> float:
        raw = ref.params[self.name]
        if self.transform == "log10":
            raw = math.log10(raw) if raw > 0 else float("nan")
        return (raw - self.offset) / self.scale


class Trajectory:
    """Lazy reader for one the solver run's HDF5 file.

    The file handle opens on first access and stays open until `close()`. Every process
    reads through a handle it opened itself: the handle is dropped on pickling (the
    `spawn`/`forkserver` path) and reopened when the owning PID no longer matches the
    current one (the `fork` path, where the dataset is never pickled). HDF5 is not
    fork-safe — concurrent reads through a handle inherited across `fork()` can return
    wrong data with no error — so an inherited handle is never read from.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._file: h5py.File | None = None
        self._pid: int | None = None
        self._steps: tuple[str, ...] | None = None
        self._rock: np.ndarray | None = None

    @property
    def file(self) -> h5py.File:
        if self._file is None or self._pid != os.getpid():
            # An inherited handle is deliberately dropped rather than closed: it belongs
            # to the process that opened it, and calling into HDF5 on it after a fork is
            # exactly what is unsafe.
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    @property
    def steps(self) -> tuple[str, ...]:
        if self._steps is None:
            self._steps = tuple(sorted(self.file["steps"].keys()))
        return self._steps

    @property
    def rock(self) -> np.ndarray:
        """Static solid mask, True = solid."""
        if self._rock is None:
            self._rock = np.asarray(self.file["rock"][:]) != 0
        return self._rock

    def frame(self, step: str, field: str = "phi", fill: float = SOLID_FILL) -> np.ndarray:
        """Read one field at one step as float32 with NaN replaced by `fill`."""
        raw = np.asarray(self.file["steps"][step][field][:], dtype=np.float32)
        return np.nan_to_num(raw, nan=fill, posinf=fill, neginf=fill)

    def close(self) -> None:
        """Release the handle. Cached steps and rock survive; reads reopen on demand."""
        if self._file is not None and self._pid == os.getpid():
            self._file.close()
        self._file = None
        self._pid = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_file"] = None  # h5py handles are not picklable
        state["_pid"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"Trajectory({self.path.name!r}, n_steps={len(self.steps)})"


def stack_inputs(solid: np.ndarray, frames: Sequence[np.ndarray], cond: np.ndarray | None = None) -> np.ndarray:
    """The input channel convention, in one place: static channels first — geometry, then
    one constant channel per run condition — then the history frames oldest to newest,
    each contributing its F channels — so the most recent frame is the last F channels
    (see `Task`)."""
    static = [solid.astype(np.float32)[None]]
    if cond is not None and len(cond):
        static.append(np.broadcast_to(np.asarray(cond, dtype=np.float32)[:, None, None, None], (len(cond), *solid.shape)))
    return np.concatenate([*static, *frames], axis=0)


Sample = tuple[Tensor, Tensor, Tensor, dict[str, int]]
"""What a dataset yields: `(inputs, target, mask, meta)`.

`meta` is `{"run": index into `dataset.runs`, "t": time index of the last input frame}`.
The default collate turns each into a `(B,)` tensor, so the evaluator can attribute
every scored sample to its trajectory — the unit that splits, statistics and
stratification are all defined on.
"""


class WindowDataset(Dataset):
    """Windows of (solid mask + `history` past multi-field frames) -> next frame.

    Item `i` maps to a run and a time index `t`; inputs cover the `history` frames
    ending at `t` (oldest first) and the target is the frame at `t + 1`. Each frame
    stacks `fields` (normalised, filled) into `F` channels, in the order given by
    `channels`. `future > 1` (push-forward training) makes the target the `future` frames
    `t + 1 .. t + future` stacked on a leading axis, `(future, F, D, H, W)`; the default
    keeps the single frame `(F, D, H, W)` every consumer of a window expects.

    `roi` names a region of `RunRef.regions` to crop to along the last (flow) axis;
    `None` keeps the solver's whole domain. The default is the rock: the solver's domain also holds an
    inlet reservoir that is always fully invaded, a porous plate that is a boundary
    device, and an outlet buffer. Scoring those hands persistence a perfect score on a
    third of the voxels and makes `saturation` disagree with the solver's own rock saturation.
    Cropping (rather than masking) also keeps `trapped_volume`'s inlet/outlet slices
    meaningful: the rock's first slice touches the reservoir, its last the plate.
    """

    def __init__(
        self,
        runs: Sequence[RunRef],
        history: int = 1,
        fields: Sequence[FieldSpec] = DEFAULT_FIELDS,
        stride: int = 1,
        roi: str | None = "rock",
        conditions: Sequence[ConditionSpec] = (),
        future: int = 1,
    ) -> None:
        if history < 1:
            raise ValueError(f"history must be at least 1, got {history}")
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        if future < 1:
            raise ValueError(f"future must be at least 1, got {future}")

        self.history = history
        self.future = future
        self.roi = roi
        self.runs = list(runs)
        self.trajectories = [Trajectory(ref.h5_path) for ref in self.runs]
        self.spans: list[slice] = [self._span(ref) for ref in self.runs]

        self.conditions = tuple(conditions)
        self.cond_values = np.array([[spec.value(ref) for spec in self.conditions] for ref in self.runs], dtype=np.float32)
        for ref, values in zip(self.runs, self.cond_values, strict=True):
            for spec, value in zip(self.conditions, values, strict=True):
                if not np.isfinite(value):
                    raise ValueError(f"run {ref.run_id!r} has no usable condition {spec.name!r} (value {value})")

        self.fields = tuple(fields)
        if not self.fields:
            raise ValueError("at least one field is required")
        self.channels: tuple[str, ...] = tuple(c for f in self.fields for c in f.channels)
        self.frame_fill = np.array([f.fill_value for f in self.fields for _ in f.channels], dtype=np.float32)

        self.index: list[tuple[int, int]] = []
        try:
            for run_idx, traj in enumerate(self.trajectories):
                n_steps = len(traj.steps)
                first_t = history - 1
                last_t = n_steps - 1 - future  # need a frame at t + future
                if last_t < first_t:
                    raise ValueError(
                        f"run {self.runs[run_idx].run_id!r} has {n_steps} steps, too few for history={history}, future={future}"
                    )
                self.index.extend((run_idx, t) for t in range(first_t, last_t + 1, stride))
        finally:
            # Counting steps opened every file. Close them all: a dataset constructed in
            # the parent must never hand an open HDF5 handle to a forked DataLoader
            # worker. The step lists are cached, so reopening on first read costs nothing.
            for traj in self.trajectories:
                traj.close()

    collate = None  # the default collate stacks tensors; point datasets override this

    def encode(self, run_idx: int, frames: Sequence[np.ndarray]) -> Tensor:
        """The model input for one run given `frames` (oldest first): a batch of one, `(1, Cin, D, H, W)`.

        `encode`/`decode` are the stream interface rollout composes on: `encode` builds a
        batch of one from grid frames, `decode` returns a grid frame. The point stream
        (`points.PointWindowDataset`) implements the same two.
        """
        return torch.from_numpy(stack_inputs(self.solid(run_idx), frames, self.cond_values[run_idx]))[None]

    def decode(self, inputs: Tensor, pred: Tensor) -> Tensor:
        """A prediction as a grid frame with solid voxels forced to each field's fill.

        The solid mask is channel 0 of `inputs` (see `stack_inputs`); the model is never
        trusted on voxels that are not fluid.
        """
        fluid = inputs[:, :1] == 0
        fill = torch.as_tensor(self.frame_fill, device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1, 1)
        return torch.where(fluid, pred, fill)

    def _span(self, ref: RunRef) -> slice:
        if self.roi is None:
            return slice(None)
        regions = ref.regions
        if self.roi not in regions:
            raise ValueError(
                f"run {ref.run_id!r} has no region {self.roi!r}; its regions are {sorted(regions)} "
                "(roi=None keeps the whole domain)"
            )
        lo, hi = regions[self.roi]
        return slice(int(lo), int(hi))

    def n_steps(self, run_idx: int) -> int:
        return len(self.trajectories[run_idx].steps)

    def solid(self, run_idx: int) -> np.ndarray:
        """Bool solid mask of one run, cropped to the ROI."""
        return np.ascontiguousarray(self.trajectories[run_idx].rock[..., self.spans[run_idx]])

    def frame(self, run_idx: int, t: int) -> np.ndarray:
        """All fields of one run at time index `t`, normalised, filled and cropped to the ROI: `(F, D, H, W)`.

        Built in place: each field is read cropped straight into its rows of one preallocated
        output (a scalar via `read_direct`, the velocity as one decode plus one strided copy —
        per-component hyperslabs would decode every chunk three times), then normalised and
        filled with in-place passes over the cropped array. Bit-identical to the read →
        normalise → fill → move-axis → crop → concatenate pipeline it replaced
        (`tests/test_data_windows.py::test_frame_is_bit_identical_to_the_reference_pipeline`)
        at about a third of the numpy time: a training window is two of these, and with the
        zstd decode at ~40 ms the loader workers, not the GPU, were setting the pace.
        """
        traj = self.trajectories[run_idx]
        step, span = traj.steps[t], self.spans[run_idx]
        group = traj.file["steps"][step]
        out: np.ndarray | None = None
        row = 0
        for spec in self.fields:
            dset = group[spec.name]
            n = len(spec.channels)
            if out is None:
                d, h, w = dset.shape[:3]
                out = np.empty((len(self.channels), d, h, len(range(*span.indices(w)))), dtype=np.float32)
            components = dset.shape[3] if dset.ndim == 4 else 1
            if components != n or dset.ndim not in (3, 4):
                raise ValueError(f"field {spec.name!r} has {components} components; expected {n} ({spec.channels})")
            block = out[row : row + n]
            if dset.ndim == 4:
                block[...] = np.moveaxis(dset[:, :, span, :], -1, 0)  # (D,H,W,3) -> (3,D,H,W), one decode
            else:
                dset.read_direct(block[0], source_sel=np.s_[:, :, span])
            if spec.offset != 0.0:
                np.subtract(block, spec.offset, out=block)
            if spec.scale != 1.0:
                np.divide(block, spec.scale, out=block)
            np.nan_to_num(block, copy=False, nan=spec.fill_value, posinf=spec.fill_value, neginf=spec.fill_value)
            row += n
        assert out is not None  # at least one field: enforced in __init__
        return out

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Sample:
        run_idx, t = self.index[i]
        solid = self.solid(run_idx)
        past = [self.frame(run_idx, t - k) for k in reversed(range(self.history))]
        if self.future == 1:
            target = self.frame(run_idx, t + 1)
        else:
            target = np.stack([self.frame(run_idx, t + 1 + k) for k in range(self.future)])
        return (
            torch.from_numpy(stack_inputs(solid, past, self.cond_values[run_idx])),
            torch.from_numpy(target),
            torch.from_numpy(~solid)[None],
            {"run": run_idx, "t": t},
        )

    def __repr__(self) -> str:
        return (
            f"WindowDataset(n_runs={len(self.runs)}, history={self.history}, roi={self.roi!r}, "
            f"channels={self.channels}, n_windows={len(self)})"
        )
