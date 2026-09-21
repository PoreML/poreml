"""Guards against the solver's real output — run only where the dataset is on disk.

The fixture proves the code against a layout we wrote ourselves; these tests prove it
against what the solver actually writes. They read one frame of one run and are skipped, not
failed, when `data/case` (or `$POREML_DATA_ROOT`) holds no drainage run — `poreml download
--campaign drainage --family bentheimer --size 128` is enough.
"""

import csv
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from poreml.data import Trajectory, discover
from poreml.metrics.descriptors import saturation

DATA_ROOT = Path(os.environ.get("POREML_DATA_ROOT", Path(__file__).resolve().parents[1] / "data" / "case"))

pytestmark = pytest.mark.skipif(not (DATA_ROOT / "drainage" / "runs").is_dir(), reason="no drainage run under data/case")


@pytest.fixture(scope="module")
def drainage():
    return discover(DATA_ROOT, "drainage")


@pytest.fixture(scope="module")
def finished_run(drainage):
    for ref in sorted(drainage.values(), key=lambda r: r.run_id):
        if ref.status == "finished" and ref.shape and ref.shape[0] == 128:
            return ref
    pytest.skip("no finished 128^3 drainage run")


def test_discovery_reads_bobs_nested_layout_and_provenance(drainage):
    assert len(drainage) > 0
    ref = next(iter(drainage.values()))
    assert ref.geometry["family"] in {"bentheimer", "blob", "buffberea", "castlegate", "poly", "sphere"}
    assert ref.geometry["id"] and ref.geometry["sha256"]
    assert set(ref.regions) == {"inbuf", "rock", "plate", "outbuf"}
    assert ref.status in {"finished", "running", "diverged", "failed"}
    assert not np.isnan(ref.params["M"]) and not np.isnan(ref.params["theta"])


def test_rock_roi_saturation_matches_bobs_own_metrics_csv(finished_run):
    """`saturation` over the rock ROI reproduces the `saturation` column the solver logs.

    the solver's `metrics.csv` reports rock saturation (and `sat_all` for the whole domain);
    the two differ by ~0.15 on a drained run, so this pins both the descriptor and the
    ROI convention to the solver's ground truth.
    """
    traj = Trajectory(finished_run.h5_path)
    try:
        last = traj.steps[-1]
        lo, hi = finished_run.regions["rock"]
        phi = torch.from_numpy(traj.frame(last)[..., lo:hi])[None, None]
        mask = torch.from_numpy(~traj.rock[..., lo:hi])[None, None]
    finally:
        traj.close()
    ours = saturation(phi, mask, phase="nw").item()

    with (finished_run.h5_path.parent / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    logged = {int(row["step"]): float(row["saturation"]) for row in rows}
    assert int(last) in logged, "the last HDF5 step has no metrics.csv row"

    assert ours == pytest.approx(logged[int(last)], abs=0.02)
    assert ours > 0.1  # a drained run, not an empty frame
