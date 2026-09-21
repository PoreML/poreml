import math
import os
import signal
from concurrent.futures.process import BrokenProcessPool

import pytest

from poreml.metric import _pool_map, aggregate, group_of


def _row(run, t0, h, **values):
    return {"run": run, "family": "blob", "group": "gen", "t0": t0, "h": h, "t": t0 + h, **values}


def test_aggregate_averages_per_run_first_then_over_runs():
    rows = [
        _row("long", 0, 1, mae=0.1),
        _row("long", 64, 1, mae=0.3),  # two windows: run mean 0.2
        _row("short", 0, 1, mae=0.6),  # one window
        _row("long", 0, 64, mae=1.0),
        _row("long", 64, 64, mae=3.0),
        _row("short", 0, 64, mae=4.0),
    ]
    out = aggregate(rows, ["mae"], [1, 64], {"gen": ["long", "short"]})
    assert out["summary"]["mae"] == [0.4, 3.0]  # (0.2 + 0.6) / 2 and (2.0 + 4.0) / 2, not the window mean 0.333 / 2.667
    assert out["n_runs_per_step"]["mae"] == [2, 2] and out["n_windows_per_step"]["mae"] == [3, 3]
    assert out["groups"]["gen"]["summary"]["mae"] == [0.4, 3.0] and out["groups"]["gen"]["runs"] == ["long", "short"]


def test_aggregate_skips_nan_and_missing_and_counts_what_it_used():
    rows = [_row("a", 0, 1, x=float("nan")), _row("a", 64, 1, x=2.0), _row("b", 0, 1, x=4.0), _row("b", 0, 7, x=8.0)]
    out = aggregate(rows, ["x", "absent"], [1, 7], {"gen": ["a"], "uCT": ["b"]})
    assert (
        out["summary"]["x"] == [3.0, 8.0] and out["n_windows_per_step"]["x"] == [2, 1] and out["n_runs_per_step"]["x"] == [2, 1]
    )
    assert math.isnan(out["summary"]["absent"][0]) and out["n_runs_per_step"]["absent"] == [0, 0]
    assert out["groups"]["gen"]["summary"]["x"] == [2.0, out["groups"]["gen"]["summary"]["x"][1]]
    assert math.isnan(out["groups"]["gen"]["summary"]["x"][1]) and out["groups"]["uCT"]["summary"]["x"] == [4.0, 8.0]


def test_aggregate_handles_list_valued_metrics_elementwise():
    rows = [_row("a", 0, 1, hist=[1.0, 3.0]), _row("a", 64, 1, hist=[3.0, 5.0]), _row("b", 0, 1, hist=[0.0, 0.0])]
    out = aggregate(rows, ["hist"], [1], {"gen": ["a", "b"]})
    assert out["summary"]["hist"] == [[1.0, 2.0]]  # run a: [2, 4]; run b: [0, 0]; over runs: [1, 2]


def test_group_of_maps_families_to_sources():
    assert group_of("blob") == "gen" and group_of("bentheimer") == "uCT" and group_of("gdl_ct_20") == "uCT"
    assert group_of(None) is None and group_of("synthetic") is None


def _double_or_die(job: int) -> int:  # module level: a spawned worker imports it by name
    if job == 2:
        os.kill(os.getpid(), signal.SIGKILL)  # what the kernel's OOM killer does to a worker
    return 2 * job


def test_pool_map_runs_jobs_on_spawned_workers():
    assert sorted(_pool_map(_double_or_die, [0, 1, 3], 2, None, ())) == [0, 2, 6]


def test_a_killed_worker_fails_the_stage_instead_of_hanging():
    with pytest.raises(BrokenProcessPool):
        list(_pool_map(_double_or_die, range(4), 2, None, ()))
