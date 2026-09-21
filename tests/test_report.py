import json

from poreml.report import write_report


def test_write_report_tables_the_model(tmp_path):
    out = tmp_path / "output"
    single = {
        "n_runs": 1,
        "n_samples": 3,
        "metrics": {"mae": 0.1, "hist/pred": [0.5, 0.5]},
        "provenance": {"poreml_commit": "abc", "device_name": "cpu"},
    }
    roll = {
        "horizon": 4,
        "start_fraction": 0.25,
        "at_horizon": {"mae": 0.3},
    }
    (out / "m1").mkdir(parents=True)
    (out / "m1" / "results_test.json").write_text(json.dumps(single))
    (out / "m1" / "rollout_test.json").write_text(json.dumps(roll))
    text = write_report(out)
    assert (out / "report.md").read_text() == text
    assert "persistence" not in text
    assert "| metric | m1 |" in text
    assert "| mae | 0.1000 |" in text
    assert "h = 4" in text and "| mae | 0.3000 |" in text
    assert "hist/pred" not in text  # lists are not tabulated


def test_write_report_takes_a_title(tmp_path):
    out = tmp_path / "output"
    (out / "m1").mkdir(parents=True)
    (out / "m1" / "results_test.json").write_text(
        json.dumps(
            {
                "n_runs": 1,
                "n_samples": 3,
                "metrics": {"mae": 0.1},
                "provenance": {"poreml_commit": "abc", "device_name": "cpu"},
            }
        )
    )

    assert write_report(out).startswith("# Report\n")  # the default is unchanged
    assert write_report(out, title="shift: GDL").startswith("# shift: GDL\n")


def test_rollout_curves_plots_the_model_alone(tmp_path):
    import pytest

    pytest.importorskip("matplotlib")
    from poreml.report import rollout_curves

    out = tmp_path / "output"
    (out / "m1").mkdir(parents=True)
    (out / "m1" / "rollout_test.json").write_text(
        json.dumps(
            {
                "horizon": 3,
                "summary": {"mae": [0.1, 0.2, 0.3]},
            }
        )
    )
    written = rollout_curves(out)
    text = (out / "rollout_curves.svg").read_text()
    assert written == [out / "rollout_curves.svg"]
    assert "m1" in text and "persistence" not in text
