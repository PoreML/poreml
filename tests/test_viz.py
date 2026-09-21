import json

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista")

from poreml import viz  # noqa: E402


def _volume(n: int = 16):
    rng = np.random.default_rng(0)
    solid = rng.random((n, n, n)) < 0.3
    red = (np.arange(n)[None, None, :] < n // 2) & ~solid
    return solid, red


def test_render_jobs_writes_labelled_frames_that_are_not_black(tmp_path):
    solid, red = _volume()
    jobs = [
        viz.RenderJob(path=tmp_path / "single.png", panels=[viz.Panel(red=red, label="truth")]),
        viz.RenderJob(
            path=tmp_path / "pair.png", panels=[viz.Panel(red=red, label="truth"), viz.Panel(red=~red & ~solid, label="pred")]
        ),
    ]

    viz.render_jobs(jobs, solid, window=(160, 120))

    from PIL import Image

    single = np.asarray(Image.open(tmp_path / "single.png"))
    pair = np.asarray(Image.open(tmp_path / "pair.png"))
    assert single.shape == (120, 160, 3) and single.any() and single.min() < 255
    assert pair.shape == (120, 320, 3)


def test_write_gif_animates_the_frames(tmp_path):
    from PIL import Image

    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.png"
        Image.fromarray(np.full((8, 8, 3), i * 100, dtype=np.uint8)).save(p)
        paths.append(p)

    viz.write_gif(paths, tmp_path / "out.gif", fps=5)

    gif = Image.open(tmp_path / "out.gif")
    assert gif.n_frames == 3


def test_plot_rollout_curves_writes_an_svg_per_metric_panel(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"model": "persistence", "summary": {"mae": [0.1, 0.2, 0.3], "iou": [0.9, 0.8, 0.7]}}))
    b.write_text(json.dumps({"model": "unet3d", "summary": {"mae": [0.05, 0.1, 0.15], "iou": [0.95, 0.9, 0.85]}}))

    out = viz.plot_rollout_curves({"persistence": a, "unet3d": b}, ["mae", "iou"], tmp_path / "curves.svg")

    text = out.read_text()
    assert out.suffix == ".svg" and "persistence" in text and "unet3d" in text


def test_plot_metric_curve_writes_one_svg_with_every_label(tmp_path):
    out = viz.plot_metric_curve(
        {"unet": ([1, 7, 64], [0.1, 0.2, None]), "fno": ([1, 7, 64], [0.05, 0.1, 0.2])}, "mae@phi", tmp_path / "c.svg"
    )
    text = out.read_text()
    assert "unet" in text and "fno" in text and "mae@phi" in text


def test_plot_training_curves_reads_the_csv_and_writes_an_svg(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "epoch,step,train_loss,epoch_seconds,peak_memory_mb,lr,mae,iou,hist/pred[0]\n"
        "0,10,0.5,1.0,,0.001,0.4,0.6,0.5\n1,20,0.3,1.0,,0.001,0.2,0.8,0.5\n"
    )

    out = viz.plot_training_curves(csv_path, tmp_path / "train.svg")

    text = out.read_text()
    assert "train loss" in text and "mae" in text and "iou" in text and "hist/pred" not in text


def test_plot_eval_curves_draws_model_and_rollout(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "epoch,step,mae,rollout/mae,iou,curvature_hist/pred[0],curvature_hist/pred[1]\n"
        "1,10,0.3,0.6,0.9,0.1,0.9\n"
        "3,30,0.2,0.5,0.95,0.2,0.8\n"
    )

    out = viz.plot_eval_curves(csv_path, tmp_path / "curves.svg")

    text = out.read_text()
    assert out.exists() and "<svg" in text
    assert "mae" in text and "iou" in text and "curvature_hist" not in text


def test_plot_eval_curves_keeps_a_column_blank_at_the_first_row(tmp_path):
    # A metric that is NaN (blank) at epoch 1 — e.g. w1_norm with a zero-variance target —
    # must still get a panel: only row 0 being non-numeric must not drop the whole column.
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("epoch,step,mae\n1,10,\n3,30,0.2\n")

    out = viz.plot_eval_curves(csv_path, tmp_path / "curves.svg")

    assert "mae" in out.read_text()


def test_plot_eval_curves_raises_when_no_scalar_columns_remain(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("epoch,step\n1,10\n3,30\n")

    with pytest.raises(ValueError, match="no scalar metric columns"):
        viz.plot_eval_curves(csv_path, tmp_path / "curves.svg")
