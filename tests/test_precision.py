"""Inference precision follows the model's declared precision (`precision_of`): a bf16 model's
eval, rollout and per-epoch selection pass run its forward under bf16 autocast and hand fp32
outputs to the metrics; a tf32 model (the project default since 2026-09-10) has the process's
matmul precision switched and is otherwise untouched; fp32 is torch's default. Measured 2026-09-08
on an H200 with the gdl AB-UPT checkpoint over 592k pore voxels: 7.95 -> 1.08 s per forward,
one-step mae@phi equal to three digits; 2026-09-10 with the drainage Transolver checkpoint over
494k: fp32 0.313 -> tf32 0.058 s per forward, |phi - fp32| mean 2e-5."""

import json

import pytest
import torch
from torch import nn

from poreml.config import ModelConfig, TaskConfig
from poreml.evaluate import evaluate, rollout_split
from poreml.precision import autocast, configure, inference_model
from poreml.train import train
from tests.test_train_eval import cfg  # noqa: F401, F811  (the fixture-dataset config)


class _Probe(nn.Module):
    """Records the autocast state its forward ran under and returns a bf16-able output."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.seen: list[tuple[bool, torch.dtype]] = []

    def forward(self, x):
        y = self.linear(x)
        self.seen.append((torch.is_autocast_enabled("cpu"), y.dtype))
        return y


def test_fp32_returns_the_model_itself():
    net = _Probe()
    assert inference_model(net, "fp32", "cpu") is net
    with autocast("fp32", "cpu"):
        assert not torch.is_autocast_enabled("cpu")


def test_bf16_wraps_the_forward_in_autocast_and_returns_fp32():
    net = _Probe()
    wrapped = inference_model(net, "bf16", "cpu")
    assert wrapped is not net
    out = wrapped(torch.zeros(4, 3))
    assert out.dtype == torch.float32
    assert net.seen == [(True, torch.bfloat16)]
    wrapped.eval()
    assert not net.training  # eval() reaches the wrapped model


def test_results_record_the_inference_precision(cfg):  # noqa: F811
    bf16 = cfg.model_copy(
        update={
            "task": TaskConfig(name="next_frame", history=1, params={"roi": None}),
            "model": ModelConfig(name="unet3d", params={"base": 4, "depth": 1}, precision="bf16"),
        }
    )
    run_dir = train(bf16)
    ckpt = run_dir / "ckpts" / "best.pt"

    results = evaluate(bf16, ckpt=ckpt)
    assert all(v == v for v in results.summary.values() if isinstance(v, float))  # finite, not NaN
    assert json.loads((ckpt.parent / "results_test.json").read_text())["precision"] == "bf16"

    rollout_split(bf16, ckpt=ckpt, horizon=2)
    assert json.loads((ckpt.parent / "rollout_test.json").read_text())["precision"] == "bf16"

    default_ckpt = train(cfg) / "ckpts" / "best.pt"  # the fixture config states no precision: the project default
    evaluate(cfg, ckpt=default_ckpt)
    assert json.loads((default_ckpt.parent / "results_test.json").read_text())["precision"] == "tf32"
    meta = json.loads((default_ckpt.parents[1] / "run_meta.json").read_text())
    assert meta["model"]["precision"] == "tf32" and meta["run"]["segments"][0]["precision"] == "tf32"


def test_tf32_configures_the_matmul_precision_and_returns_the_model_itself():
    """tf32 is fp32 storage with tensor-core matmuls: a process-level switch, no wrapper."""
    net = _Probe()
    try:
        assert inference_model(net, "tf32", "cpu") is net
        assert torch.get_float32_matmul_precision() == "high"
        with autocast("tf32", "cpu"):
            assert not torch.is_autocast_enabled("cpu")
        inference_model(net, "fp32", "cpu")
        assert torch.get_float32_matmul_precision() == "highest"
        with pytest.raises(ValueError):
            configure("fp16")
    finally:
        torch.set_float32_matmul_precision("highest")
