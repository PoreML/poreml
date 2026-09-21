"""What produced an artifact.

A result row is only reproducible if it resolves to code, evaluator, data, split, seed
and hardware. `collect` gathers those once; `train` writes them to `provenance.json` and
`evaluate` embeds them in `results_<split>.json`. Nothing here is required to succeed —
a missing git binary yields `None`, never a failed run — but nothing is guessed either.
"""

import hashlib
import platform
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from .config import Config
from .metrics import METRICS_VERSION


def _git_commit() -> str | None:
    """Commit of the poreml checkout this code was imported from, or None outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _package_version() -> str | None:
    try:
        return version("poreml")
    except PackageNotFoundError:
        return None


def sha256_of(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(cfg: Config, device: str) -> dict[str, Any]:
    """Everything a result must be traceable to, as a JSON-ready dict."""
    return {
        "poreml_version": _package_version(),
        "poreml_commit": _git_commit(),
        "metrics_version": METRICS_VERSION,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.startswith("cuda") and torch.cuda.is_available()
            else platform.processor() or None
        ),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "seed": cfg.seed,
        "data_root": str(Path(cfg.data.root).resolve()),
        "campaign": cfg.data.campaign,
        "split": {"path": str(Path(cfg.data.split).resolve()), "sha256": sha256_of(cfg.data.split)},
    }
