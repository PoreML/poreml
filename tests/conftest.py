import os
import signal
from pathlib import Path

import pytest

from poreml.models.unet import UNet3D
from poreml.registry import MODELS
from tests.fixtures import write_fake_dataset


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A temporary directory holding a three-run synthetic drainage campaign."""
    write_fake_dataset(tmp_path)
    return tmp_path


class _Interruptible(UNet3D):
    """A small UNet whose k-th *training* forward either raises or sends SIGTERM to the
    process — the two ways a real run stops mid-epoch. Counters live on the class so a
    test can arm them before `train` builds the model."""

    forwards = 0
    crash_at: int | None = None
    sigterm_at: int | None = None

    def forward(self, x):
        if self.training:
            type(self).forwards += 1
            if type(self).crash_at is not None and type(self).forwards == type(self).crash_at:
                raise RuntimeError("simulated crash")
            if type(self).sigterm_at is not None and type(self).forwards == type(self).sigterm_at:
                os.kill(os.getpid(), signal.SIGTERM)
        return super().forward(x)


@pytest.fixture
def interruptible():
    MODELS.register("_test_interruptible")(_Interruptible)
    _Interruptible.forwards, _Interruptible.crash_at, _Interruptible.sigterm_at = 0, None, None
    yield _Interruptible
    MODELS._items.pop("_test_interruptible")
