from __future__ import annotations

import numpy as np
import pytest

from quantengine.config import QuantEngineConfig
from quantengine.data.gpu_backend import BackendInfo
from quantengine.data.loader import DataBundle


@pytest.fixture
def sample_data_bundle() -> DataBundle:
    timestamps = np.array(
        ["2026-03-01T09:30:00", "2026-03-01T09:31:00", "2026-03-01T09:32:00"],
        dtype="datetime64[ns]",
    )
    close = np.array([[100.0, 200.0], [101.0, 199.0], [103.0, 198.0]])
    return DataBundle(
        symbols=["AAA", "BBB"],
        timestamps=timestamps,
        open=close.copy(),
        high=close.copy() + 1.0,
        low=close.copy() - 1.0,
        close=close.copy(),
        volume=np.array([[1000.0, 1100.0], [1200.0, 1300.0], [1400.0, 1500.0]]),
        backend=BackendInfo(
            requested="auto",
            active="cpu",
            reason="test",
            gpu_available=False,
            cudf_available=False,
            cupy_available=False,
        ),
    )


@pytest.fixture
def default_config() -> QuantEngineConfig:
    return QuantEngineConfig()

