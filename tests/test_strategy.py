from __future__ import annotations

import numpy as np

from quantengine.data.loader import DataBundle
from quantengine.data.gpu_backend import BackendInfo
from quantengine.strategy import get_strategy, list_strategies


def test_list_strategies_contains_builtins():
    names = set(list_strategies())
    assert "sma_cross" in names
    assert "rsi_mean_reversion" in names


def test_get_strategy_generates_signal_shape():
    strategy = get_strategy("sma_cross")
    data = DataBundle(
        symbols=["AAA"],
        timestamps=np.array(["2026-03-01T09:30:00", "2026-03-01T09:31:00", "2026-03-01T09:32:00"], dtype="datetime64[ns]"),
        open=np.array([[100.0], [101.0], [102.0]]),
        high=np.array([[101.0], [102.0], [103.0]]),
        low=np.array([[99.0], [100.0], [101.0]]),
        close=np.array([[100.0], [101.0], [103.0]]),
        volume=np.array([[1000.0], [1000.0], [1000.0]]),
        backend=BackendInfo("auto", "cpu", "test", False, False, False),
    )
    signal = strategy.generate_signals(data, {"fast": 1, "slow": 2})
    assert signal.shape == (3, 1)
