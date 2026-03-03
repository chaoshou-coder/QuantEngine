from __future__ import annotations

from typing import Any

import numpy as np

from quantengine.indicators.technical import sma
from quantengine.strategy.base import BaseStrategy, ParameterSpace
from quantengine.strategy.registry import register_strategy

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


@register_strategy("sma_cross")
class SmaCrossStrategy(BaseStrategy):
    def parameters(self) -> dict[str, ParameterSpace]:
        return {
            "fast": ParameterSpace(kind="int", low=5, high=30, step=1),
            "slow": ParameterSpace(kind="int", low=20, high=120, step=5),
        }

    def generate_signals(self, data, params: dict[str, Any]):
        fast = int(params.get("fast", 10))
        slow = int(params.get("slow", 30))
        if fast <= 0 or slow <= 0:
            raise ValueError("fast 和 slow 必须为正整数")
        if fast >= slow:
            raise ValueError("fast 必须小于 slow")
        if data.close.shape[0] < slow:
            raise ValueError("数据长度不足以计算 SMA")

        close = data.close
        fast_ma = sma(close, fast)
        slow_ma = sma(close, slow)
        signal = (fast_ma > slow_ma).astype(float)
        signal = signal * 2.0 - 1.0

        if cp is not None and isinstance(signal, cp.ndarray):
            signal = cp.nan_to_num(signal, nan=0.0)
        else:
            signal = np.nan_to_num(signal, nan=0.0)
        return signal
