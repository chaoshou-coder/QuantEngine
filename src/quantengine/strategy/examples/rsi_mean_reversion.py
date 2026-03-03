from __future__ import annotations

from typing import Any

import numpy as np

from quantengine.indicators.technical import rsi
from quantengine.strategy.base import BaseStrategy, ParameterSpace
from quantengine.strategy.registry import register_strategy

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


@register_strategy("rsi_mean_reversion")
class RSIMeanReversionStrategy(BaseStrategy):
    def parameters(self) -> dict[str, ParameterSpace]:
        return {
            "window": ParameterSpace(kind="int", low=6, high=30, step=1),
            "lower": ParameterSpace(kind="float", low=10.0, high=40.0, step=5.0),
            "upper": ParameterSpace(kind="float", low=60.0, high=90.0, step=5.0),
        }

    def generate_signals(self, data, params: dict[str, Any]):
        window = int(params.get("window", 14))
        lower = float(params.get("lower", 30.0))
        upper = float(params.get("upper", 70.0))
        if lower >= upper:
            raise ValueError("lower 必须小于 upper")

        score = rsi(data.close, window=window)
        if cp is not None and isinstance(score, cp.ndarray):
            long_sig = (score < lower).astype(cp.float64)
            short_sig = (score > upper).astype(cp.float64)
            signal = long_sig - short_sig
            return cp.nan_to_num(signal, nan=0.0)

        long_sig = (score < lower).astype(np.float64)
        short_sig = (score > upper).astype(np.float64)
        signal = long_sig - short_sig
        return np.nan_to_num(signal, nan=0.0)
