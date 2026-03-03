from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None
from quantengine.data.gpu_backend import to_numpy


def _get_xp(values: Any):
    if cp is not None and isinstance(values, cp.ndarray):
        return cp
    return np


def _wilder_smooth_cpu_impl(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    alpha = 1.0 / window
    if values.ndim == 1:
        n = values.shape[0]
        if n > 0:
            out[0] = values[0]
        for i in range(1, n):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out

    n, m = values.shape
    for c in range(m):
        out[0, c] = values[0, c]
        for i in range(1, n):
            out[i, c] = alpha * values[i, c] + (1.0 - alpha) * out[i - 1, c]
    return out


def _ema_cpu_impl(values: np.ndarray, span: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    if values.ndim == 1:
        n = values.shape[0]
        if n > 0:
            out[0] = values[0]
        for i in range(1, n):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out

    n, m = values.shape
    for c in range(m):
        out[0, c] = values[0, c]
        for i in range(1, n):
            out[i, c] = alpha * values[i, c] + (1.0 - alpha) * out[i - 1, c]
    return out


try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


if njit is not None:

    @njit
    def _sma_numba(values: np.ndarray, window: int) -> np.ndarray:
        out = np.empty_like(values, dtype=np.float64)
        if values.ndim == 1:
            n = values.shape[0]
            for i in range(n):
                out[i] = np.nan
            if n < window:
                return out
            prefix = np.zeros(n + 1, dtype=np.float64)
            for i in range(n):
                prefix[i + 1] = prefix[i] + values[i]
            for i in range(window - 1, n):
                out[i] = (prefix[i + 1] - prefix[i + 1 - window]) / float(window)
            return out

        n, m = values.shape
        for i in range(n):
            for c in range(m):
                out[i, c] = np.nan
        if n < window:
            return out
        for c in range(m):
            prefix = np.zeros(n + 1, dtype=np.float64)
            for i in range(n):
                prefix[i + 1] = prefix[i] + values[i, c]
            for i in range(window - 1, n):
                out[i, c] = (prefix[i + 1] - prefix[i + 1 - window]) / float(window)
        return out

    @njit
    def _wilder_smooth_numba(values: np.ndarray, window: int) -> np.ndarray:
        out = np.zeros_like(values, dtype=np.float64)
        alpha = 1.0 / window
        if values.ndim == 1:
            n = values.shape[0]
            if n > 0:
                out[0] = values[0]
            for i in range(1, n):
                out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
            return out

        n, m = values.shape
        for c in range(m):
            out[0, c] = values[0, c]
            for i in range(1, n):
                out[i, c] = alpha * values[i, c] + (1.0 - alpha) * out[i - 1, c]
        return out

    @njit
    def _ema_numba(values: np.ndarray, span: int) -> np.ndarray:
        out = np.zeros_like(values, dtype=np.float64)
        alpha = 2.0 / (span + 1.0)
        if values.ndim == 1:
            n = values.shape[0]
            if n > 0:
                out[0] = values[0]
            for i in range(1, n):
                out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
            return out

        n, m = values.shape
        for c in range(m):
            out[0, c] = values[0, c]
            for i in range(1, n):
                out[i, c] = alpha * values[i, c] + (1.0 - alpha) * out[i - 1, c]
        return out


def sma(values: Any, window: int) -> Any:
    xp = _get_xp(values)
    if window <= 0:
        raise ValueError("window must be > 0")
    if xp is np and njit is not None:
        return _sma_numba(np.asarray(values, dtype=np.float64), window)
    csum = xp.cumsum(values, axis=0, dtype=float)
    out = xp.full_like(values, xp.nan, dtype=float)
    if values.shape[0] < window:
        return out
    out[window - 1] = csum[window - 1] / float(window)
    if window > 1:
        out[window:] = (csum[window:] - csum[:-window]) / float(window)
    return out


def wilder_smooth(values: Any, window: int) -> Any:
    xp = _get_xp(values)
    if window <= 0:
        raise ValueError("window must be > 0")
    values_np = np.asarray(to_numpy(values), dtype=np.float64)
    if njit is not None:
        out_np = _wilder_smooth_numba(values_np, window)
    else:
        out_np = _wilder_smooth_cpu_impl(values_np, window)
    return xp.asarray(out_np)


def ema(values: Any, span: int) -> Any:
    xp = _get_xp(values)
    if span <= 0:
        raise ValueError("span must be > 0")
    values_np = np.asarray(to_numpy(values), dtype=np.float64)
    if njit is not None:
        out_np = _ema_numba(values_np, span)
    else:
        out_np = _ema_cpu_impl(values_np, span)
    return xp.asarray(out_np)


def rsi(close: Any, window: int = 14) -> Any:
    xp = _get_xp(close)
    delta = xp.zeros_like(close, dtype=float)
    delta[1:] = close[1:] - close[:-1]
    gains = xp.where(delta > 0, delta, 0.0)
    losses = xp.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_smooth(gains, window)
    avg_loss = wilder_smooth(losses, window)
    rs = xp.divide(avg_gain, avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(close: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Any, Any, Any]:
    fast_line = ema(close, fast)
    slow_line = ema(close, slow)
    macd_line = fast_line - slow_line
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(high: Any, low: Any, close: Any, window: int = 14) -> Any:
    xp = _get_xp(close)
    prev_close = xp.vstack([close[0:1], close[:-1]])
    tr_1 = high - low
    tr_2 = xp.abs(high - prev_close)
    tr_3 = xp.abs(low - prev_close)
    tr = xp.maximum(tr_1, xp.maximum(tr_2, tr_3))
    return ema(tr, window)
