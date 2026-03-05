from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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
    close_arr = xp.asarray(close, dtype=float)
    prev_close = xp.empty_like(close_arr, dtype=float)
    prev_close[0:1] = close_arr[0:1]
    prev_close[1:] = close_arr[:-1]
    tr_1 = high - low
    tr_2 = xp.abs(high - prev_close)
    tr_3 = xp.abs(low - prev_close)
    tr = xp.maximum(tr_1, xp.maximum(tr_2, tr_3))
    return ema(tr, window)


def parabolic_sar(high: Any, low: Any, close: Any, step: float = 0.02, maximum: float = 0.2) -> Any:
    """Vectorised Parabolic SAR. Returns array same shape as *close*."""
    high_np = np.asarray(to_numpy(high), dtype=np.float64)
    low_np = np.asarray(to_numpy(low), dtype=np.float64)
    close_np = np.asarray(to_numpy(close), dtype=np.float64)

    if close_np.ndim == 1:
        out = _parabolic_sar_1d(high_np, low_np, close_np, step, maximum)
    else:
        n, m = close_np.shape
        out = np.empty_like(close_np)
        for col in range(m):
            out[:, col] = _parabolic_sar_1d(high_np[:, col], low_np[:, col], close_np[:, col], step, maximum)
    xp = _get_xp(close)
    return xp.asarray(out)


def _parabolic_sar_1d_py(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    step: float,
    maximum: float,
) -> np.ndarray:
    n = len(close)
    sar = np.zeros(n, dtype=np.float64)
    if n < 2:
        return sar

    is_long = close[1] > close[0]
    af = step
    ep = high[0] if is_long else low[0]
    sar[0] = low[0] if is_long else high[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        sar_i = prev_sar + af * (ep - prev_sar)

        if is_long:
            sar_i = min(sar_i, low[i - 1])
            if i >= 2:
                sar_i = min(sar_i, low[i - 2])
            if sar_i > low[i]:
                is_long = False
                sar_i = ep
                af = step
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, maximum)
        else:
            sar_i = max(sar_i, high[i - 1])
            if i >= 2:
                sar_i = max(sar_i, high[i - 2])
            if sar_i < high[i]:
                is_long = True
                sar_i = ep
                af = step
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, maximum)

        sar[i] = sar_i
    return sar


if njit is not None:
    _parabolic_sar_1d_numba = njit(cache=True)(_parabolic_sar_1d_py)
else:
    _parabolic_sar_1d_numba = None


def _parabolic_sar_1d(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    step: float,
    maximum: float,
) -> np.ndarray:
    if _parabolic_sar_1d_numba is not None:
        return _parabolic_sar_1d_numba(high, low, close, step, maximum)
    return _parabolic_sar_1d_py(high, low, close, step, maximum)


def adx(high: Any, low: Any, close: Any, window: int = 14) -> tuple[Any, Any, Any]:
    """ADX indicator. Returns (adx_line, plus_di, minus_di)."""
    xp = _get_xp(close)
    high_np = np.asarray(to_numpy(high), dtype=np.float64)
    low_np = np.asarray(to_numpy(low), dtype=np.float64)
    close_np = np.asarray(to_numpy(close), dtype=np.float64)

    if close_np.ndim == 1:
        adx_out, pdi, mdi = _adx_1d(high_np, low_np, close_np, window)
    else:
        n, m = close_np.shape
        adx_out = np.empty_like(close_np)
        pdi = np.empty_like(close_np)
        mdi = np.empty_like(close_np)
        for col in range(m):
            adx_out[:, col], pdi[:, col], mdi[:, col] = _adx_1d(
                high_np[:, col], low_np[:, col], close_np[:, col], window
            )
    return xp.asarray(adx_out), xp.asarray(pdi), xp.asarray(mdi)


def _adx_1d_py(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    adx_out = np.zeros(n, dtype=np.float64)
    pdi = np.zeros(n, dtype=np.float64)
    mdi = np.zeros(n, dtype=np.float64)
    if n < 2:
        return adx_out, pdi, mdi

    up_move = np.zeros(n, dtype=np.float64)
    down_move = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        up_move[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        down_move[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    alpha = 1.0 / window
    sm_tr = np.zeros(n, dtype=np.float64)
    sm_up = np.zeros(n, dtype=np.float64)
    sm_dn = np.zeros(n, dtype=np.float64)

    sm_tr[1] = tr[1]
    sm_up[1] = up_move[1]
    sm_dn[1] = down_move[1]
    for i in range(2, n):
        sm_tr[i] = sm_tr[i - 1] * (1.0 - alpha) + tr[i]
        sm_up[i] = sm_up[i - 1] * (1.0 - alpha) + up_move[i]
        sm_dn[i] = sm_dn[i - 1] * (1.0 - alpha) + down_move[i]

    for i in range(1, n):
        if sm_tr[i] > 0:
            pdi[i] = 100.0 * sm_up[i] / sm_tr[i]
            mdi[i] = 100.0 * sm_dn[i] / sm_tr[i]
        di_sum = pdi[i] + mdi[i]
        if di_sum > 0:
            dx = 100.0 * abs(pdi[i] - mdi[i]) / di_sum
        else:
            dx = 0.0
        if i == 1:
            adx_out[i] = dx
        else:
            adx_out[i] = adx_out[i - 1] * (1.0 - alpha) + dx * alpha

    return adx_out, pdi, mdi


if njit is not None:
    _adx_1d_numba = njit(cache=True)(_adx_1d_py)
else:
    _adx_1d_numba = None


def _adx_1d(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _adx_1d_numba is not None:
        return _adx_1d_numba(high, low, close, window)
    return _adx_1d_py(high, low, close, window)


def bollinger_bands(close: Any, period: int = 20, std_dev: float = 2.0) -> tuple[Any, Any, Any]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    xp = _get_xp(close)
    middle = sma(close, period)
    close_np = np.asarray(to_numpy(close), dtype=np.float64)
    was_1d = close_np.ndim == 1
    if was_1d:
        close_np = close_np.reshape(-1, 1)

    n, m = close_np.shape
    std_arr = np.full((n, m), np.nan, dtype=np.float64)
    if n >= period:
        for col in range(m):
            windows = sliding_window_view(close_np[:, col], window_shape=period)
            std_arr[period - 1 :, col] = np.std(windows, axis=-1, ddof=0)

    if was_1d:
        std_arr = std_arr.ravel()

    std_xp = xp.asarray(std_arr)
    upper = middle + std_dev * std_xp
    lower = middle - std_dev * std_xp
    return upper, middle, lower


def stochastic(high: Any, low: Any, close: Any, k_period: int = 14, d_period: int = 3) -> tuple[Any, Any]:
    """Stochastic Oscillator. Returns (K, D)."""
    xp = _get_xp(close)
    high_np = np.asarray(to_numpy(high), dtype=np.float64)
    low_np = np.asarray(to_numpy(low), dtype=np.float64)
    close_np = np.asarray(to_numpy(close), dtype=np.float64)
    was_1d = close_np.ndim == 1
    if was_1d:
        high_np = high_np.reshape(-1, 1)
        low_np = low_np.reshape(-1, 1)
        close_np = close_np.reshape(-1, 1)

    n, m = close_np.shape
    k_arr = np.full((n, m), np.nan, dtype=np.float64)
    if n >= k_period:
        for col in range(m):
            h_win = sliding_window_view(high_np[:, col], window_shape=k_period)
            l_win = sliding_window_view(low_np[:, col], window_shape=k_period)
            highest = np.max(h_win, axis=-1)
            lowest = np.min(l_win, axis=-1)
            denom = highest - lowest
            denom = np.where(denom > 0, denom, 1.0)
            k_arr[k_period - 1 :, col] = (close_np[k_period - 1 :, col] - lowest) / denom * 100.0

    if was_1d:
        k_arr = k_arr.ravel()

    k_xp = xp.asarray(k_arr)
    k_xp = xp.nan_to_num(k_xp, nan=50.0)
    d_xp = sma(k_xp, d_period)
    return k_xp, d_xp


def donchian(high: Any, low: Any, period: int = 20) -> tuple[Any, Any]:
    """Donchian Channel. Returns (upper, lower)."""
    xp = _get_xp(high)
    high_np = np.asarray(to_numpy(high), dtype=np.float64)
    low_np = np.asarray(to_numpy(low), dtype=np.float64)
    was_1d = high_np.ndim == 1
    if was_1d:
        high_np = high_np.reshape(-1, 1)
        low_np = low_np.reshape(-1, 1)

    n, m = high_np.shape
    upper = np.full((n, m), np.nan, dtype=np.float64)
    lower = np.full((n, m), np.nan, dtype=np.float64)
    if n >= period:
        for col in range(m):
            h_win = sliding_window_view(high_np[:, col], window_shape=period)
            l_win = sliding_window_view(low_np[:, col], window_shape=period)
            upper[period - 1 :, col] = np.max(h_win, axis=-1)
            lower[period - 1 :, col] = np.min(l_win, axis=-1)

    if was_1d:
        upper = upper.ravel()
        lower = lower.ravel()
    return xp.asarray(upper), xp.asarray(lower)


def cci(high: Any, low: Any, close: Any, period: int = 14) -> Any:
    """Commodity Channel Index."""
    xp = _get_xp(close)
    high_np = np.asarray(to_numpy(high), dtype=np.float64)
    low_np = np.asarray(to_numpy(low), dtype=np.float64)
    close_np = np.asarray(to_numpy(close), dtype=np.float64)
    tp = (high_np + low_np + close_np) / 3.0
    was_1d = tp.ndim == 1
    if was_1d:
        tp = tp.reshape(-1, 1)

    n, m = tp.shape
    cci_arr = np.full((n, m), np.nan, dtype=np.float64)
    if n >= period:
        for col in range(m):
            tp_col = tp[:, col]
            windows = sliding_window_view(tp_col, window_shape=period)
            tp_mean = np.mean(windows, axis=-1)
            mean_dev = np.mean(np.abs(windows - tp_mean[:, None]), axis=-1)
            denom = 0.015 * mean_dev
            denom = np.where(denom > 0, denom, 1e-12)
            cci_arr[period - 1 :, col] = (tp_col[period - 1 :] - tp_mean) / denom

    if was_1d:
        cci_arr = cci_arr.ravel()
    return xp.asarray(cci_arr)
