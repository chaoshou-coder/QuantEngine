from __future__ import annotations

import numpy as np
import pytest

from quantengine.indicators.technical import atr, ema, macd, rsi, sma


def _manual_ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, values.shape[0]):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def test_sma_matches_manual_calculation_1d() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    actual = sma(values, window=3)
    expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_sma_handles_2d_input() -> None:
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    actual = sma(values, window=2)
    expected = np.array([[np.nan, np.nan], [1.5, 15.0], [2.5, 25.0], [3.5, 35.0]])
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("window", [0, -1])
def test_sma_rejects_non_positive_window(window: int) -> None:
    with pytest.raises(ValueError, match="window must be > 0"):
        sma(np.array([1.0, 2.0]), window=window)


def test_sma_returns_all_nan_when_window_too_large() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=float)
    actual = sma(values, window=5)
    assert np.isnan(actual).all()


def test_ema_matches_manual_calculation_1d() -> None:
    values = np.array([10.0, 12.0, 11.0, 13.0], dtype=float)
    actual = ema(values, span=3)
    expected = _manual_ema(values, span=3)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_ema_handles_2d_input() -> None:
    values = np.array([[10.0, 100.0], [12.0, 98.0], [14.0, 97.0]], dtype=float)
    actual = ema(values, span=2)
    expected_col0 = _manual_ema(values[:, 0], span=2)
    expected_col1 = _manual_ema(values[:, 1], span=2)
    expected = np.column_stack([expected_col0, expected_col1])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("span", [0, -2])
def test_ema_rejects_non_positive_span(span: int) -> None:
    with pytest.raises(ValueError, match="span must be > 0"):
        ema(np.array([1.0, 2.0]), span=span)


def test_rsi_monotonic_uptrend_is_near_100() -> None:
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    out = rsi(close, window=3)
    assert out[-1] > 99.0


def test_rsi_monotonic_downtrend_is_near_0() -> None:
    close = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
    out = rsi(close, window=3)
    assert out[-1] < 1.0


def test_macd_hist_equals_macd_minus_signal() -> None:
    close = np.array([10.0, 11.0, 12.0, 11.0, 13.0, 14.0], dtype=float)
    macd_line, signal_line, hist = macd(close, fast=3, slow=5, signal=2)
    np.testing.assert_allclose(hist, macd_line - signal_line, rtol=1e-12, atol=1e-12)


def test_macd_matches_manual_ema_diff() -> None:
    close = np.array([10.0, 11.0, 12.0, 11.0, 13.0, 14.0], dtype=float)
    fast = _manual_ema(close, span=3)
    slow = _manual_ema(close, span=5)
    macd_line, signal_line, _ = macd(close, fast=3, slow=5, signal=2)
    expected_macd = fast - slow
    expected_signal = _manual_ema(expected_macd, span=2)
    np.testing.assert_allclose(macd_line, expected_macd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(signal_line, expected_signal, rtol=1e-12, atol=1e-12)


def test_atr_matches_true_range_ema() -> None:
    high = np.array([11.0, 12.0, 13.0, 14.0], dtype=float)
    low = np.array([9.0, 10.0, 11.0, 12.0], dtype=float)
    close = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)
    prev_close = np.array([10.0, 10.0, 11.0, 12.0], dtype=float)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    expected = _manual_ema(tr, span=3)
    actual = atr(high, low, close, window=3)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
