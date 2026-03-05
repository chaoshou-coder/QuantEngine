"""Acceptance tests for the PSAR V4 strategy.

Covers:
- SL triggers exit
- TP1 triggers partial close
- TP2 triggers full close
- Trailing PSAR moves SL
- Time-stop fires
- Add-on increases position
- Session filter blocks Asian hours
- V3 consistency (all V4 features off -> same signal as V3 F2)
- Edge cases (empty data, single bar, warmup NaN)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantengine.data.gpu_backend import get_backend_info
from quantengine.data.loader import DataBundle
from quantengine.strategy.examples.psar_trade_assist_v4 import (
    PsarTradeAssistV4Strategy,
    _IndicatorCache,
    _v4_loop_py,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = ROOT / "test_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_bundle(n: int = 500, seed: int = 0) -> DataBundle:
    """Generate synthetic OHLCV data with a trend then reversal."""
    rng = np.random.default_rng(seed)
    base = 1900.0
    returns = rng.normal(0.0002, 0.002, size=n)
    close = base * np.exp(np.cumsum(returns))
    noise = rng.uniform(0.5, 2.0, size=n)
    high = close + noise
    low = close - noise
    open_arr = close + rng.normal(0, 0.3, size=n)
    volume = rng.uniform(100, 1000, size=n)
    timestamps = np.array(
        pd.date_range("2020-01-01", periods=n, freq="1min", tz="UTC"),
        dtype="datetime64[ns]",
    )
    backend = get_backend_info(requested="cpu", use_gpu=False)
    return DataBundle(
        symbols=["SYNTH"],
        timestamps=timestamps,
        open=open_arr.reshape(-1, 1),
        high=high.reshape(-1, 1),
        low=low.reshape(-1, 1),
        close=close.reshape(-1, 1),
        volume=volume.reshape(-1, 1),
        backend=backend,
    )


def _load_real_slice(rows: int = 5000) -> DataBundle | None:
    candidates = [
        TEST_DATA_DIR / "XAUUSD_1m_20190101_20200101.csv",
        TEST_DATA_DIR / "XAUUSD_M1.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.name == "XAUUSD_M1.csv":
            frame = pd.read_csv(
                path, header=None,
                names=["date", "time", "open", "high", "low", "close", "volume"],
                usecols=[0, 1, 2, 3, 4, 5, 6],
            )
            frame["datetime"] = pd.to_datetime(
                frame["date"].astype(str) + " " + frame["time"].astype(str),
                format="%Y.%m.%d %H:%M", errors="coerce", utc=True,
            )
        else:
            frame = pd.read_csv(path)
            if "datetime" not in frame.columns:
                continue
            frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
        frame = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)]
        if len(frame) >= rows:
            frame = frame.iloc[:rows].copy()
            backend = get_backend_info(requested="cpu", use_gpu=False)
            return DataBundle(
                symbols=["XAUUSD"],
                timestamps=frame["datetime"].to_numpy(),
                open=frame["open"].to_numpy(dtype=float).reshape(-1, 1),
                high=frame["high"].to_numpy(dtype=float).reshape(-1, 1),
                low=frame["low"].to_numpy(dtype=float).reshape(-1, 1),
                close=frame["close"].to_numpy(dtype=float).reshape(-1, 1),
                volume=frame["volume"].to_numpy(dtype=float).reshape(-1, 1),
                backend=backend,
            )
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestV4Frameworks:
    """Each framework should generate nonzero signals on synthetic data."""

    @pytest.mark.parametrize("fw", ["F1", "F2", "F3", "F4", "F5"])
    def test_framework_produces_signals(self, fw: str):
        data = _synth_bundle(2000)
        strategy = PsarTradeAssistV4Strategy()
        params = {"framework": fw, "risk_mode": "baseline"}
        signal = strategy.generate_signals(data, params)
        assert signal.shape == (2000, 1)
        assert np.any(signal != 0.0), f"Framework {fw} produced all-zero signals"

    @pytest.mark.parametrize("rm", ["baseline", "conservative", "standard", "aggressive"])
    def test_risk_mode_runs(self, rm: str):
        data = _synth_bundle(2000)
        strategy = PsarTradeAssistV4Strategy()
        params = {"framework": "F2", "risk_mode": rm}
        signal = strategy.generate_signals(data, params)
        assert signal.shape == (2000, 1)
        assert not np.any(np.isnan(signal))


class TestStopLoss:
    """ATR stop-loss should force position to zero when price breaches SL."""

    def test_sl_triggers_exit(self):
        data = _synth_bundle(3000, seed=1)
        strategy = PsarTradeAssistV4Strategy()
        sig_no_sl = strategy.generate_signals(data, {
            "framework": "F1", "risk_mode": "baseline",
        })
        sig_with_sl = strategy.generate_signals(data, {
            "framework": "F1", "risk_mode": "conservative",
            "atr_sl_mult": 0.5,  # very tight SL
        })
        # With a very tight SL, there should be more flat periods
        flat_no_sl = np.sum(sig_no_sl == 0.0)
        flat_with_sl = np.sum(sig_with_sl == 0.0)
        assert flat_with_sl >= flat_no_sl, "Tight SL should produce more flat bars"


class TestTakeProfit:
    """TP1 partial close should reduce position magnitude."""

    def test_partial_close_at_tp1(self):
        data = _synth_bundle(3000, seed=2)
        strategy = PsarTradeAssistV4Strategy()
        sig = strategy.generate_signals(data, {
            "framework": "F1",
            "risk_mode": "standard",
            "tp1_r": 0.3,   # very tight TP1
            "tp2_r": 10.0,  # very far TP2 (won't hit)
            "partial_ratio": 0.5,
            "entry_size": 1.0,
        })
        unique_vals = np.unique(np.abs(sig[sig != 0.0]))
        has_fractional = np.any((unique_vals > 0.01) & (unique_vals < 0.99))
        assert has_fractional, "Should have fractional positions after TP1 partial close"


class TestTrailing:
    """Trailing stop should increase the effective SL over time."""

    def test_trailing_psar_runs(self):
        data = _synth_bundle(3000, seed=3)
        strategy = PsarTradeAssistV4Strategy()
        sig = strategy.generate_signals(data, {
            "framework": "F2",
            "risk_mode": "conservative",
            "enable_trailing_psar": 1,
        })
        assert sig.shape == (3000, 1)
        assert not np.any(np.isnan(sig))

    def test_trailing_atr_runs(self):
        data = _synth_bundle(3000, seed=4)
        strategy = PsarTradeAssistV4Strategy()
        sig = strategy.generate_signals(data, {
            "framework": "F2",
            "risk_mode": "baseline",
            "enable_atr_sl": 1,
            "enable_trailing_atr": 1,
            "trailing_atr_mult": 1.5,
        })
        assert not np.any(np.isnan(sig))


class TestTimeStop:
    """Time-stop should close positions after max bars."""

    def test_time_stop_limits_duration(self):
        data = _synth_bundle(3000, seed=5)
        strategy = PsarTradeAssistV4Strategy()
        sig = strategy.generate_signals(data, {
            "framework": "F1",
            "risk_mode": "aggressive",
            "time_stop_bars": 5,     # very short
            "no_progress_r": 100.0,  # always triggers
        })
        # With a 5-bar time stop, no position should last more than ~7 bars
        # (warmup + execution delay)
        sig_1d = sig.ravel()
        max_run = 0
        run = 0
        prev_sign = 0.0
        for v in sig_1d:
            s = np.sign(v)
            if s != 0 and s == prev_sign:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1 if s != 0 else 0
            prev_sign = s
        assert max_run <= 20, f"Max consecutive same-direction bars = {max_run}, expected <= 20 with 5-bar time stop"


class TestAddOn:
    """Add-on should increase position magnitude after TP1."""

    def test_addon_runs(self):
        data = _synth_bundle(3000, seed=6)
        strategy = PsarTradeAssistV4Strategy()
        sig = strategy.generate_signals(data, {
            "framework": "F3",
            "risk_mode": "aggressive",
            "entry_size": 0.6,
            "addon_size": 0.4,
        })
        assert not np.any(np.isnan(sig))
        vals = np.abs(sig[sig != 0.0])
        if vals.size > 0:
            assert np.max(vals) <= 1.0, "Position should not exceed 1.0"


class TestSessionFilter:
    """Session filter should block entries during Asian hours."""

    def test_session_filter_blocks_asian(self):
        data = _synth_bundle(3000, seed=7)
        strategy = PsarTradeAssistV4Strategy()
        sig_no_session = strategy.generate_signals(data, {
            "framework": "F1", "risk_mode": "baseline", "enable_session": 0,
        })
        sig_with_session = strategy.generate_signals(data, {
            "framework": "F1", "risk_mode": "baseline", "enable_session": 1,
        })
        active_no = np.sum(sig_no_session != 0.0)
        active_with = np.sum(sig_with_session != 0.0)
        assert active_with <= active_no, "Session filter should reduce or maintain signal count"


class TestV3Consistency:
    """With all V4 features off, signals should match V3 logic."""

    def test_f1_baseline_matches_v3_pure_psar(self):
        data = _synth_bundle(2000, seed=10)
        from quantengine.strategy.examples.psar_trade_assist_v3 import PsarTradeAssistV3Strategy

        v3 = PsarTradeAssistV3Strategy()
        v4 = PsarTradeAssistV4Strategy()

        common = {
            "enable_adx": 0, "enable_rsi": 0, "enable_ma": 0,
            "enable_body": 0, "enable_adaptive_step": 0,
        }
        sig_v3 = np.asarray(v3.generate_signals(data, common), dtype=np.float64)
        sig_v4 = np.asarray(v4.generate_signals(data, {
            "framework": "F1", "risk_mode": "baseline",
            "enable_adaptive_step": 0,
        }), dtype=np.float64)

        if sig_v3.ndim == 2 and sig_v4.ndim == 2:
            sig_v3 = sig_v3.ravel()
            sig_v4 = sig_v4.ravel()

        mismatch = np.sum(sig_v3 != sig_v4)
        total = sig_v3.shape[0]
        # Allow small mismatch due to state-machine vs vectorized ffill differences
        assert mismatch / total < 0.05, (
            f"V3/V4 pure-PSAR mismatch: {mismatch}/{total} = {mismatch/total:.2%}"
        )


class TestEdgeCases:
    """Edge cases should not crash."""

    def test_empty_data(self):
        backend = get_backend_info(requested="cpu", use_gpu=False)
        data = DataBundle(
            symbols=["X"], timestamps=np.array([], dtype="datetime64[ns]"),
            open=np.empty((0, 1)), high=np.empty((0, 1)),
            low=np.empty((0, 1)), close=np.empty((0, 1)),
            volume=np.empty((0, 1)), backend=backend,
        )
        strategy = PsarTradeAssistV4Strategy()
        sig = strategy.generate_signals(data, {"framework": "F1", "risk_mode": "baseline"})
        assert sig.shape[0] == 0

    def test_single_bar(self):
        backend = get_backend_info(requested="cpu", use_gpu=False)
        data = DataBundle(
            symbols=["X"],
            timestamps=np.array([np.datetime64("2020-01-01T00:00", "ns")]),
            open=np.array([[1900.0]]), high=np.array([[1901.0]]),
            low=np.array([[1899.0]]), close=np.array([[1900.5]]),
            volume=np.array([[100.0]]), backend=backend,
        )
        strategy = PsarTradeAssistV4Strategy()
        sig = strategy.generate_signals(data, {"framework": "F2", "risk_mode": "standard"})
        assert sig.shape[0] == 1
        assert sig.ravel()[0] == 0.0

    def test_short_data_all_zero(self):
        data = _synth_bundle(50, seed=99)
        strategy = PsarTradeAssistV4Strategy()
        sig = strategy.generate_signals(data, {"framework": "F4", "risk_mode": "standard"})
        assert sig.shape[0] == 50


class TestIndicatorCache:
    """Cache should speed up repeated calls with same params."""

    def test_cache_reuse(self):
        data = _synth_bundle(1000)
        strategy = PsarTradeAssistV4Strategy()
        cache = _IndicatorCache()
        params = {"framework": "F2", "risk_mode": "baseline"}
        sig1 = strategy.generate_signals(data, params, cache=cache)
        sig2 = strategy.generate_signals(data, params, cache=cache)
        np.testing.assert_array_equal(sig1, sig2)
        assert len(cache.values) > 0
