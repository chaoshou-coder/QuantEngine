from __future__ import annotations

import numpy as np
import pytest

from quantengine.data.gpu_backend import BackendInfo
from quantengine.data.loader import DataBundle
from quantengine.engine.commission import FixedCommission
from quantengine.engine.portfolio import simulate_portfolio, simulate_portfolio_batch
from quantengine.engine.slippage import FixedSlippage


def _make_single_asset_bundle(open_prices: list[float], close_prices: list[float], volumes: list[float] | None = None) -> DataBundle:
    n = len(open_prices)
    open_arr = np.asarray(open_prices, dtype=float).reshape(n, 1)
    close_arr = np.asarray(close_prices, dtype=float).reshape(n, 1)
    high_arr = np.maximum(open_arr, close_arr) + 1.0
    low_arr = np.minimum(open_arr, close_arr) - 1.0
    volume_arr = (
        np.asarray(volumes, dtype=float).reshape(n, 1)
        if volumes is not None
        else np.full((n, 1), 1_000.0, dtype=float)
    )
    timestamps = np.array([f"2026-03-01T00:{i:02d}:00" for i in range(n)], dtype="datetime64[ns]")
    return DataBundle(
        symbols=["AAA"],
        timestamps=timestamps,
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        volume=volume_arr,
        backend=BackendInfo("auto", "cpu", "test", False, False, False),
    )


def _make_empty_bundle() -> DataBundle:
    empty = np.empty((0, 1), dtype=float)
    return DataBundle(
        symbols=["AAA"],
        timestamps=np.array([], dtype="datetime64[ns]"),
        open=empty,
        high=empty,
        low=empty,
        close=empty,
        volume=empty,
        backend=BackendInfo("auto", "cpu", "test", False, False, False),
    )


def test_simulate_portfolio_matches_batch_single_combo() -> None:
    data = _make_single_asset_bundle(
        open_prices=[100.0, 101.0, 102.0, 103.0, 104.0],
        close_prices=[100.0, 102.0, 101.0, 105.0, 106.0],
    )
    signal = np.array([[0.0], [1.0], [1.0], [0.0], [0.0]], dtype=float)
    slippage = FixedSlippage(points=0.0)
    commission = FixedCommission(value=0.0)
    single = simulate_portfolio(
        data=data,
        signal=signal,
        slippage=slippage,
        commission=commission,
        rules=None,
        initial_cash=10_000.0,
        record_trades=False,
    )
    equity_batch, returns_batch = simulate_portfolio_batch(
        data=data,
        signal=signal,
        slippage=slippage,
        commission=commission,
        rules=None,
        initial_cash=10_000.0,
    )
    np.testing.assert_allclose(single.equity_curve, equity_batch[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(single.returns, returns_batch[:, 0], rtol=1e-12, atol=1e-12)


def test_simulate_portfolio_batch_supports_multi_combo() -> None:
    data = _make_single_asset_bundle(
        open_prices=[100.0, 101.0, 102.0, 103.0],
        close_prices=[100.0, 102.0, 101.0, 104.0],
    )
    signal = np.array(
        [
            [[0.0, 0.0]],
            [[1.0, -1.0]],
            [[1.0, -1.0]],
            [[0.0, 0.0]],
        ],
        dtype=float,
    )
    equity_batch, returns_batch = simulate_portfolio_batch(
        data=data,
        signal=signal,
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        initial_cash=1_000.0,
    )
    assert equity_batch.shape == (4, 2)
    assert returns_batch.shape == (4, 2)
    np.testing.assert_allclose(equity_batch[0], np.array([1_000.0, 1_000.0]))


def test_simulate_portfolio_batch_rejects_invalid_signal_dim() -> None:
    data = _make_single_asset_bundle(
        open_prices=[100.0, 101.0, 102.0],
        close_prices=[100.0, 101.0, 102.0],
    )
    with pytest.raises(ValueError, match="signal must be 2D or 3D array"):
        simulate_portfolio_batch(
            data=data,
            signal=np.array([0.0, 1.0, 1.0]),
            slippage=FixedSlippage(points=0.0),
            commission=FixedCommission(value=0.0),
        )


def test_simulate_portfolio_rejects_mismatched_signal_shape() -> None:
    data = _make_single_asset_bundle(
        open_prices=[100.0, 101.0, 102.0, 103.0],
        close_prices=[100.0, 101.0, 102.0, 103.0],
    )
    with pytest.raises(ValueError, match="signal shape"):
        simulate_portfolio(
            data=data,
            signal=np.array([[0.0], [1.0], [1.0]], dtype=float),
            slippage=FixedSlippage(points=0.0),
            commission=FixedCommission(value=0.0),
            rules=None,
        )


def test_simulate_portfolio_records_trade_log() -> None:
    data = _make_single_asset_bundle(
        open_prices=[100.0, 101.0, 102.0, 103.0],
        close_prices=[100.0, 102.0, 101.0, 104.0],
    )
    signal = np.array([[0.0], [1.0], [0.0], [0.0]], dtype=float)
    result = simulate_portfolio(
        data=data,
        signal=signal,
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        rules=None,
        initial_cash=1_000.0,
        record_trades=True,
    )
    assert len(result.trades) == 2
    assert result.trades[0]["side"] == "BUY"
    assert result.trades[1]["side"] == "SELL"


def test_simulate_portfolio_clips_signal_and_shifts_to_positions() -> None:
    data = _make_single_asset_bundle(
        open_prices=[100.0, 101.0, 102.0, 103.0],
        close_prices=[100.0, 101.0, 102.0, 103.0],
    )
    signal = np.array([[2.0], [2.0], [-2.0], [0.0]], dtype=float)
    result = simulate_portfolio(
        data=data,
        signal=signal,
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        rules=None,
        initial_cash=1_000.0,
        record_trades=False,
    )
    np.testing.assert_allclose(result.positions[:, 0], np.array([0.0, 1.0, 1.0, -1.0]))


def test_simulate_portfolio_batch_clips_signal_tensor() -> None:
    data = _make_single_asset_bundle(
        open_prices=[100.0, 101.0, 102.0, 103.0],
        close_prices=[100.0, 101.0, 102.0, 103.0],
    )
    raw_signal = np.array([[[2.0, -2.0]], [[2.0, -2.0]], [[-3.0, 3.0]], [[0.0, 0.0]]], dtype=float)
    clipped_signal = np.clip(raw_signal, -1.0, 1.0)
    eq_raw, ret_raw = simulate_portfolio_batch(
        data=data,
        signal=raw_signal,
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        initial_cash=500.0,
    )
    eq_clipped, ret_clipped = simulate_portfolio_batch(
        data=data,
        signal=clipped_signal,
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        initial_cash=500.0,
    )
    np.testing.assert_allclose(eq_raw, eq_clipped)
    np.testing.assert_allclose(ret_raw, ret_clipped)


def test_simulate_portfolio_batch_returns_empty_when_combo_count_is_zero() -> None:
    data = _make_single_asset_bundle(
        open_prices=[100.0, 101.0, 102.0],
        close_prices=[100.0, 101.0, 102.0],
    )
    signal = np.empty((3, 1, 0), dtype=float)
    equity_batch, returns_batch = simulate_portfolio_batch(
        data=data,
        signal=signal,
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
    )
    assert equity_batch.shape == (3, 0)
    assert returns_batch.shape == (3, 0)


def test_simulate_portfolio_batch_returns_empty_for_zero_bars() -> None:
    data = _make_empty_bundle()
    signal = np.empty((0, 1), dtype=float)
    equity_batch, returns_batch = simulate_portfolio_batch(
        data=data,
        signal=signal,
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
    )
    assert equity_batch.shape == (0, 0)
    assert returns_batch.shape == (0, 0)
