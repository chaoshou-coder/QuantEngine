from __future__ import annotations

import numpy as np
import pytest

from quantengine.data.gpu_backend import BackendInfo
from quantengine.data.loader import DataBundle
from quantengine.engine.commission import FixedCommission
from quantengine.engine.portfolio import simulate_portfolio
from quantengine.engine.rules import TradingRules
from quantengine.engine.slippage import FixedSlippage


def _cpu_backend() -> BackendInfo:
    return BackendInfo(
        requested="auto",
        active="cpu",
        reason="test",
        gpu_available=False,
        cudf_available=False,
        cupy_available=False,
    )


def _make_bundle(
    open_prices: list[float],
    close_prices: list[float] | None = None,
    timestamps: np.ndarray | None = None,
) -> DataBundle:
    open_arr = np.asarray(open_prices, dtype=float).reshape(-1, 1)
    if close_prices is None:
        close_arr = open_arr.copy()
    else:
        close_arr = np.asarray(close_prices, dtype=float).reshape(-1, 1)

    n_bars = open_arr.shape[0]
    if timestamps is None:
        base = np.datetime64("2026-03-02T09:30:00")
        timestamps = base + np.arange(n_bars).astype("timedelta64[m]")

    high = np.maximum(open_arr, close_arr) + 1.0
    low = np.minimum(open_arr, close_arr) - 1.0
    volume = np.full((n_bars, 1), 10_000.0, dtype=float)
    return DataBundle(
        symbols=["XAUUSD"],
        timestamps=timestamps,
        open=open_arr,
        high=high,
        low=low,
        close=close_arr,
        volume=volume,
        backend=_cpu_backend(),
    )


def _run(
    data: DataBundle,
    signal: list[float],
    rules: TradingRules,
    initial_cash: float = 1_000.0,
    record_trades: bool = False,
):
    return simulate_portfolio(
        data=data,
        signal=np.asarray(signal, dtype=float).reshape(-1, 1),
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        rules=rules,
        initial_cash=initial_cash,
        contract_multiplier=1.0,
        record_trades=record_trades,
    )


def _event_types(result) -> list[str]:
    if result.risk_events is None:
        return []
    return [str(item["type"]) for item in result.risk_events]


def test_trading_rules_track_a_defaults_are_backward_compatible():
    rules = TradingRules()
    assert rules.max_risk_per_trade == 0.0
    assert rules.max_daily_loss == 0.0
    assert rules.max_weekly_loss == 0.0
    assert rules.max_drawdown_limit == 0.0
    assert rules.max_drawdown_action == "stop"
    assert rules.max_position == 0.0
    assert rules.max_addon_count == 0


def test_max_position_clips_target_and_records_position_limit_event():
    data = _make_bundle(open_prices=[100, 100, 100, 100, 100, 100])
    result = _run(
        data=data,
        signal=[0, 1, 1, 1, 1, 1],
        rules=TradingRules(max_position=0.4),
    )
    assert np.max(np.abs(result.positions)) <= 0.4 + 1e-9
    assert "position_limit" in _event_types(result)


def test_max_risk_per_trade_scales_delta_and_records_event():
    data = _make_bundle(open_prices=[100, 100, 100, 100, 100, 100])
    result = _run(
        data=data,
        signal=[0, 1, 1, 1, 1, 1],
        rules=TradingRules(max_risk_per_trade=0.05),
        initial_cash=1_000.0,
    )
    assert result.positions[2, 0] == pytest.approx(0.5, rel=1e-6, abs=1e-6)
    assert "position_limit" in _event_types(result)


def test_max_addon_count_blocks_second_addon():
    data = _make_bundle(open_prices=[100, 100, 100, 100, 100, 100])
    result = _run(
        data=data,
        signal=[0.0, 0.4, 0.8, 1.0, 1.0, 1.0],
        rules=TradingRules(max_addon_count=1),
    )
    assert result.positions[4, 0] == pytest.approx(0.8, rel=1e-6, abs=1e-6)
    assert "position_limit" in _event_types(result)


def test_daily_loss_breach_flattens_following_bars_in_same_day():
    data = _make_bundle(
        open_prices=[100, 100, 100, 90, 80, 80, 80],
        close_prices=[100, 100, 90, 80, 80, 80, 80],
    )
    result = _run(
        data=data,
        signal=[0, 1, 1, 1, 1, 1, 1],
        rules=TradingRules(max_daily_loss=0.01),
        initial_cash=1_000.0,
    )
    assert np.allclose(result.positions[3:, 0], 0.0)
    assert "daily_loss_breach" in _event_types(result)


def test_weekly_loss_breach_flattens_following_bars_in_same_week():
    base = np.datetime64("2026-03-02T09:30:00")
    timestamps = base + np.arange(7).astype("timedelta64[D]")
    data = _make_bundle(
        open_prices=[100, 100, 100, 90, 80, 80, 80],
        close_prices=[100, 100, 90, 80, 80, 80, 80],
        timestamps=timestamps,
    )
    result = _run(
        data=data,
        signal=[0, 1, 1, 1, 1, 1, 1],
        rules=TradingRules(max_weekly_loss=0.01),
        initial_cash=1_000.0,
    )
    assert np.allclose(result.positions[4:, 0], 0.0)
    assert "weekly_loss_breach" in _event_types(result)


def test_drawdown_stop_blocks_new_exposure_after_breach():
    data = _make_bundle(
        open_prices=[100, 100, 100, 90, 90, 90],
        close_prices=[100, 100, 90, 90, 90, 90],
    )
    result = _run(
        data=data,
        signal=[0, 1, 1, 1, 1, 1],
        rules=TradingRules(max_drawdown_limit=0.01, max_drawdown_action="stop"),
        initial_cash=1_000.0,
    )
    assert np.allclose(result.positions[3:, 0], 0.0)
    assert "drawdown_breach" in _event_types(result)


def test_drawdown_reduce_halves_exposure_instead_of_full_stop():
    data = _make_bundle(
        open_prices=[100, 100, 100, 95, 95, 95],
        close_prices=[100, 100, 95, 95, 95, 95],
    )
    result = _run(
        data=data,
        signal=[0, 2, 2, 2, 2, 2],
        rules=TradingRules(max_drawdown_limit=0.005, max_drawdown_action="reduce"),
        initial_cash=1_000.0,
    )
    assert result.positions[3, 0] == pytest.approx(0.5, rel=1e-6, abs=1e-6)
    assert "drawdown_breach" in _event_types(result)
