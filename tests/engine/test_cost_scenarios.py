from __future__ import annotations

import numpy as np
import pytest

from quantengine.data.gpu_backend import BackendInfo
from quantengine.data.loader import DataBundle
from quantengine.engine.backtest import BacktestEngine, CostScenario, run_cost_scenarios
from quantengine.engine.commission import FixedCommission
from quantengine.engine.rules import TradingRules
from quantengine.engine.slippage import FixedSlippage
from quantengine.strategy.base import BaseStrategy


class CountingTrendStrategy(BaseStrategy):
    name = "counting_trend"

    def __init__(self) -> None:
        self.generate_calls = 0

    def parameters(self):
        return {}

    def generate_signals(self, data: DataBundle, params: dict):
        self.generate_calls += 1
        # bar1 开始持有，保证存在交易成本
        return np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)


def _build_single_asset_data() -> DataBundle:
    timestamps = np.array(
        [
            "2026-03-01T09:30:00",
            "2026-03-01T09:31:00",
            "2026-03-01T09:32:00",
            "2026-03-01T09:33:00",
        ],
        dtype="datetime64[ns]",
    )
    close = np.array([[100.0], [102.0], [104.0], [106.0]], dtype=float)
    return DataBundle(
        symbols=["AAA"],
        timestamps=timestamps,
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close.copy(),
        volume=np.array([[1000.0], [1000.0], [1000.0], [1000.0]], dtype=float),
        backend=BackendInfo(
            requested="auto",
            active="cpu",
            reason="test",
            gpu_available=False,
            cudf_available=False,
            cupy_available=False,
        ),
    )


def _build_engine() -> BacktestEngine:
    return BacktestEngine(
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        rules=TradingRules(margin_ratio=0.0),
        initial_cash=10_000.0,
        contract_multiplier=1.0,
        risk_free_rate=0.0,
        periods_per_year=252,
    )


def test_run_cost_scenarios_generates_signals_once_and_returns_all_reports():
    engine = _build_engine()
    data = _build_single_asset_data()
    strategy = CountingTrendStrategy()
    scenarios = [
        CostScenario(
            name="low",
            slippage_model="fixed",
            slippage_value=0.0,
            commission_model="fixed",
            commission_value=0.0,
        ),
        CostScenario(
            name="high",
            slippage_model="fixed",
            slippage_value=0.5,
            commission_model="fixed",
            commission_value=5.0,
        ),
    ]

    reports = run_cost_scenarios(
        engine=engine,
        data=data,
        strategy=strategy,
        params={},
        scenarios=scenarios,
    )

    assert strategy.generate_calls == 1
    assert set(reports.keys()) == {"low", "high"}
    assert reports["low"].metadata["cost_scenario"]["name"] == "low"
    assert reports["high"].metadata["cost_scenario"]["name"] == "high"


def test_run_cost_scenarios_higher_cost_should_not_improve_total_return():
    engine = _build_engine()
    data = _build_single_asset_data()
    strategy = CountingTrendStrategy()
    scenarios = [
        CostScenario(
            name="low",
            slippage_model="fixed",
            slippage_value=0.0,
            commission_model="fixed",
            commission_value=0.0,
        ),
        CostScenario(
            name="high",
            slippage_model="fixed",
            slippage_value=1.0,
            commission_model="fixed",
            commission_value=8.0,
        ),
    ]

    reports = run_cost_scenarios(
        engine=engine,
        data=data,
        strategy=strategy,
        params={},
        scenarios=scenarios,
    )

    low_ret = reports["low"].performance["total_return"]
    high_ret = reports["high"].performance["total_return"]
    assert high_ret <= low_ret


def test_run_cost_scenarios_raises_for_empty_scenarios():
    engine = _build_engine()
    data = _build_single_asset_data()
    strategy = CountingTrendStrategy()

    with pytest.raises(ValueError, match="scenarios 不能为空"):
        run_cost_scenarios(
            engine=engine,
            data=data,
            strategy=strategy,
            params={},
            scenarios=[],
        )


def test_run_cost_scenarios_raises_for_duplicate_names():
    engine = _build_engine()
    data = _build_single_asset_data()
    strategy = CountingTrendStrategy()
    scenarios = [
        CostScenario(
            name="dup",
            slippage_model="fixed",
            slippage_value=0.0,
            commission_model="fixed",
            commission_value=0.0,
        ),
        CostScenario(
            name="dup",
            slippage_model="percent",
            slippage_value=0.001,
            commission_model="percent",
            commission_value=0.001,
        ),
    ]

    with pytest.raises(ValueError, match="scenarios.name 不能重复"):
        run_cost_scenarios(
            engine=engine,
            data=data,
            strategy=strategy,
            params={},
            scenarios=scenarios,
        )
