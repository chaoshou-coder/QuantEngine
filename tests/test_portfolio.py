from __future__ import annotations

import numpy as np

from quantengine.data.loader import DataBundle
from quantengine.data.gpu_backend import BackendInfo
from quantengine.engine.portfolio import simulate_portfolio
from quantengine.engine.slippage import FixedSlippage
from quantengine.engine.commission import FixedCommission
from quantengine.engine.rules import TradingRules
from quantengine.strategy.base import BaseStrategy


class FlatBuyStrategy(BaseStrategy):
    name = "flat_buy"

    def parameters(self):
        return {}

    def generate_signals(self, data: DataBundle, params: dict):
        return np.array([[0.0], [1.0], [1.0]])


def test_simulate_portfolio_pnl_uses_current_position(sample_data_bundle):
    data = DataBundle(
        symbols=["AAA"],
        timestamps=sample_data_bundle.timestamps,
        open=np.array([[100.0], [100.0], [102.0]]),
        high=np.array([[101.0], [103.0], [105.0]]),
        low=np.array([[99.0], [99.0], [101.0]]),
        close=np.array([[100.0], [102.0], [104.0]]),
        volume=np.array([[1000.0], [1000.0], [1000.0]]),
        backend=BackendInfo("auto", "cpu", "test", False, False, False),
    )
    result = simulate_portfolio(
        data=data,
        signal=FlatBuyStrategy().generate_signals(data, {}),
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        rules=TradingRules(margin_ratio=0.0),
        initial_cash=1000.0,
        contract_multiplier=1.0,
        record_trades=False,
    )
    # signal=[0,1,1] → positions: bar0=0, bar1=signal[0]=0, bar2=signal[1]=1
    # bar2 PnL: prev_pos=0*(open2-close1) + curr_pos=1*(close2-open2) = 1*(104-102) = 2
    assert result.equity_curve[0] == 1000.0
    assert result.equity_curve[-1] == 1002.0


def test_portfolio_metadata_records_bankrupt_state(sample_data_bundle):
    data = DataBundle(
        symbols=["AAA"],
        timestamps=sample_data_bundle.timestamps,
        open=np.array([[100.0], [1.0], [1.0]]),
        high=np.array([[100.0], [1.0], [1.0]]),
        low=np.array([[99.0], [1.0], [1.0]]),
        close=np.array([[100.0], [1.0], [1.0]]),
        volume=np.array([[1000.0], [1000.0], [1000.0]]),
        backend=BackendInfo("auto", "cpu", "test", False, False, False),
    )
    result = simulate_portfolio(
        data=data,
        signal=np.array([[0.0], [1.0], [1.0]]),
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        rules=TradingRules(margin_ratio=2.0),
        initial_cash=1.0,
        contract_multiplier=1.0,
        record_trades=False,
    )
    assert "bankrupt" in result.metadata
