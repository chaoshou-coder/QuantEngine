from __future__ import annotations

from quantengine.config import QuantEngineConfig
from quantengine.engine.backtest import BacktestEngine
from quantengine.engine.commission import build_commission
from quantengine.engine.rules import TradingRules
from quantengine.engine.slippage import build_slippage


def build_engine(config: QuantEngineConfig) -> BacktestEngine:
    slippage = build_slippage(
        model=config.slippage.model,
        value=config.slippage.value,
        impact=config.slippage.impact,
    )
    commission = build_commission(
        model=config.commission.model,
        value=config.commission.value,
        tiers=config.commission.tiers,
    )
    rules = TradingRules(
        margin_ratio=config.rules.margin_ratio,
        limit_up_ratio=config.rules.limit_up_ratio,
        limit_down_ratio=config.rules.limit_down_ratio,
    )
    return BacktestEngine(
        slippage=slippage,
        commission=commission,
        rules=rules,
        initial_cash=config.runtime.initial_cash,
        contract_multiplier=config.runtime.contract_multiplier,
        risk_free_rate=config.runtime.risk_free_rate,
        periods_per_year=config.runtime.periods_per_year,
    )

