from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from quantengine.data.gpu_backend import xp_from_array

from quantengine.data.loader import DataBundle
from quantengine.data.gpu_backend import to_numpy
from quantengine.metrics.performance import calculate_performance_metrics
from quantengine.metrics.risk import calculate_risk_metrics
from quantengine.metrics.trade_analysis import calculate_trade_metrics
from quantengine.strategy.base import BaseStrategy

from .commission import CommissionModel
from .portfolio import PortfolioResult, simulate_portfolio
from .rules import TradingRules
from .slippage import SlippageModel


@dataclass
class BacktestReport:
    strategy: str
    params: dict[str, Any]
    portfolio: PortfolioResult
    performance: dict[str, float]
    risk: dict[str, float]
    trade_metrics: dict[str, float | str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def objective(self, metric: str) -> float:
        if metric in self.performance:
            return float(self.performance[metric])
        if metric in self.risk:
            return float(self.risk[metric])
        if metric in self.trade_metrics:
            value = self.trade_metrics[metric]
            if not isinstance(value, (int, float)):
                raise TypeError(f"指标 {metric} 非数值类型，无法作为目标: {type(value).__name__}")
            return float(value)
        raise KeyError(f"未知指标: {metric}")


def _to_isoformat(ts: Any) -> str:
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


class BacktestEngine:
    def __init__(
        self,
        slippage: SlippageModel,
        commission: CommissionModel,
        rules: TradingRules | None = None,
        initial_cash: float = 1_000_000.0,
        contract_multiplier: float = 1.0,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252 * 390,
    ):
        self.slippage = slippage
        self.commission = commission
        self.rules = rules
        self.initial_cash = initial_cash
        self.contract_multiplier = contract_multiplier
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run(
        self,
        data: DataBundle,
        strategy: BaseStrategy,
        params: dict[str, Any],
        record_trades: bool = True,
    ) -> BacktestReport:
        strategy.on_init(self)
        signal = strategy.generate_signals(data, params)
        portfolio = simulate_portfolio(
            data=data,
            signal=signal,
            slippage=self.slippage,
            commission=self.commission,
            rules=self.rules,
            initial_cash=self.initial_cash,
            contract_multiplier=self.contract_multiplier,
            record_trades=record_trades,
        )
        performance = calculate_performance_metrics(
            returns=portfolio.returns,
            equity_curve=portfolio.equity_curve,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
        )
        risk = calculate_risk_metrics(returns=portfolio.returns, equity_curve=portfolio.equity_curve)
        trade_metrics = calculate_trade_metrics(portfolio.trades)
        return BacktestReport(
            strategy=strategy.name,
            params=params,
            portfolio=portfolio,
            performance=performance,
            risk=risk,
            trade_metrics=trade_metrics,
            metadata={
                "symbols": data.symbols,
                "backend": data.backend.active,
                "timestamps": [_to_isoformat(ts) for ts in data.timestamps],
            },
        )

    def run_multi_strategy(
        self,
        data: DataBundle,
        strategies: list[tuple[BaseStrategy, dict[str, Any], float]],
        record_trades: bool = True,
    ) -> BacktestReport:
        if not strategies:
            raise ValueError("strategies 不能为空")
        aggregate = None
        total_weight = 0.0
        names: list[str] = []
        for strategy, params, weight in strategies:
            strategy.on_init(self)
            signal = strategy.generate_signals(data, params)
            if aggregate is None:
                aggregate = signal * weight
            else:
                aggregate = aggregate + signal * weight
            total_weight += weight
            names.append(strategy.name)
        if aggregate is None:
            raise ValueError("无法生成聚合信号")

        xp = xp_from_array(aggregate)
        # 避免总权重为 0，且将组合信号约束在 [-1, 1]
        scaled = xp.clip(aggregate / (total_weight if abs(total_weight) > 1e-12 else 1.0), -1.0, 1.0)

        portfolio = simulate_portfolio(
            data=data,
            signal=scaled,
            slippage=self.slippage,
            commission=self.commission,
            rules=self.rules,
            initial_cash=self.initial_cash,
            contract_multiplier=self.contract_multiplier,
            record_trades=record_trades,
        )
        performance = calculate_performance_metrics(
            returns=portfolio.returns,
            equity_curve=portfolio.equity_curve,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
        )
        risk = calculate_risk_metrics(returns=portfolio.returns, equity_curve=portfolio.equity_curve)
        trade_metrics = calculate_trade_metrics(portfolio.trades)
        return BacktestReport(
            strategy="+".join(names),
            params={"portfolio": "multi_strategy"},
            portfolio=portfolio,
            performance=performance,
            risk=risk,
            trade_metrics=trade_metrics,
            metadata={
                "symbols": data.symbols,
                "backend": data.backend.active,
                "weights": [item[2] for item in strategies],
                "timestamps": [_to_isoformat(ts) for ts in data.timestamps],
            },
        )
