from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantengine.audit.bundle import AuditBundle, build_audit_bundle
from quantengine.data.gpu_backend import to_numpy, xp_from_array
from quantengine.data.loader import DataBundle
from quantengine.metrics.performance import calculate_performance_metrics, market_returns_from_close
from quantengine.metrics.risk import calculate_risk_metrics
from quantengine.metrics.trade_analysis import calculate_trade_metrics
from quantengine.strategy.base import BaseStrategy

from .commission import CommissionModel, build_commission
from .portfolio import PortfolioResult, simulate_portfolio, simulate_portfolio_batch
from .rules import TradingRules
from .slippage import SlippageModel, build_slippage


@dataclass
class BacktestReport:
    strategy: str
    params: dict[str, Any]
    portfolio: PortfolioResult
    performance: dict[str, float]
    risk: dict[str, float]
    trade_metrics: dict[str, float | str]
    metadata: dict[str, Any] = field(default_factory=dict)
    audit_bundle: AuditBundle | None = None

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


@dataclass
class CostScenario:
    name: str
    slippage_model: str
    slippage_value: float
    commission_model: str
    commission_value: float
    slippage_impact: float = 0.0
    commission_tiers: list | None = None


def _to_isoformat(ts: Any) -> str:
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def _validate_cost_scenarios(scenarios: list[CostScenario]) -> None:
    if not scenarios:
        raise ValueError("scenarios 不能为空")
    names = [item.name for item in scenarios]
    if any(not name.strip() for name in names):
        raise ValueError("scenarios.name 不能为空")
    if len(set(names)) != len(names):
        raise ValueError("scenarios.name 不能重复")


def _portfolio_from_batch(
    data: DataBundle,
    signal: Any,
    equity_curve: Any,
    returns: Any,
    initial_cash: float,
    contract_multiplier: float,
    scenario: CostScenario,
) -> PortfolioResult:
    signal_np = np.asarray(to_numpy(signal), dtype=float)
    if signal_np.ndim == 1:
        signal_np = signal_np.reshape(-1, 1)

    n_bars, n_assets = data.close.shape
    positions = np.zeros((n_bars, n_assets), dtype=float)
    if n_bars > 1:
        positions[1:] = np.clip(signal_np[:-1], -1.0, 1.0)

    turnover = np.zeros(n_bars, dtype=float)
    if n_bars > 1:
        turnover[1:] = np.sum(np.abs(positions[1:] - positions[:-1]), axis=1)

    equity_np = np.asarray(to_numpy(equity_curve), dtype=float)
    returns_np = np.asarray(to_numpy(returns), dtype=float)
    bankrupt_mask = equity_np <= 0.0
    bankruptcy_index = int(np.where(bankrupt_mask)[0][0]) if np.any(bankrupt_mask) else None
    return PortfolioResult(
        equity_curve=equity_np,
        returns=returns_np,
        positions=positions,
        turnover=turnover,
        trades=[],
        metadata={
            "symbols": data.symbols,
            "backend": data.backend.active,
            "initial_cash": initial_cash,
            "contract_multiplier": contract_multiplier,
            "bankrupt": bool(np.any(bankrupt_mask)),
            "bankruptcy_index": bankruptcy_index,
            "cost_scenario": {
                "name": scenario.name,
                "slippage_model": scenario.slippage_model,
                "slippage_value": scenario.slippage_value,
                "slippage_impact": scenario.slippage_impact,
                "commission_model": scenario.commission_model,
                "commission_value": scenario.commission_value,
                "commission_tiers": scenario.commission_tiers,
            },
        },
    )


def run_cost_scenarios(
    engine: BacktestEngine,
    data: DataBundle,
    strategy: BaseStrategy,
    params: dict[str, Any],
    scenarios: list[CostScenario],
) -> dict[str, BacktestReport]:
    """对同一策略+参数，用不同成本情景分别回测。

    信号只生成一次，仅成本参数不同。
    返回 {scenario.name: BacktestReport}。
    """
    _validate_cost_scenarios(scenarios)

    strategy.on_init(engine)
    signal = strategy.generate_signals(data, params)
    market_returns = market_returns_from_close(data.close)
    reports: dict[str, BacktestReport] = {}

    # 复用同一份信号，逐场景调用 batch 仿真（当前 batch API 仅支持单套成本参数）。
    for scenario in scenarios:
        slippage = build_slippage(
            model=scenario.slippage_model,
            value=scenario.slippage_value,
            impact=scenario.slippage_impact,
        )
        commission = build_commission(
            model=scenario.commission_model,
            value=scenario.commission_value,
            tiers=scenario.commission_tiers,
        )

        equity_batch, returns_batch = simulate_portfolio_batch(
            data=data,
            signal=signal,
            slippage=slippage,
            commission=commission,
            rules=engine.rules,
            initial_cash=engine.initial_cash,
            contract_multiplier=engine.contract_multiplier,
        )
        portfolio = _portfolio_from_batch(
            data=data,
            signal=signal,
            equity_curve=equity_batch[:, 0],
            returns=returns_batch[:, 0],
            initial_cash=engine.initial_cash,
            contract_multiplier=engine.contract_multiplier,
            scenario=scenario,
        )
        performance = calculate_performance_metrics(
            returns=portfolio.returns,
            equity_curve=portfolio.equity_curve,
            risk_free_rate=engine.risk_free_rate,
            periods_per_year=engine.periods_per_year,
            market_returns=market_returns,
        )
        risk = calculate_risk_metrics(returns=portfolio.returns, equity_curve=portfolio.equity_curve)
        trade_metrics = calculate_trade_metrics(portfolio.trades)
        reports[scenario.name] = BacktestReport(
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
                "cost_scenario": portfolio.metadata["cost_scenario"],
            },
        )

    return reports


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
            market_returns=market_returns_from_close(data.close),
        )
        risk = calculate_risk_metrics(returns=portfolio.returns, equity_curve=portfolio.equity_curve)
        trade_metrics = calculate_trade_metrics(portfolio.trades)
        audit_bundle = build_audit_bundle(
            data=data,
            strategy_name=strategy.name,
            params=params,
            slippage=self.slippage,
            commission=self.commission,
            rules=self.rules,
            initial_cash=self.initial_cash,
            contract_multiplier=self.contract_multiplier,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            record_trades=record_trades,
            portfolio=portfolio,
            performance=performance,
            risk=risk,
            trade_metrics=trade_metrics,
        )
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
            audit_bundle=audit_bundle,
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
            market_returns=market_returns_from_close(data.close),
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
