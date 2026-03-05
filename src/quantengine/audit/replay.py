from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from quantengine.data.loader import DataBundle
from quantengine.engine.commission import build_commission
from quantengine.engine.rules import TradingRules
from quantengine.engine.slippage import build_slippage
from quantengine.strategy import get_strategy

from .io import load_audit_bundle, verify_audit_bundle

if TYPE_CHECKING:
    from quantengine.engine.backtest import BacktestReport


def _build_rules(payload: dict[str, Any]) -> TradingRules | None:
    if not payload:
        return None
    return TradingRules(**payload)


def replay_from_bundle(
    bundle_path: str | Path,
    *,
    data: DataBundle,
    strict_env: bool = True,
) -> BacktestReport:
    verification = verify_audit_bundle(bundle_path, data=data, strict=False)
    if not verification["integrity_ok"]:
        raise ValueError(f"审计包完整性校验失败: {verification}")
    if verification["data_hash_match"] is False:
        raise ValueError("数据哈希不一致")
    if strict_env and not verification["env_match"]:
        raise ValueError(f"环境不一致: {verification['env_mismatches']}")

    bundle = load_audit_bundle(bundle_path)
    strategy_cfg = dict(bundle.config.get("strategy", {}))
    engine_cfg = dict(bundle.config.get("engine", {}))
    run_cfg = dict(bundle.config.get("run", {}))

    strategy_name = str(strategy_cfg.get("name", "")).strip()
    if not strategy_name:
        raise ValueError("审计包缺少 strategy.name")
    params = dict(strategy_cfg.get("params", {}))

    slippage_cfg = dict(engine_cfg.get("slippage", {}))
    commission_cfg = dict(engine_cfg.get("commission", {}))
    rules_cfg = dict(engine_cfg.get("rules", {}))
    slippage = build_slippage(
        model=str(slippage_cfg.get("model", "percent")),
        value=float(slippage_cfg.get("value", 0.0)),
        impact=float(slippage_cfg.get("impact", 1.0)),
    )
    commission = build_commission(
        model=str(commission_cfg.get("model", "percent")),
        value=float(commission_cfg.get("value", 0.0)),
        tiers=commission_cfg.get("tiers"),
    )
    from quantengine.engine.backtest import BacktestEngine

    engine = BacktestEngine(
        slippage=slippage,
        commission=commission,
        rules=_build_rules(rules_cfg),
        initial_cash=float(engine_cfg.get("initial_cash", 1_000_000.0)),
        contract_multiplier=float(engine_cfg.get("contract_multiplier", 1.0)),
        risk_free_rate=float(engine_cfg.get("risk_free_rate", 0.02)),
        periods_per_year=int(engine_cfg.get("periods_per_year", 252 * 390)),
    )
    strategy = get_strategy(strategy_name)
    report = engine.run(
        data=data,
        strategy=strategy,
        params=params,
        record_trades=bool(run_cfg.get("record_trades", True)),
    )

    expected_equity = np.asarray(bundle.equity_curve, dtype=np.float64)
    expected_returns = np.asarray(bundle.returns, dtype=np.float64)
    actual_equity = np.asarray(report.portfolio.equity_curve, dtype=np.float64)
    actual_returns = np.asarray(report.portfolio.returns, dtype=np.float64)
    if not np.array_equal(actual_equity, expected_equity):
        raise ValueError("回放结果不一致: equity_curve")
    if not np.array_equal(actual_returns, expected_returns):
        raise ValueError("回放结果不一致: returns")
    if report.portfolio.trades != bundle.trade_log:
        raise ValueError("回放结果不一致: trades")
    if (report.portfolio.risk_events or []) != bundle.risk_events:
        raise ValueError("回放结果不一致: risk_events")
    return report
