from __future__ import annotations

import numpy as np

from quantengine.data.gpu_backend import xp_from_array


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    xp = xp_from_array(returns)
    ret = xp.asarray(returns, dtype=float)
    if ret.size == 0:
        return 0.0
    return float(xp.quantile(ret, alpha))


def conditional_value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    xp = xp_from_array(returns)
    ret = xp.asarray(returns, dtype=float)
    if ret.size == 0:
        return 0.0
    var = value_at_risk(ret, alpha=alpha)
    tail = ret[ret <= var]
    if tail.size == 0:
        return 0.0
    return float(xp.mean(tail))


def ulcer_index(equity_curve: np.ndarray) -> float:
    xp = xp_from_array(equity_curve)
    equity = xp.asarray(equity_curve, dtype=float)
    if equity.size == 0:
        return 0.0
    running_max = xp.maximum.accumulate(equity)
    drawdown_pct = (equity - running_max) / xp.maximum(running_max, 1e-12) * 100.0
    return float(xp.sqrt(xp.mean(xp.square(drawdown_pct))))


def calculate_risk_metrics(returns: np.ndarray, equity_curve: np.ndarray) -> dict[str, float]:
    var_95 = value_at_risk(returns, alpha=0.05)
    cvar_95 = conditional_value_at_risk(returns, alpha=0.05)
    ui = ulcer_index(equity_curve)
    downside = xp_from_array(returns).asarray(returns, dtype=float)
    downside = downside[downside < 0]
    downside_dev = float(xp_from_array(downside).std(downside, ddof=1)) if downside.size > 1 else 0.0
    return {
        "var_95": var_95,
        "cvar_95": cvar_95,
        "ulcer_index": ui,
        "downside_deviation": downside_dev,
    }
