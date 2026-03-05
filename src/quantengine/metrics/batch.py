from __future__ import annotations

import numpy as np

from quantengine.data.gpu_backend import to_numpy, xp_from_array
from quantengine.metrics.performance import calculate_performance_metrics
from quantengine.metrics.risk import calculate_risk_metrics


def _ensure_2d(values: np.ndarray):
    xp = xp_from_array(values)
    arr = xp.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"batch 输入需为二维数组，当前形状={arr.shape}")
    return arr


def _vector_sharpe(returns: np.ndarray, risk_free_rate: float, periods_per_year: int) -> np.ndarray:
    xp = xp_from_array(returns)
    rf_period = risk_free_rate / periods_per_year
    excess = returns - rf_period
    std = xp.std(excess, axis=0, ddof=1)
    mean_excess = xp.mean(excess, axis=0)
    denom = xp.where(std > 1e-12, std, 1.0)
    out = xp.sqrt(periods_per_year) * mean_excess / denom
    out = xp.where(std > 1e-12, out, 0.0)
    return to_numpy(out).astype(float)


def _vector_sortino(returns: np.ndarray, risk_free_rate: float, periods_per_year: int) -> np.ndarray:
    xp = xp_from_array(returns)
    rf_period = risk_free_rate / periods_per_year
    excess = returns - rf_period
    mask = excess < 0
    mean_excess = xp.mean(excess, axis=0)

    n_neg = xp.sum(mask, axis=0)
    denom_count = xp.where(n_neg > 0, n_neg, 1.0)
    downside_mean = xp.sum(xp.where(mask, excess, 0.0), axis=0) / denom_count
    downside_var = xp.sum(
        xp.where(mask, (excess - downside_mean) ** 2, 0.0),
        axis=0,
    ) / xp.where(n_neg > 1, (n_neg - 1.0), 1.0)
    downside_std = xp.sqrt(downside_var)

    out = xp.sqrt(periods_per_year) * mean_excess / xp.where(downside_std > 1e-12, downside_std, 1.0)
    out = xp.where((n_neg > 1) & (downside_std > 1e-12), out, 0.0)
    return to_numpy(out).astype(float)


def _vector_annualized_return(equity: np.ndarray, periods_per_year: int) -> np.ndarray:
    xp = xp_from_array(equity)
    n_bars = equity.shape[0]
    first = equity[0]
    last = equity[-1]
    years = max((n_bars - 1) / periods_per_year, 1e-9)
    return to_numpy(
        xp.where(
            first > 0.0,
            xp.power(last / xp.where(first > 0.0, first, 1.0), 1.0 / years) - 1.0,
            0.0,
        )
    ).astype(float)


def _vector_win_rate(returns: np.ndarray) -> np.ndarray:
    xp = xp_from_array(returns)
    if returns.shape[0] <= 1:
        return np.zeros(returns.shape[1], dtype=float)
    series = returns[1:]  # skip initial zero-return bar
    wins = xp.sum(series > 0.0, axis=0)
    losses = xp.sum(series < 0.0, axis=0)
    total = wins + losses
    out = xp.where(total > 0, wins / xp.where(total > 0, total, 1.0), 0.0)
    return to_numpy(out).astype(float)


def _vector_beta(returns: np.ndarray, market_returns: np.ndarray | None) -> np.ndarray:
    if market_returns is None:
        return np.zeros(returns.shape[1], dtype=float)

    xp = xp_from_array(returns)
    mkt = xp.asarray(market_returns, dtype=float).reshape(-1)
    if mkt.shape[0] != returns.shape[0]:
        n = min(int(mkt.shape[0]), int(returns.shape[0]))
        if n <= 1:
            return np.zeros(returns.shape[1], dtype=float)
        mkt = mkt[:n]
        returns = returns[:n]

    mkt_mean = xp.mean(mkt)
    ret_mean = xp.mean(returns, axis=0)
    cov = xp.mean((returns - ret_mean[None, :]) * (mkt[:, None] - mkt_mean), axis=0)
    var = xp.var(mkt)
    out = xp.where(var > 1e-12, cov / var, 0.0)
    return to_numpy(out).astype(float)


def batch_score(
    equity_2d: np.ndarray,
    returns_2d: np.ndarray,
    metric: str,
    risk_free_rate: float,
    periods_per_year: int,
    market_returns: np.ndarray | None = None,
) -> np.ndarray:
    if returns_2d.shape != equity_2d.shape:
        raise ValueError(f"returns/equity 形状不一致: returns={returns_2d.shape}, equity={equity_2d.shape}")

    equity = _ensure_2d(equity_2d)
    returns = _ensure_2d(returns_2d)
    n_combo = equity.shape[1]
    if n_combo == 0:
        return np.zeros(0, dtype=float)

    metric_key = metric.strip().lower()
    if metric_key == "sharpe":
        return _vector_sharpe(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    if metric_key == "sortino":
        return _vector_sortino(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    if metric_key == "max_drawdown":
        xp = xp_from_array(equity)
        try:
            running_max = xp.maximum.accumulate(equity, axis=0)
        except NotImplementedError:
            eq_np = np.asarray(to_numpy(equity), dtype=np.float64)
            running_max = xp.asarray(np.maximum.accumulate(eq_np, axis=0))
        drawdown = (equity - running_max) / xp.maximum(running_max, 1e-12)
        return to_numpy(xp.min(drawdown, axis=0)).astype(float)
    if metric_key == "total_return":
        xp = xp_from_array(equity)
        first = equity[0]
        last = equity[-1]
        return to_numpy(
            xp.where(
                first > 0.0,
                last / xp.where(first > 0.0, first, 1.0) - 1.0,
                0.0,
            )
        ).astype(float)
    if metric_key == "annualized_return":
        return _vector_annualized_return(equity, periods_per_year)
    if metric_key == "calmar":
        max_dd = batch_score(
            equity_2d=equity,
            returns_2d=returns,
            metric="max_drawdown",
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            market_returns=market_returns,
        )
        ann = _vector_annualized_return(equity, periods_per_year)
        return np.where(np.abs(max_dd) > 1e-12, ann / np.abs(max_dd), 0.0)
    if metric_key == "win_rate":
        return _vector_win_rate(returns)
    if metric_key == "beta":
        return _vector_beta(returns, market_returns=market_returns)

    # 回退：保持兼容性，逐列调用现有单条计算逻辑
    scores = np.zeros(n_combo, dtype=float)
    for idx in range(n_combo):
        perf = calculate_performance_metrics(
            returns=to_numpy(returns[:, idx]),
            equity_curve=to_numpy(equity[:, idx]),
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            market_returns=market_returns,
        )
        if metric_key in perf:
            scores[idx] = float(perf[metric_key])
            continue

        risk = calculate_risk_metrics(
            returns=to_numpy(returns[:, idx]),
            equity_curve=to_numpy(equity[:, idx]),
        )
        if metric_key in risk:
            scores[idx] = float(risk[metric_key])
            continue

        raise KeyError(f"未知指标: {metric}")

    return scores
