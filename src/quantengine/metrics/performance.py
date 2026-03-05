from __future__ import annotations

import numpy as np

from quantengine.data.gpu_backend import to_numpy, xp_from_array


def max_drawdown(equity_curve: np.ndarray) -> float:
    xp = xp_from_array(equity_curve)
    equity = xp.asarray(equity_curve, dtype=float)
    running_max = xp.maximum.accumulate(equity)
    drawdown = (equity - running_max) / xp.maximum(running_max, 1e-12)
    return float(drawdown.min())


def annualized_return(equity_curve: np.ndarray, periods_per_year: int) -> float:
    xp = xp_from_array(equity_curve)
    equity = xp.asarray(equity_curve, dtype=float)
    if equity.size < 2 or equity[0] <= 0:
        return 0.0
    total = equity[-1] / equity[0]
    years = max((equity.size - 1) / periods_per_year, 1e-9)
    return float(total ** (1.0 / years) - 1.0)


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float, periods_per_year: int) -> float:
    xp = xp_from_array(returns)
    ret = xp.asarray(returns, dtype=float)
    if ret.size <= 1:
        return 0.0
    rf_period = risk_free_rate / periods_per_year
    excess = ret - rf_period
    std = xp.std(excess, ddof=1)
    if std <= 1e-12:
        return 0.0
    return float(xp.sqrt(periods_per_year) * xp.mean(excess) / std)


def sortino_ratio(returns: np.ndarray, risk_free_rate: float, periods_per_year: int) -> float:
    xp = xp_from_array(returns)
    ret = xp.asarray(returns, dtype=float)
    if ret.size <= 1:
        return 0.0
    rf_period = risk_free_rate / periods_per_year
    excess = ret - rf_period
    downside = excess[excess < 0]
    downside_std = xp.std(downside, ddof=1) if downside.size > 1 else 0.0
    if downside_std <= 1e-12:
        return 0.0
    return float(xp.sqrt(periods_per_year) * xp.mean(excess) / downside_std)


def market_returns_from_close(close: np.ndarray) -> np.ndarray:
    """Build market return series from close prices.

    - 1D close: direct close-to-close returns
    - 2D close: equal-weight average return across assets
    """
    close_np = np.asarray(to_numpy(close), dtype=np.float64)
    if close_np.ndim == 1:
        market = np.zeros_like(close_np, dtype=np.float64)
        if close_np.size > 1:
            prev = np.where(close_np[:-1] != 0.0, close_np[:-1], 1.0)
            market[1:] = (close_np[1:] - close_np[:-1]) / prev
        return market

    n_bars, n_assets = close_np.shape
    if n_assets == 0:
        return np.zeros(n_bars, dtype=np.float64)
    asset_ret = np.zeros_like(close_np, dtype=np.float64)
    if n_bars > 1:
        prev = np.where(close_np[:-1] != 0.0, close_np[:-1], 1.0)
        asset_ret[1:] = (close_np[1:] - close_np[:-1]) / prev
    return asset_ret.mean(axis=1)


def beta_to_market(returns: np.ndarray, market_returns: np.ndarray) -> float:
    xp = xp_from_array(returns)
    ret = xp.asarray(returns, dtype=float).reshape(-1)
    mkt = xp.asarray(market_returns, dtype=float).reshape(-1)
    n = int(min(ret.shape[0], mkt.shape[0]))
    if n <= 1:
        return 0.0

    ret = ret[:n]
    mkt = mkt[:n]
    ret_mean = xp.mean(ret)
    mkt_mean = xp.mean(mkt)
    cov = xp.mean((ret - ret_mean) * (mkt - mkt_mean))
    var = xp.var(mkt)
    if var <= 1e-12:
        return 0.0
    return float(cov / var)


def bar_win_rate(returns: np.ndarray) -> float:
    xp = xp_from_array(returns)
    ret = xp.asarray(returns, dtype=float).reshape(-1)
    if ret.size <= 1:
        return 0.0
    series = ret[1:]  # skip first bar (always baseline 0)
    wins = xp.sum(series > 0.0)
    losses = xp.sum(series < 0.0)
    total = wins + losses
    if total <= 0:
        return 0.0
    return float(wins / total)


def calculate_performance_metrics(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252 * 390,
    market_returns: np.ndarray | None = None,
) -> dict[str, float]:
    ret_xp = xp_from_array(returns)
    eq_xp = xp_from_array(equity_curve)
    ret = ret_xp.asarray(returns, dtype=float)
    equity = eq_xp.asarray(equity_curve, dtype=float)
    mdd = max_drawdown(equity)
    ann_ret = annualized_return(equity, periods_per_year=periods_per_year)
    sharpe = sharpe_ratio(ret, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    sortino = sortino_ratio(ret, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    vol = float(ret_xp.std(ret, ddof=1) * ret_xp.sqrt(periods_per_year)) if ret.size > 1 else 0.0
    calmar = float(ann_ret / abs(mdd)) if abs(mdd) > 1e-12 else 0.0
    total_return = float((equity[-1] / equity[0] - 1.0) if equity.size > 1 and equity[0] > 0 else 0.0)
    beta = beta_to_market(ret, market_returns) if market_returns is not None else 0.0
    win_rate = bar_win_rate(ret)
    return {
        "total_return": total_return,
        "annualized_return": ann_ret,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "beta": beta,
        "win_rate": win_rate,
    }
