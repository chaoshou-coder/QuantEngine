from __future__ import annotations

import numpy as np
import pytest

from quantengine.metrics.performance import (
    annualized_return,
    bar_win_rate,
    beta_to_market,
    calculate_performance_metrics,
    market_returns_from_close,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def test_max_drawdown_matches_manual_result() -> None:
    equity = np.array([100.0, 120.0, 90.0, 110.0], dtype=float)
    # running max = [100, 120, 120, 120], worst drawdown = (90-120)/120 = -0.25
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_for_flat_curve_is_zero() -> None:
    equity = np.array([100.0, 100.0, 100.0], dtype=float)
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_annualized_return_matches_manual_formula() -> None:
    equity = np.array([100.0, 110.0], dtype=float)
    periods_per_year = 252
    expected = 1.1**252 - 1.0
    assert annualized_return(equity, periods_per_year=periods_per_year) == pytest.approx(expected)


def test_annualized_return_short_series_guard() -> None:
    assert annualized_return(np.array([100.0]), periods_per_year=252) == 0.0


def test_sharpe_ratio_matches_manual_calculation() -> None:
    returns = np.array([0.0, 0.02, -0.01, 0.03], dtype=float)
    rf = 0.0
    periods_per_year = 252
    excess = returns
    expected = np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1)
    actual = sharpe_ratio(returns, risk_free_rate=rf, periods_per_year=periods_per_year)
    assert actual == pytest.approx(expected)


def test_sortino_ratio_matches_manual_calculation() -> None:
    returns = np.array([0.0, 0.02, -0.01, -0.03, 0.01], dtype=float)
    rf = 0.0
    periods_per_year = 252
    excess = returns
    downside = excess[excess < 0]
    expected = np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=1)
    actual = sortino_ratio(returns, risk_free_rate=rf, periods_per_year=periods_per_year)
    assert actual == pytest.approx(expected)


def test_sortino_ratio_returns_zero_when_downside_std_not_available() -> None:
    returns = np.array([0.0, 0.01, -0.005, 0.02], dtype=float)  # only one downside sample
    assert sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252) == 0.0


def test_market_returns_from_close_1d() -> None:
    close = np.array([100.0, 110.0, 99.0], dtype=float)
    actual = market_returns_from_close(close)
    expected = np.array([0.0, 0.1, -0.1], dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_market_returns_from_close_2d_equal_weight() -> None:
    close = np.array(
        [
            [100.0, 200.0],
            [110.0, 220.0],  # each asset +10%
            [121.0, 198.0],  # +10% and -10%, mean = 0
        ],
        dtype=float,
    )
    actual = market_returns_from_close(close)
    expected = np.array([0.0, 0.1, 0.0], dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_beta_to_market_matches_manual_covariance_over_variance() -> None:
    returns = np.array([0.0, 0.01, 0.02, -0.01], dtype=float)
    market = np.array([0.0, 0.02, 0.01, -0.02], dtype=float)
    ret = returns[: market.shape[0]]
    mkt = market
    cov = np.mean((ret - ret.mean()) * (mkt - mkt.mean()))
    var = np.var(mkt)
    expected = cov / var
    actual = beta_to_market(returns, market)
    assert actual == pytest.approx(expected)


def test_beta_to_market_returns_zero_when_market_variance_zero() -> None:
    returns = np.array([0.0, 0.01, -0.01], dtype=float)
    market = np.array([0.0, 0.0, 0.0], dtype=float)
    assert beta_to_market(returns, market) == 0.0


def test_bar_win_rate_skips_first_bar() -> None:
    returns = np.array([0.0, 0.1, -0.2, 0.3, 0.0], dtype=float)
    # considered bars: [0.1, -0.2, 0.3, 0.0] -> wins=2, losses=1
    assert bar_win_rate(returns) == pytest.approx(2.0 / 3.0)


def test_calculate_performance_metrics_core_fields_match_helpers() -> None:
    returns = np.array([0.0, 0.02, -0.01, 0.03], dtype=float)
    equity = np.array([100.0, 102.0, 100.98, 104.0094], dtype=float)
    market = np.array([0.0, 0.01, -0.02, 0.02], dtype=float)
    metrics = calculate_performance_metrics(
        returns=returns,
        equity_curve=equity,
        risk_free_rate=0.0,
        periods_per_year=252,
        market_returns=market,
    )
    assert metrics["max_drawdown"] == pytest.approx(max_drawdown(equity))
    assert metrics["sharpe"] == pytest.approx(sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252))
    assert metrics["sortino"] == pytest.approx(sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252))
    assert metrics["win_rate"] == pytest.approx(bar_win_rate(returns))
    assert metrics["beta"] == pytest.approx(beta_to_market(returns, market))
