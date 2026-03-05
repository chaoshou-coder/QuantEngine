from __future__ import annotations

import numpy as np

from quantengine.engine.backtest import BacktestReport
from quantengine.engine.portfolio import PortfolioResult
from quantengine.metrics.performance import (
    annualized_return,
    calculate_performance_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from quantengine.metrics.risk import conditional_value_at_risk, value_at_risk
from quantengine.metrics.stability import benjamini_hochberg_correction, parameter_sensitivity_analysis
from quantengine.metrics.trade_analysis import calculate_trade_metrics
from quantengine.visualization.reports import ReportConfig, write_backtest_report


def test_max_drawdown_and_performance_metrics():
    equity = [100.0, 120.0, 90.0, 110.0]
    returns = [0.0, 0.2, -0.25, 0.111111]
    perf = calculate_performance_metrics(returns, equity, risk_free_rate=0.0, periods_per_year=252)
    assert max_drawdown(equity) == -0.25
    assert perf["annualized_return"] != 0.0
    assert "sharpe" in perf
    assert "sortino" in perf


def test_annualized_return_guard_for_short_series():
    assert annualized_return([100.0], periods_per_year=252) == 0.0


def test_sharpe_and_sortino_zero_std():
    returns = [0.01, 0.01, 0.01]
    assert sharpe_ratio(returns, risk_free_rate=0.01, periods_per_year=252) == 0.0
    assert sortino_ratio(returns, risk_free_rate=0.01, periods_per_year=252) == 0.0


def test_risk_metrics_var_cvar():
    returns = [-0.1, -0.05, 0.02, 0.03, -0.02]
    assert value_at_risk(returns, alpha=0.2) < 0.0
    assert conditional_value_at_risk(returns, alpha=0.2) <= value_at_risk(returns, alpha=0.2)


def test_trade_metrics_has_deterministic_symbol_string():
    trades = [
        {"side": "BUY", "quantity": 1.0, "price": 100.0, "cost": 10.0, "symbol": "AAPL"},
        {"side": "SELL", "quantity": 1.0, "price": 105.0, "cost": 10.0, "symbol": "AAPL"},
        {"side": "BUY", "quantity": 2.0, "price": 50.0, "cost": 5.0, "symbol": "MSFT"},
        {"side": "SELL", "quantity": 2.0, "price": 48.0, "cost": 5.0, "symbol": "MSFT"},
    ]
    metrics = calculate_trade_metrics(trades)
    assert metrics["trade_count"] == 2  # 2 round-trip trades realized
    assert metrics["symbol_count"] == 2
    assert metrics["most_active_symbol"] in ("AAPL", "MSFT")  # tied at 2 each


def test_benjamini_hochberg_correction_marks_expected_hypotheses():
    result = benjamini_hochberg_correction([0.001, 0.01, 0.04, 0.2], alpha=0.05)
    np.testing.assert_allclose(result.adjusted_p_values, [0.004, 0.02, 0.05333333333333334, 0.2], atol=1e-12)
    assert result.rejected == [True, True, False, False]
    assert result.significant_count == 2
    assert "Benjamini-Hochberg" in result.summary


def test_parameter_sensitivity_analysis_uses_plus_minus_ten_percent():
    base_params = {"fast": 10.0, "slow": 20, "name": "demo"}

    def evaluator(params: dict[str, object]) -> float:
        return float(params["fast"]) + 2.0 * float(params["slow"])

    result = parameter_sensitivity_analysis(base_params=base_params, evaluate_fn=evaluator, perturbation=0.1)
    by_name = {item.param_name: item for item in result}
    assert set(by_name.keys()) == {"fast", "slow"}
    assert by_name["fast"].minus_value == 9.0
    assert by_name["fast"].plus_value == 11.0
    assert by_name["slow"].minus_value == 18.0
    assert by_name["slow"].plus_value == 22.0
    assert by_name["fast"].base_score == 50.0
    assert by_name["slow"].base_score == 50.0


def test_write_backtest_report_supports_zh_template_and_pdf(tmp_path, monkeypatch):
    portfolio = PortfolioResult(
        equity_curve=np.array([100.0, 101.0, 103.0], dtype=float),
        returns=np.array([0.0, 0.01, 0.019801980198019802], dtype=float),
        positions=np.zeros((3, 1), dtype=float),
        turnover=np.zeros(3, dtype=float),
        trades=[{"side": "BUY", "quantity": 1.0, "price": 100.0, "cost": 0.1, "symbol": "XAUUSD"}],
        metadata={},
    )
    report = BacktestReport(
        strategy="demo",
        params={"fast": 10, "slow": 20},
        portfolio=portfolio,
        performance={"sharpe": 1.23, "win_rate": 0.55},
        risk={"max_drawdown": -0.1},
        trade_metrics={"trade_count": 1, "profit_factor": 1.8},
        metadata={},
    )

    class _FakeHtml:
        def __init__(self, string: str, base_url: str):
            self.string = string
            self.base_url = base_url

        def write_pdf(self, path: str) -> None:
            from pathlib import Path

            Path(path).write_bytes(b"%PDF-FAKE")

    monkeypatch.setattr("quantengine.visualization.reports._load_weasyprint_html", lambda: _FakeHtml)
    config = ReportConfig(
        include_trade_metrics=False,
        include_equity_plot=False,
        include_multiple_testing=True,
        include_parameter_sensitivity=True,
        export_pdf=True,
    )
    mt_result = benjamini_hochberg_correction([0.01, 0.03, 0.2], alpha=0.05)
    sensitivity = parameter_sensitivity_analysis(
        base_params={"fast": 10.0},
        evaluate_fn=lambda params: float(params["fast"]),
        perturbation=0.1,
    )
    outputs = write_backtest_report(
        report=report,
        timestamps=["2024-01-01", "2024-01-02", "2024-01-03"],
        output_path=tmp_path / "report.html",
        config=config,
        multiple_testing=mt_result,
        parameter_sensitivity=sensitivity,
    )

    html_text = outputs["html"].read_text(encoding="utf-8")
    assert "回测报告" in html_text
    assert "多重检验校正" in html_text
    assert "参数敏感度" in html_text
    assert "交易指标" not in html_text
    assert outputs["pdf"] is not None
    assert outputs["pdf"].exists()
