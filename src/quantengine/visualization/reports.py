from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from quantengine.engine.backtest import BacktestReport
from quantengine.optimizer.base import OptimizationResult
from quantengine.optimizer.walk_forward import WalkForwardResult

from .plots import build_equity_figure


def _dict_to_table(data: dict[str, object]) -> str:
    rows = []
    for key, value in data.items():
        safe_key = html.escape(str(key))
        safe_value = html.escape(str(value))
        rows.append(f"<tr><td>{safe_key}</td><td>{safe_value}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def write_backtest_report_html(
    report: BacktestReport,
    timestamps: Sequence,
    output_path: str | Path,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    figure_html = "<p>plotly not available</p>"
    try:
        fig = build_equity_figure(
            timestamps=timestamps,
            equity_curve=report.portfolio.equity_curve,
            title=f"Equity - {report.strategy}",
        )
        figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception:
        pass

    meta = {
        "strategy": report.strategy,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trades": len(report.portfolio.trades),
    }
    report_html = f"""
<html>
<head>
  <meta charset="utf-8" />
  <title>QuantEngine Backtest Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 16px; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>Backtest Report</h1>
  <h2>Meta</h2>
  {_dict_to_table(meta)}
  <h2>Params</h2>
  {_dict_to_table(report.params)}
  <h2>Performance</h2>
  {_dict_to_table(report.performance)}
  <h2>Risk</h2>
  {_dict_to_table(report.risk)}
  <h2>Trade Metrics</h2>
  {_dict_to_table(report.trade_metrics)}
  <h2>Equity & Drawdown</h2>
  {figure_html}
</body>
</html>
"""
    out.write_text(report_html, encoding="utf-8")
    return out


def write_optimization_report_html(result: OptimizationResult, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    method = html.escape(str(result.method))
    metric = html.escape(str(result.metric))
    top_trials = sorted(result.trials, key=lambda item: item.score, reverse=result.maximize)[:20]
    rows = []
    for idx, trial in enumerate(top_trials, start=1):
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{trial.score:.6f}</td>"
            f"<td>{html.escape(json.dumps(trial.params, ensure_ascii=False))}</td>"
            "</tr>"
        )
    report_html = f"""
<html>
<head>
  <meta charset="utf-8" />
  <title>QuantEngine Optimization Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 16px; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>Optimization Report</h1>
  <p>method={method}, metric={metric}, maximize={result.maximize}</p>
  <p>best_score={result.best_score:.6f}, best_params={html.escape(json.dumps(result.best_params, ensure_ascii=False))}</p>
  <table>
    <thead><tr><th>#</th><th>Score</th><th>Params</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out.write_text(report_html, encoding="utf-8")
    return out


def write_walk_forward_report_html(result: WalkForwardResult, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fold_rows: list[str] = []
    metric = None
    if result.folds:
        metric = None
        for key in ("sharpe", "total_return", "annualized_return"):
            if key in result.folds[0].oos_report.performance:
                metric = key
                break

    for fold in result.folds:
        metric_key = metric or "performance"
        is_metric = fold.is_report.performance.get(metric_key)
        oos_metric = fold.oos_report.performance.get(metric_key)
        is_value = "N/A" if is_metric is None else html.escape(str(is_metric))
        oos_value = "N/A" if oos_metric is None else html.escape(str(oos_metric))
        ratio = "N/A"
        if isinstance(is_metric, (int, float)) and isinstance(oos_metric, (int, float)) and abs(float(is_metric)) > 1e-12:
            ratio = html.escape(f"{(float(oos_metric) / float(is_metric)):.6f}")
        fold_rows.append(
            "<tr>"
            f"<td>{fold.fold_index}</td>"
            f"<td>{fold.is_start}-{fold.is_end}</td>"
            f"<td>{fold.oos_start}-{fold.oos_end}</td>"
            f"<td>{html.escape(str(fold.best_params))}</td>"
            f"<td>{is_value}</td>"
            f"<td>{oos_value}</td>"
            f"<td>{ratio}</td>"
            "</tr>"
        )

    agg_perf_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in result.aggregate_oos_performance.items()
    )
    agg_risk_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in result.aggregate_oos_risk.items()
    )

    report_html = f"""
<html>
<head>
  <meta charset="utf-8" />
  <title>QuantEngine Walk-Forward Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 16px; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
    .warn {{ color: #c0392b; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Walk-Forward Report</h1>
  <p>method={html.escape(result.config.optimization_method)}, n_splits={result.config.n_splits}, in_sample_ratio={result.config.in_sample_ratio}, expanding={result.config.expanding}</p>
  <p class='warn'>overfitting_ratio={result.overfitting_ratio:.6f}</p>
  <h2>Fold Comparison (IS/OOS)</h2>
  <table>
    <thead><tr><th>Fold</th><th>IS Range</th><th>OOS Range</th><th>Best Params</th><th>IS Metric</th><th>OOS Metric</th><th>OOS/IS</th></tr></thead>
    <tbody>
      {''.join(fold_rows)}
    </tbody>
  </table>
  <h2>Aggregate OOS Performance</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{agg_perf_rows}</tbody>
  </table>
  <h2>Aggregate OOS Risk</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{agg_risk_rows}</tbody>
  </table>
</body>
</html>
"""
    out.write_text(report_html, encoding="utf-8")
    return out
