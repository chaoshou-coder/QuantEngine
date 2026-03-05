from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

from quantengine.engine.backtest import BacktestReport
from quantengine.metrics.stability import (
    MultipleTestingResult,
    ParameterSensitivityPoint,
    RegimeResult,
    benjamini_hochberg_correction,
)
from quantengine.optimizer.base import OptimizationResult
from quantengine.optimizer.walk_forward import WalkForwardResult

from .plots import build_equity_figure


@dataclass(slots=True)
class ReportConfig:
    language: str = "zh-CN"
    title: str | None = None
    include_meta: bool = True
    include_params: bool = True
    include_performance: bool = True
    include_risk: bool = True
    include_trade_metrics: bool = True
    include_equity_plot: bool = True
    include_cost_comparison: bool = True
    include_multiple_testing: bool = True
    include_parameter_sensitivity: bool = True
    include_regime_breakdown: bool = True
    include_plotlyjs: str | bool = "cdn"
    float_precision: int = 6
    export_pdf: bool = False
    pdf_output_path: str | Path | None = None


def _labels(language: str) -> dict[str, str]:
    if language.strip().lower().startswith("zh"):
        return {
            "title": "QuantEngine 回测报告",
            "meta": "元信息",
            "params": "参数",
            "performance": "绩效指标",
            "risk": "风险指标",
            "trade_metrics": "交易指标",
            "equity_plot": "净值与回撤",
            "cost_comparison": "成本情景对比",
            "multiple_testing": "多重检验校正",
            "parameter_sensitivity": "参数敏感度",
            "regime_breakdown": "市场状态分解",
            "no_data": "无数据",
            "plot_unavailable": "plotly 不可用，图表未生成",
            "hypothesis": "假设",
            "raw_p": "原始 p 值",
            "adjusted_p": "校正后 p 值",
            "significant": "是否显著",
        }
    return {
        "title": "QuantEngine Backtest Report",
        "meta": "Meta",
        "params": "Params",
        "performance": "Performance",
        "risk": "Risk",
        "trade_metrics": "Trade Metrics",
        "equity_plot": "Equity & Drawdown",
        "cost_comparison": "Cost Scenario Comparison",
        "multiple_testing": "Multiple Testing Correction",
        "parameter_sensitivity": "Parameter Sensitivity",
        "regime_breakdown": "Regime Breakdown",
        "no_data": "No Data",
        "plot_unavailable": "plotly not available",
        "hypothesis": "Hypothesis",
        "raw_p": "Raw p-value",
        "adjusted_p": "Adjusted p-value",
        "significant": "Significant",
    }


def _format_value(value: Any, float_precision: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Real):
        return f"{float(value):.{float_precision}f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _rows_to_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], float_precision: int) -> str:
    if not rows:
        return "<p>-</p>"
    header_html = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_format_value(cell, float_precision))}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _dict_to_table(
    data: Mapping[str, Any],
    float_precision: int,
    key_header: str = "Key",
    value_header: str = "Value",
) -> str:
    rows = [(key, value) for key, value in data.items()]
    return _rows_to_table([key_header, value_header], rows, float_precision=float_precision)


def _render_section(title: str, body: str) -> str:
    return f"<h2>{html.escape(title)}</h2>{body}"


def _normalize_multiple_testing(
    data: MultipleTestingResult | Mapping[str, Any] | None,
) -> MultipleTestingResult | None:
    if data is None:
        return None
    if isinstance(data, MultipleTestingResult):
        return data
    if not isinstance(data, Mapping):
        return None

    raw = data.get("raw_p_values", data.get("p_values", []))
    adjusted = data.get("adjusted_p_values")
    rejected = data.get("rejected")
    alpha = float(data.get("alpha", 0.05))

    if adjusted is None or rejected is None:
        return benjamini_hochberg_correction(raw, alpha=alpha)

    raw_list = [float(item) for item in list(raw)]
    adj_list = [float(item) for item in list(adjusted)]
    rej_list = [bool(item) for item in list(rejected)]
    significant = int(sum(1 for item in rej_list if item))
    summary = str(
        data.get(
            "summary",
            f"Benjamini-Hochberg 校正完成：显著 {significant}/{len(raw_list)}，alpha={alpha:.4f}。",
        )
    )
    return MultipleTestingResult(
        method=str(data.get("method", "benjamini-hochberg")),
        alpha=alpha,
        raw_p_values=raw_list,
        adjusted_p_values=adj_list,
        rejected=rej_list,
        significant_count=significant,
        summary=summary,
    )


def _normalize_parameter_sensitivity(
    data: Sequence[ParameterSensitivityPoint] | Sequence[Mapping[str, Any]] | None,
) -> list[ParameterSensitivityPoint]:
    if not data:
        return []
    if isinstance(data[0], ParameterSensitivityPoint):
        return [item for item in data if isinstance(item, ParameterSensitivityPoint)]

    points: list[ParameterSensitivityPoint] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        points.append(
            ParameterSensitivityPoint(
                param_name=str(item.get("param_name", "")),
                base_value=float(item.get("base_value", 0.0)),
                minus_value=float(item.get("minus_value", 0.0)),
                plus_value=float(item.get("plus_value", 0.0)),
                minus_score=float(item.get("minus_score", 0.0)),
                base_score=float(item.get("base_score", 0.0)),
                plus_score=float(item.get("plus_score", 0.0)),
            )
        )
    return points


def _cost_scenario_rows(cost_scenarios: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for name, item in cost_scenarios.items():
        if isinstance(item, BacktestReport):
            performance = item.performance
            risk = item.risk
        elif isinstance(item, Mapping):
            performance = item.get("performance", item)
            risk = item.get("risk", {})
        else:
            continue

        total_return = performance.get("total_return", 0.0)
        sharpe = performance.get("sharpe", 0.0)
        win_rate = performance.get("win_rate", 0.0)
        max_dd = risk.get("max_drawdown", performance.get("max_drawdown", 0.0))
        rows.append([name, total_return, sharpe, win_rate, max_dd])
    return rows


def _regime_rows(regimes: Sequence[RegimeResult | Mapping[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in regimes:
        if isinstance(item, RegimeResult):
            rows.append([item.regime_name, item.n_bars, item.sharpe, item.win_rate, item.max_dd])
            continue
        if not isinstance(item, Mapping):
            continue
        rows.append(
            [
                item.get("regime_name", item.get("name", "-")),
                item.get("n_bars", item.get("bars", 0)),
                item.get("sharpe", 0.0),
                item.get("win_rate", 0.0),
                item.get("max_dd", item.get("max_drawdown", 0.0)),
            ]
        )
    return rows


def _load_weasyprint_html():
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("weasyprint 未安装，无法导出 PDF。请先安装 weasyprint。") from exc
    return HTML


def export_html_to_pdf(
    html_content: str,
    html_output_path: str | Path,
    pdf_output_path: str | Path | None = None,
) -> Path:
    html_out = Path(html_output_path)
    pdf_out = Path(pdf_output_path) if pdf_output_path else html_out.with_suffix(".pdf")
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    html_cls = _load_weasyprint_html()
    html_cls(string=html_content, base_url=str(html_out.parent)).write_pdf(str(pdf_out))
    return pdf_out


def write_backtest_report(
    report: BacktestReport,
    timestamps: Sequence,
    output_path: str | Path,
    config: ReportConfig | None = None,
    cost_scenario_reports: Mapping[str, BacktestReport] | Mapping[str, Any] | None = None,
    multiple_testing: MultipleTestingResult | Mapping[str, Any] | None = None,
    parameter_sensitivity: Sequence[ParameterSensitivityPoint] | Sequence[Mapping[str, Any]] | None = None,
    regime_breakdown: Sequence[RegimeResult | Mapping[str, Any]] | None = None,
) -> dict[str, Path | None]:
    cfg = config or ReportConfig()
    labels = _labels(cfg.language)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    metadata = report.metadata if isinstance(report.metadata, Mapping) else {}
    trades = report.portfolio.trades if isinstance(report.portfolio.trades, list) else []
    meta = {
        "strategy": report.strategy,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "trades": len(trades),
        "language": cfg.language,
    }

    sections: list[str] = []
    if cfg.include_meta:
        sections.append(
            _render_section(
                labels["meta"],
                _dict_to_table(meta, float_precision=cfg.float_precision, key_header="Key", value_header="Value"),
            )
        )
    if cfg.include_params:
        sections.append(
            _render_section(
                labels["params"],
                _dict_to_table(
                    report.params,
                    float_precision=cfg.float_precision,
                    key_header="Key",
                    value_header="Value",
                ),
            )
        )
    if cfg.include_performance:
        sections.append(
            _render_section(
                labels["performance"],
                _dict_to_table(
                    report.performance,
                    float_precision=cfg.float_precision,
                    key_header="Metric",
                    value_header="Value",
                ),
            )
        )
    if cfg.include_risk:
        sections.append(
            _render_section(
                labels["risk"],
                _dict_to_table(
                    report.risk,
                    float_precision=cfg.float_precision,
                    key_header="Metric",
                    value_header="Value",
                ),
            )
        )
    if cfg.include_trade_metrics:
        sections.append(
            _render_section(
                labels["trade_metrics"],
                _dict_to_table(
                    report.trade_metrics,
                    float_precision=cfg.float_precision,
                    key_header="Metric",
                    value_header="Value",
                ),
            )
        )

    if cfg.include_equity_plot:
        figure_html = f"<p>{html.escape(labels['plot_unavailable'])}</p>"
        try:
            fig = build_equity_figure(
                timestamps=timestamps,
                equity_curve=report.portfolio.equity_curve,
                title=f"Equity - {report.strategy}",
            )
            figure_html = fig.to_html(full_html=False, include_plotlyjs=cfg.include_plotlyjs)
        except Exception:
            pass
        sections.append(_render_section(labels["equity_plot"], figure_html))

    scenario_source = cost_scenario_reports
    if scenario_source is None:
        from_metadata = metadata.get("cost_scenarios")
        if isinstance(from_metadata, Mapping):
            scenario_source = from_metadata
    if cfg.include_cost_comparison and scenario_source:
        rows = _cost_scenario_rows(scenario_source)
        if rows:
            sections.append(
                _render_section(
                    labels["cost_comparison"],
                    _rows_to_table(
                        headers=["Scenario", "Total Return", "Sharpe", "Win Rate", "Max Drawdown"],
                        rows=rows,
                        float_precision=cfg.float_precision,
                    ),
                )
            )

    mt_source = _normalize_multiple_testing(multiple_testing)
    if mt_source is None:
        mt_source = _normalize_multiple_testing(metadata.get("multiple_testing"))
    if cfg.include_multiple_testing and mt_source is not None:
        mt_rows = [
            [idx, raw, adj, "Yes" if rejected else "No"]
            for idx, (raw, adj, rejected) in enumerate(
                zip(mt_source.raw_p_values, mt_source.adjusted_p_values, mt_source.rejected, strict=False),
                start=1,
            )
        ]
        mt_table = _rows_to_table(
            headers=[
                labels["hypothesis"],
                labels["raw_p"],
                labels["adjusted_p"],
                labels["significant"],
            ],
            rows=mt_rows,
            float_precision=cfg.float_precision,
        )
        mt_body = f"<p>{html.escape(mt_source.summary)}</p>{mt_table}"
        sections.append(_render_section(labels["multiple_testing"], mt_body))

    sensitivity_source = _normalize_parameter_sensitivity(parameter_sensitivity)
    if not sensitivity_source:
        metadata_sensitivity = metadata.get("parameter_sensitivity")
        if isinstance(metadata_sensitivity, Sequence):
            sensitivity_source = _normalize_parameter_sensitivity(metadata_sensitivity)
    if cfg.include_parameter_sensitivity and sensitivity_source:
        sens_rows = [
            [
                item.param_name,
                item.minus_value,
                item.base_value,
                item.plus_value,
                item.minus_score,
                item.base_score,
                item.plus_score,
            ]
            for item in sensitivity_source
        ]
        sections.append(
            _render_section(
                labels["parameter_sensitivity"],
                _rows_to_table(
                    headers=[
                        "Param",
                        "-10% Value",
                        "Base Value",
                        "+10% Value",
                        "-10% Score",
                        "Base Score",
                        "+10% Score",
                    ],
                    rows=sens_rows,
                    float_precision=cfg.float_precision,
                ),
            )
        )

    regimes_source = regime_breakdown
    if regimes_source is None:
        from_metadata = metadata.get("regime_breakdown")
        if isinstance(from_metadata, Sequence):
            regimes_source = from_metadata
    if cfg.include_regime_breakdown and regimes_source:
        rows = _regime_rows(regimes_source)
        if rows:
            sections.append(
                _render_section(
                    labels["regime_breakdown"],
                    _rows_to_table(
                        headers=["Regime", "Bars", "Sharpe", "Win Rate", "Max Drawdown"],
                        rows=rows,
                        float_precision=cfg.float_precision,
                    ),
                )
            )

    report_title = cfg.title or labels["title"]
    report_html = f"""
<html>
<head>
  <meta charset="utf-8" />
  <title>{html.escape(report_title)}</title>
  <style>
    body {{ font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 16px; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>{html.escape(report_title)}</h1>
  {"".join(sections)}
</body>
</html>
"""
    out.write_text(report_html, encoding="utf-8")
    pdf_path: Path | None = None
    if cfg.export_pdf:
        pdf_path = export_html_to_pdf(
            html_content=report_html,
            html_output_path=out,
            pdf_output_path=cfg.pdf_output_path,
        )
    return {"html": out, "pdf": pdf_path}


def write_backtest_report_html(
    report: BacktestReport,
    timestamps: Sequence,
    output_path: str | Path,
) -> Path:
    output = write_backtest_report(
        report=report,
        timestamps=timestamps,
        output_path=output_path,
        config=ReportConfig(),
    )
    return output["html"] if output["html"] is not None else Path(output_path)


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
  <title>QuantEngine 参数优化报告</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 16px; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>参数优化报告</h1>
  <p>method={method}, metric={metric}, maximize={result.maximize}</p>
  <p>best_score={result.best_score:.6f}, best_params={html.escape(json.dumps(result.best_params, ensure_ascii=False))}</p>
  <table>
    <thead><tr><th>#</th><th>Score</th><th>Params</th></tr></thead>
    <tbody>
      {"".join(rows)}
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
        if (
            isinstance(is_metric, (int, float))
            and isinstance(oos_metric, (int, float))
            and abs(float(is_metric)) > 1e-12
        ):
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
  <title>QuantEngine Walk-Forward 报告</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 16px; width: 100%; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
    .warn {{ color: #c0392b; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Walk-Forward 报告</h1>
  <p>method={html.escape(result.config.optimization_method)}, n_splits={result.config.n_splits}, in_sample_ratio={result.config.in_sample_ratio}, expanding={result.config.expanding}</p>
  <p class='warn'>overfitting_ratio={result.overfitting_ratio:.6f}</p>
  <h2>分折对比（IS/OOS）</h2>
  <table>
    <thead><tr><th>Fold</th><th>IS Range</th><th>OOS Range</th><th>Best Params</th><th>IS Metric</th><th>OOS Metric</th><th>OOS/IS</th></tr></thead>
    <tbody>
      {"".join(fold_rows)}
    </tbody>
  </table>
  <h2>OOS 聚合绩效</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{agg_perf_rows}</tbody>
  </table>
  <h2>OOS 聚合风险</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{agg_risk_rows}</tbody>
  </table>
</body>
</html>
"""
    out.write_text(report_html, encoding="utf-8")
    return out
