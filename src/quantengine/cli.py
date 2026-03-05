from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from quantengine.check_deps import run_check
from quantengine.config import ConfigError, QuantEngineConfig, load_config
from quantengine.data.loader import DataLoader
from quantengine.engine import BacktestEngine
from quantengine.engine.factory import build_engine
from quantengine.optimizer import (
    BayesianOptimizer,
    GeneticOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    WalkForwardAnalyzer,
    WalkForwardConfig,
)
from quantengine.strategy import ParameterSpace, get_strategy, list_strategies
from quantengine.visualization.reports import (
    write_backtest_report_html,
    write_optimization_report_html,
    write_walk_forward_report_html,
)

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", default=None, help="YAML 配置路径")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """QuantEngine CLI"""
    logging.basicConfig(level=logging.INFO)
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    ctx.obj = {"config": config}


@main.command("check-deps")
@click.option("--engine", is_flag=True, help="同时检查 GPU/优化器可选依赖")
@click.option("--json", "output_json", is_flag=True, help="输出 JSON 供 CI 解析")
def check_deps_cmd(engine: bool, output_json: bool) -> None:
    """检查 Python 版本与核心/可选依赖。"""
    result = run_check(include_engine=engine)
    if output_json:
        payload = {
            "python": {
                "name": result["python"].name,
                "required": result["python"].required,
                "installed": result["python"].installed,
                "status": result["python"].status,
            },
            "core": [
                {"name": r.name, "required": r.required, "installed": r.installed, "status": r.status}
                for r in result["core"]
            ],
            "engine": (
                [
                    {"name": r.name, "required": r.required, "installed": r.installed, "status": r.status}
                    for r in result["engine"]
                ]
                if result["engine"]
                else None
            ),
            "all_ok": result["all_ok"],
        }
        console.print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    table = Table(title="依赖检查")
    table.add_column("包名", style="cyan")
    table.add_column("要求版本", style="dim")
    table.add_column("已安装版本", justify="right")
    table.add_column("状态", justify="center")
    for r in [result["python"]] + result["core"] + (result["engine"] or []):
        installed = r.installed or "-"
        style = "green" if r.status == "OK" else "red" if r.status == "缺失" else "yellow"
        table.add_row(r.name, r.required, installed, f"[{style}]{r.status}[/{style}]")
    console.print(table)
    if not result["all_ok"]:
        raise SystemExit(1)


@main.command("list-strategies")
def list_strategies_cmd() -> None:
    table = Table(title="Registered Strategies")
    table.add_column("Name", style="cyan")
    for name in list_strategies():
        table.add_row(name)
    console.print(table)


@main.command("backtest")
@click.option("--strategy", default=None, help="策略名称")
@click.option("--strategies", default=None, help="多策略，逗号分隔")
@click.option("--data", "data_path", required=True, help="数据文件或目录")
@click.option("--params", default="{}", help="JSON 参数（单策略）或策略参数映射（多策略）")
@click.option("--symbols", default="", help="symbol 过滤，逗号分隔")
@click.option("--output", default=None, help="结果 JSON 输出路径")
@click.option("--report", "report_path", default=None, help="HTML 报告输出路径")
@click.pass_context
def backtest_cmd(
    ctx: click.Context,
    strategy: str | None,
    strategies: str | None,
    data_path: str,
    params: str,
    symbols: str,
    output: str | None,
    report_path: str | None,
) -> None:
    config: QuantEngineConfig = ctx.obj["config"]
    symbol_list = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    loader = DataLoader(backend=config.runtime.backend, use_gpu=config.runtime.use_gpu)
    data = loader.load(data_path, symbols=symbol_list or None)
    engine = _build_engine(config)
    params_obj = _parse_json(params)

    if strategies:
        strategy_specs: list[tuple] = []
        names = [item.strip() for item in strategies.split(",") if item.strip()]
        for name in names:
            each_params = params_obj.get(name, {}) if isinstance(params_obj, dict) else {}
            strategy_specs.append((get_strategy(name), each_params, 1.0))
        result = engine.run_multi_strategy(data=data, strategies=strategy_specs)
    else:
        if not strategy:
            raise click.ClickException("单策略回测必须提供 --strategy")
        result = engine.run(data=data, strategy=get_strategy(strategy), params=params_obj)

    _print_metrics(result.performance, title="Performance")
    _print_metrics(result.risk, title="Risk")
    _print_metrics(result.trade_metrics, title="Trade Metrics")

    output_path = Path(output) if output else _default_result_path(config, prefix="backtest")
    payload = {
        "type": "backtest",
        "strategy": result.strategy,
        "params": result.params,
        "performance": result.performance,
        "risk": result.risk,
        "trade_metrics": result.trade_metrics,
        "portfolio": result.portfolio.to_dict(),
        "timestamps": [_to_isoformat(ts) for ts in data.timestamps],
    }
    _write_json(output_path, payload)
    console.print(f"[green]结果已写入[/green] {output_path}")

    if report_path:
        html_path = write_backtest_report_html(result, timestamps=data.timestamps, output_path=report_path)
        console.print(f"[green]报告已写入[/green] {html_path}")


@main.command("optimize")
@click.option("--strategy", required=True, help="策略名称")
@click.option("--data", "data_path", required=True, help="数据文件或目录")
@click.option("--method", default="bayesian", type=click.Choice(["grid", "random", "bayesian", "genetic"]))
@click.option("--n-trials", default=None, type=int, help="搜索次数")
@click.option("--metric", default=None, help="目标指标，例如 sharpe")
@click.option("--minimize", is_flag=True, default=False, help="最小化目标")
@click.option("--symbols", default="", help="symbol 过滤，逗号分隔")
@click.option("--param-grid", default=None, help="仅 grid 使用，JSON 格式参数网格")
@click.option("--output", default=None, help="优化结果 JSON 输出路径")
@click.option("--report", "report_path", default=None, help="优化报告 HTML 路径")
@click.pass_context
def optimize_cmd(
    ctx: click.Context,
    strategy: str,
    data_path: str,
    method: str,
    n_trials: int | None,
    metric: str | None,
    minimize: bool,
    symbols: str,
    param_grid: str | None,
    output: str | None,
    report_path: str | None,
) -> None:
    config: QuantEngineConfig = ctx.obj["config"]
    symbol_list = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    loader = DataLoader(backend=config.runtime.backend, use_gpu=config.runtime.use_gpu)
    data = loader.load(data_path, symbols=symbol_list or None)
    engine = _build_engine(config)

    metric_name = metric or config.optimize.metric
    maximize = False if minimize else config.optimize.maximize
    trial_count = n_trials or config.optimize.n_trials
    strategy_factory = _build_strategy_factory(strategy, param_grid)

    if method == "grid":
        optimizer = GridSearchOptimizer(
            engine=engine,
            data=data,
            strategy_factory=strategy_factory,
            metric=metric_name,
            maximize=maximize,
            max_workers=config.optimize.max_workers,
            batch_size=config.optimize.batch_size,
            show_progress=True,
        )
    elif method == "random":
        optimizer = RandomSearchOptimizer(
            engine=engine,
            data=data,
            strategy_factory=strategy_factory,
            n_trials=trial_count,
            metric=metric_name,
            maximize=maximize,
            random_seed=config.optimize.random_seed,
            max_workers=config.optimize.max_workers,
            batch_size=config.optimize.batch_size,
            show_progress=True,
        )
    elif method == "bayesian":
        optimizer = BayesianOptimizer(
            engine=engine,
            data=data,
            strategy_factory=strategy_factory,
            n_trials=trial_count,
            metric=metric_name,
            maximize=maximize,
            random_seed=config.optimize.random_seed,
            max_workers=config.optimize.max_workers,
            show_progress=True,
        )
    else:
        optimizer = GeneticOptimizer(
            engine=engine,
            data=data,
            strategy_factory=strategy_factory,
            n_generations=max(1, trial_count // max(2, config.optimize.batch_size)),
            population_size=max(4, config.optimize.batch_size),
            metric=metric_name,
            maximize=maximize,
            random_seed=config.optimize.random_seed,
            show_progress=True,
        )

    result = optimizer.optimize()
    console.print(
        f"[green]best[/green] method={result.method} metric={result.metric} "
        f"score={result.best_score:.6f} params={result.best_params}"
    )
    output_path = Path(output) if output else _default_result_path(config, prefix=f"opt_{method}")
    _write_json(output_path, {"type": "optimization", **result.as_dict()})
    console.print(f"[green]结果已写入[/green] {output_path}")

    if report_path:
        html_path = write_optimization_report_html(result, output_path=report_path)
        console.print(f"[green]报告已写入[/green] {html_path}")


@main.command("walk-forward")
@click.option("--strategy", required=True, help="策略名称")
@click.option("--data", "data_path", required=True, help="数据文件或目录")
@click.option("--n-splits", default=5, type=int, help="折数")
@click.option("--is-ratio", default=0.7, type=float, help="IS 区间占比")
@click.option("--method", default="bayesian", type=click.Choice(["grid", "random", "bayesian", "genetic"]))
@click.option("--n-trials", default=None, type=int, help="每折优化次数")
@click.option("--metric", default=None, help="目标指标")
@click.option("--minimize", is_flag=True, default=False, help="最小化目标")
@click.option("--symbols", default="", help="symbol 过滤，逗号分隔")
@click.option("--expanding", is_flag=True, default=False, help="是否使用 expanding 窗口")
@click.option("--output", default=None, help="结果 JSON 输出路径")
@click.option("--report", "report_path", default=None, help="Walk-Forward HTML 路径")
@click.pass_context
def walk_forward_cmd(
    ctx: click.Context,
    strategy: str,
    data_path: str,
    n_splits: int,
    is_ratio: float,
    method: str,
    n_trials: int | None,
    metric: str | None,
    minimize: bool,
    symbols: str,
    expanding: bool,
    output: str | None,
    report_path: str | None,
) -> None:
    config: QuantEngineConfig = ctx.obj["config"]
    symbol_list = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    loader = DataLoader(backend=config.runtime.backend, use_gpu=config.runtime.use_gpu)
    data = loader.load(data_path, symbols=symbol_list or None)
    engine = _build_engine(config)
    strategy_factory = _build_strategy_factory(strategy, None)

    metric_name = metric or config.optimize.metric
    maximize = False if minimize else config.optimize.maximize
    trial_count = n_trials or config.optimize.n_trials

    analyzer = WalkForwardAnalyzer(
        engine=engine,
        data=data,
        strategy_factory=strategy_factory,
        metric=metric_name,
        maximize=maximize,
        wf_config=WalkForwardConfig(
            n_splits=n_splits,
            in_sample_ratio=is_ratio,
            expanding=expanding,
            optimization_method=method,
            n_trials=trial_count,
        ),
        show_progress=True,
    )
    result = analyzer.run()
    console.print(f"[green]walk-forward[/green] folds={len(result.folds)} overfitting={result.overfitting_ratio:.6f}")
    output_path = Path(output) if output else _default_result_path(config, prefix="walk_forward")
    payload = {"type": "walk_forward", **result.as_dict()}
    _write_json(output_path, payload)
    console.print(f"[green]结果已写入[/green] {output_path}")
    if report_path:
        html_path = write_walk_forward_report_html(result, output_path=report_path)
        console.print(f"[green]报告已写入[/green] {html_path}")


@main.command("report")
@click.option("--result", "result_path", required=True, help="backtest/optimization JSON 路径")
@click.option("--output", "output_path", required=True, help="HTML 输出路径")
def report_cmd(result_path: str, output_path: str) -> None:
    path = Path(result_path)
    if not path.exists():
        raise click.ClickException(f"结果文件不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rtype = payload.get("type")
    if rtype == "backtest":
        portfolio = _Portfolio(
            equity_curve=np.asarray(payload.get("portfolio", {}).get("equity_curve", []), dtype=float),
            trades=payload.get("portfolio", {}).get("trades", []),
            metadata=payload.get("portfolio", {}).get("metadata", {}),
        )
        report = _BacktestProxy(
            strategy=payload.get("strategy", "unknown"),
            params=payload.get("params", {}),
            performance=payload.get("performance", {}),
            risk=payload.get("risk", {}),
            trade_metrics=payload.get("trade_metrics", {}),
            portfolio=portfolio,
        )
        timestamps = payload.get("timestamps", [])
        write_backtest_report_html(report, timestamps=timestamps, output_path=output_path)
        console.print(f"[green]报告已写入[/green] {output_path}")
        return
    if rtype == "optimization":
        result = _OptimizationProxy(
            method=payload["method"],
            metric=payload["metric"],
            maximize=payload["maximize"],
            best_params=payload["best_params"],
            best_score=payload["best_score"],
            trials=[
                _TrialProxy(params=item.get("params", {}), score=float(item.get("score", 0.0)))
                for item in payload.get("trials", [])
            ],
        )
        write_optimization_report_html(result, output_path=output_path)
        console.print(f"[green]报告已写入[/green] {output_path}")
        return
    raise click.ClickException("不支持的 result type")


def _build_engine(config: QuantEngineConfig) -> BacktestEngine:
    return build_engine(config)


def _build_strategy_factory(strategy_name: str, param_grid: str | None):
    override = _parse_json(param_grid) if param_grid else None

    def _factory():
        strategy = get_strategy(strategy_name)
        if override:
            base_method = strategy.parameters

            def _params(_strategy: Any) -> dict[str, ParameterSpace]:
                spaces = base_method()
                for key, values in override.items():
                    values_list = list(values)
                    spaces[key] = ParameterSpace(kind="choice", choices=values_list)
                return spaces

            strategy.parameters = MethodType(_params, strategy)
        return strategy

    return _factory


def _parse_json(raw: str | None) -> Any:
    if raw is None:
        return {}
    token = raw.strip()
    if not token:
        return {}
    try:
        return json.loads(token)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"JSON 解析失败: {exc}") from exc


def _to_isoformat(ts: Any) -> str:
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_result_path(config: QuantEngineConfig, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(config.results_root) / f"{prefix}_{timestamp}.json"


def _print_metrics(metrics: dict[str, Any], title: str) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            table.add_row(key, f"{float(value):.6f}")
        else:
            table.add_row(key, str(value))
    console.print(table)


@dataclass
class _Portfolio:
    equity_curve: np.ndarray
    trades: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _BacktestProxy:
    strategy: str
    params: dict[str, Any]
    performance: dict[str, float]
    risk: dict[str, float]
    trade_metrics: dict[str, float]
    portfolio: _Portfolio


@dataclass
class _TrialProxy:
    params: dict[str, Any]
    score: float


@dataclass
class _OptimizationProxy:
    method: str
    metric: str
    maximize: bool
    best_params: dict[str, Any]
    best_score: float
    trials: list[_TrialProxy]


if __name__ == "__main__":
    main()
