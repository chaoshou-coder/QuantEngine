from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from quantengine.data.loader import DataBundle
from quantengine.engine.backtest import BacktestEngine, BacktestReport
from quantengine.optimizer.bayesian import BayesianOptimizer
from quantengine.optimizer.genetic import GeneticOptimizer
from quantengine.optimizer.grid import GridSearchOptimizer
from quantengine.optimizer.random_search import RandomSearchOptimizer
from quantengine.strategy.base import BaseStrategy


@dataclass
class WalkForwardConfig:
    n_splits: int = 5
    in_sample_ratio: float = 0.7
    expanding: bool = False
    optimization_method: str = "bayesian"
    n_trials: int = 100


@dataclass
class WalkForwardFold:
    fold_index: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    best_params: dict[str, Any]
    is_report: BacktestReport
    oos_report: BacktestReport


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    aggregate_oos_performance: dict[str, float]
    aggregate_oos_risk: dict[str, float]
    overfitting_ratio: float
    config: WalkForwardConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "folds": [
                {
                    "fold_index": fold.fold_index,
                    "is_start": fold.is_start,
                    "is_end": fold.is_end,
                    "oos_start": fold.oos_start,
                    "oos_end": fold.oos_end,
                    "best_params": fold.best_params,
                    "is_report": _report_to_dict(fold.is_report),
                    "oos_report": _report_to_dict(fold.oos_report),
                }
                for fold in self.folds
            ],
            "aggregate_oos_performance": self.aggregate_oos_performance,
            "aggregate_oos_risk": self.aggregate_oos_risk,
            "overfitting_ratio": self.overfitting_ratio,
        }


def _report_to_dict(report: BacktestReport) -> dict[str, Any]:
    return {
        "strategy": report.strategy,
        "params": report.params,
        "performance": report.performance,
        "risk": report.risk,
        "trade_metrics": report.trade_metrics,
        "metadata": report.metadata,
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


class WalkForwardAnalyzer:
    def __init__(
        self,
        engine: BacktestEngine,
        data: DataBundle,
        strategy_factory: Callable[[], BaseStrategy],
        wf_config: WalkForwardConfig | None = None,
        metric: str = "sharpe",
        maximize: bool = True,
        show_progress: bool = False,
    ):
        self.engine = engine
        self.data = data
        self.strategy_factory = strategy_factory
        self.wf_config = wf_config or WalkForwardConfig()
        self.metric = metric
        self.maximize = maximize
        self.show_progress = show_progress

    def run(self) -> WalkForwardResult:
        if self.wf_config.n_splits <= 0:
            raise ValueError("n_splits 必须大于 0")
        if not 0 < self.wf_config.in_sample_ratio < 1:
            raise ValueError("in_sample_ratio 必须在 (0, 1) 区间")

        folds = self._build_folds(len(self.data.timestamps))
        if not folds:
            raise ValueError("样本长度不足，无法构建 walk-forward 折叠")

        progress = None
        task = None
        if self.show_progress:
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            progress.start()
            task = progress.add_task("walk-forward", total=len(folds))
        results: list[WalkForwardFold] = []
        try:
            for fold in folds:
                fold_index, is_start, is_end, oos_start, oos_end = fold
                is_data = self.data.slice_by_index(is_start, is_end)
                oos_data = self.data.slice_by_index(oos_start, oos_end)
                optimizer = self._build_optimizer(is_data)
                wf_opt_result = optimizer.optimize()
                best_params = wf_opt_result.best_params
                if wf_opt_result.best_report is None:
                    is_report = self.engine.run(is_data, self.strategy_factory(), best_params, record_trades=False)
                else:
                    is_report = wf_opt_result.best_report
                oos_report = self.engine.run(oos_data, self.strategy_factory(), best_params, record_trades=False)
                results.append(
                    WalkForwardFold(
                        fold_index=fold_index,
                        is_start=is_start,
                        is_end=is_end,
                        oos_start=oos_start,
                        oos_end=oos_end,
                        best_params=best_params,
                        is_report=is_report,
                        oos_report=oos_report,
                    )
                )
                if task is not None and progress is not None:
                    progress.update(task, advance=1)
        finally:
            if progress is not None:
                progress.stop()

        aggregate_oos_performance = self._average_metrics([item.oos_report.performance for item in results])
        aggregate_oos_risk = self._average_metrics([item.oos_report.risk for item in results])
        overfitting_ratio = self._calc_overfitting(results)
        return WalkForwardResult(
            folds=results,
            aggregate_oos_performance=aggregate_oos_performance,
            aggregate_oos_risk=aggregate_oos_risk,
            overfitting_ratio=overfitting_ratio,
            config=self.wf_config,
        )

    def _build_optimizer(self, data: DataBundle):
        method = self.wf_config.optimization_method.strip().lower()
        method_trials = max(1, int(self.wf_config.n_trials))
        if method == "grid":
            return GridSearchOptimizer(
                engine=self.engine,
                data=data,
                strategy_factory=self.strategy_factory,
                metric=self.metric,
                maximize=self.maximize,
                show_progress=self.show_progress,
            )
        if method == "random":
            return RandomSearchOptimizer(
                engine=self.engine,
                data=data,
                strategy_factory=self.strategy_factory,
                n_trials=method_trials,
                metric=self.metric,
                maximize=self.maximize,
                random_seed=42,
                show_progress=self.show_progress,
            )
        if method == "genetic":
            generations = max(1, method_trials // max(2, 10))
            return GeneticOptimizer(
                engine=self.engine,
                data=data,
                strategy_factory=self.strategy_factory,
                n_generations=generations,
                population_size=10,
                metric=self.metric,
                maximize=self.maximize,
                random_seed=42,
                show_progress=self.show_progress,
            )
        if method == "bayesian":
            return BayesianOptimizer(
                engine=self.engine,
                data=data,
                strategy_factory=self.strategy_factory,
                n_trials=method_trials,
                metric=self.metric,
                maximize=self.maximize,
                random_seed=42,
                show_progress=self.show_progress,
            )
        raise ValueError(f"未知 walk-forward 优化方法: {method}")

    def _build_folds(self, total_bars: int) -> list[tuple[int, int, int, int, int]]:
        if total_bars < 4:
            return []
        block = max(3, total_bars // max(1, self.wf_config.n_splits))
        is_size = max(2, int(block * self.wf_config.in_sample_ratio))
        oos_size = max(1, block - is_size)
        if is_size + oos_size >= total_bars:
            return []
        folds = []
        if self.wf_config.expanding:
            is_end = is_size
            for idx in range(self.wf_config.n_splits):
                oos_start = is_end
                oos_end = oos_start + oos_size
                if oos_end > total_bars:
                    break
                folds.append((idx + 1, 0, is_end, oos_start, oos_end))
                is_end = oos_end
        else:
            step = is_size + oos_size
            for idx in range(self.wf_config.n_splits):
                is_start = idx * step
                is_end = is_start + is_size
                oos_start = is_end
                oos_end = oos_start + oos_size
                if oos_end > total_bars:
                    break
                folds.append((idx + 1, is_start, is_end, oos_start, oos_end))
        return folds

    @staticmethod
    def _average_metrics(items: list[dict[str, float]]) -> dict[str, float]:
        metric_values: dict[str, list[float]] = {}
        for metrics in items:
            if not metrics:
                continue
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                metric_values.setdefault(key, []).append(float(value))
        return {key: _mean(values) for key, values in metric_values.items()}

    def _calc_overfitting(self, folds: list[WalkForwardFold]) -> float:
        values: list[float] = []
        for fold in folds:
            is_value = fold.is_report.performance.get(self.metric)
            if is_value is None:
                continue
            oos_value = fold.oos_report.performance.get(self.metric)
            if oos_value is None:
                continue
            if not isinstance(is_value, (int, float)) or not isinstance(oos_value, (int, float)):
                continue
            if abs(float(is_value)) < 1e-12:
                continue
            values.append(float(oos_value) / float(is_value))
        return _mean(values)
