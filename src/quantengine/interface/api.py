from __future__ import annotations

from pathlib import Path
from typing import Any

from quantengine.config import QuantEngineConfig, load_config
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
from quantengine.strategy import get_strategy
from quantengine.visualization import write_backtest_report_html, write_optimization_report_html


class QuantEngineAPI:
    def __init__(self, config_path: str | Path | None = None):
        self.config: QuantEngineConfig = load_config(config_path)

    def backtest(
        self,
        strategy_name: str,
        data_path: str | Path,
        params: dict[str, Any],
        symbols: list[str] | None = None,
    ):
        loader = DataLoader(
            backend=self.config.runtime.backend,
            use_gpu=self.config.runtime.use_gpu,
        )
        data = loader.load(path=data_path, symbols=symbols)
        engine = self._build_engine()
        strategy = get_strategy(strategy_name)
        return engine.run(data=data, strategy=strategy, params=params)

    def backtest_multi(
        self,
        strategy_specs: list[tuple[str, dict[str, Any], float]],
        data_path: str | Path,
        symbols: list[str] | None = None,
    ):
        loader = DataLoader(
            backend=self.config.runtime.backend,
            use_gpu=self.config.runtime.use_gpu,
        )
        data = loader.load(path=data_path, symbols=symbols)
        engine = self._build_engine()
        strategies = [(get_strategy(name), params, weight) for name, params, weight in strategy_specs]
        return engine.run_multi_strategy(data=data, strategies=strategies)

    def optimize(
        self,
        strategy_name: str,
        data_path: str | Path,
        method: str,
        n_trials: int | None = None,
        metric: str | None = None,
        maximize: bool | None = None,
        symbols: list[str] | None = None,
        show_progress: bool = False,
    ):
        loader = DataLoader(
            backend=self.config.runtime.backend,
            use_gpu=self.config.runtime.use_gpu,
        )
        data = loader.load(path=data_path, symbols=symbols)
        engine = self._build_engine()
        strategy_factory = lambda: get_strategy(strategy_name)
        metric_name = metric or self.config.optimize.metric
        maximize_metric = self.config.optimize.maximize if maximize is None else maximize
        trials = n_trials or self.config.optimize.n_trials

        normalized = method.strip().lower()
        if normalized == "grid":
            optimizer = GridSearchOptimizer(
                engine=engine,
                data=data,
                strategy_factory=strategy_factory,
                metric=metric_name,
                maximize=maximize_metric,
                max_workers=self.config.optimize.max_workers,
                batch_size=self.config.optimize.batch_size,
                show_progress=show_progress,
            )
        elif normalized == "random":
            optimizer = RandomSearchOptimizer(
                engine=engine,
                data=data,
                strategy_factory=strategy_factory,
                n_trials=trials,
                metric=metric_name,
                maximize=maximize_metric,
                random_seed=self.config.optimize.random_seed,
                max_workers=self.config.optimize.max_workers,
                batch_size=self.config.optimize.batch_size,
                show_progress=show_progress,
            )
        elif normalized == "bayesian":
            optimizer = BayesianOptimizer(
                engine=engine,
                data=data,
                strategy_factory=strategy_factory,
                n_trials=trials,
                metric=metric_name,
                maximize=maximize_metric,
                random_seed=self.config.optimize.random_seed,
                max_workers=self.config.optimize.max_workers,
                show_progress=show_progress,
            )
        elif normalized == "genetic":
            optimizer = GeneticOptimizer(
                engine=engine,
                data=data,
                strategy_factory=strategy_factory,
                n_generations=max(1, trials // max(2, self.config.optimize.batch_size)),
                population_size=max(4, self.config.optimize.batch_size),
                metric=metric_name,
                maximize=maximize_metric,
                random_seed=self.config.optimize.random_seed,
                show_progress=show_progress,
            )
        else:
            raise ValueError(f"未知优化方法: {method}")
        return optimizer.optimize()

    def walk_forward(
        self,
        strategy_name: str,
        data_path: str | Path,
        n_splits: int = 5,
        in_sample_ratio: float = 0.7,
        method: str = "bayesian",
        n_trials: int | None = None,
        metric: str | None = None,
        maximize: bool | None = None,
        symbols: list[str] | None = None,
        expanding: bool = False,
        show_progress: bool = False,
    ):
        loader = DataLoader(
            backend=self.config.runtime.backend,
            use_gpu=self.config.runtime.use_gpu,
        )
        data = loader.load(path=data_path, symbols=symbols)
        engine = self._build_engine()
        strategy_factory = lambda: get_strategy(strategy_name)
        metric_name = metric or self.config.optimize.metric
        maximize_metric = self.config.optimize.maximize if maximize is None else maximize
        trials = n_trials or self.config.optimize.n_trials

        analyzer = WalkForwardAnalyzer(
            engine=engine,
            data=data,
            strategy_factory=strategy_factory,
            wf_config=WalkForwardConfig(
                n_splits=n_splits,
                in_sample_ratio=in_sample_ratio,
                optimization_method=method,
                n_trials=trials,
                expanding=expanding,
            ),
            metric=metric_name,
            maximize=maximize_metric,
            show_progress=show_progress,
        )
        return analyzer.run()

    def generate_backtest_report(self, report, output_file: str | Path):
        return write_backtest_report_html(
            report,
            timestamps=report.metadata.get("timestamps", []),
            output_path=output_file,
        )

    def generate_optimization_report(self, result, output_file: str | Path):
        return write_optimization_report_html(result, output_path=output_file)

    def _build_engine(self) -> BacktestEngine:
        return build_engine(self.config)
