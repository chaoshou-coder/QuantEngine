from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from quantengine.data.loader import DataBundle
from quantengine.engine.backtest import BacktestEngine
from quantengine.strategy.base import BaseStrategy, ParameterSpace

from .base import OptimizationResult, Optimizer, TrialResult, score_from_report
from .grid import _pick_best
from .random_search import RandomSearchOptimizer

logger = logging.getLogger(__name__)

try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    optuna = None


class BayesianOptimizer(Optimizer):
    method = "bayesian"

    def __init__(
        self,
        engine: BacktestEngine,
        data: DataBundle,
        strategy_factory: Callable[[], BaseStrategy],
        n_trials: int = 200,
        metric: str = "sharpe",
        maximize: bool = True,
        random_seed: int = 42,
        max_workers: int | None = None,
        show_progress: bool = False,
    ):
        self.engine = engine
        self.data = data
        self.strategy_factory = strategy_factory
        self.n_trials = max(1, int(n_trials))
        self.metric = metric
        self.maximize = maximize
        self.random_seed = random_seed
        self.max_workers = max_workers
        self.show_progress = show_progress

    def optimize(self) -> OptimizationResult:
        if optuna is None:
            logger.warning("optuna 未安装，贝叶斯优化降级为随机搜索")
            fallback = RandomSearchOptimizer(
                engine=self.engine,
                data=self.data,
                strategy_factory=self.strategy_factory,
                n_trials=self.n_trials,
                metric=self.metric,
                maximize=self.maximize,
                random_seed=self.random_seed,
                max_workers=self.max_workers,
                show_progress=self.show_progress,
            )
            return fallback.optimize()

        spaces = self.strategy_factory().parameters()
        direction = "maximize" if self.maximize else "minimize"
        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        study = optuna.create_study(direction=direction, sampler=sampler)
        trial_results: list[TrialResult] = []

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
            task = progress.add_task("bayesian", total=self.n_trials)
            callbacks = []
        else:
            callbacks = []

        def _callback(study: Any, trial: Any) -> None:
            del study
            if task is not None and progress is not None:
                progress.update(task, advance=1)
                if trial.state and trial.value is not None:
                    progress.update(task, description=f"bayesian score={trial.value:.6f}")

        if self.show_progress:
            callbacks.append(_callback)

        def objective(trial):
            params = _suggest_params(trial, spaces)
            strategy = self.strategy_factory()
            report = self.engine.run(self.data, strategy, params, record_trades=False)
            score = score_from_report(report, self.metric)
            trial_results.append(TrialResult(params=params, score=score, report=report))
            return score

        try:
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.max_workers or 1, callbacks=callbacks)
        finally:
            if progress is not None:
                progress.stop()

        best_trial = _pick_best(trial_results, maximize=self.maximize)
        return OptimizationResult(
            method=self.method,
            metric=self.metric,
            maximize=self.maximize,
            best_params=best_trial.params,
            best_score=best_trial.score,
            best_report=best_trial.report,
            trials=trial_results,
        )


def _suggest_params(trial, spaces: dict[str, ParameterSpace]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, space in spaces.items():
        if space.kind == "int":
            params[key] = trial.suggest_int(key, int(space.low), int(space.high), step=int(space.step or 1))
        elif space.kind == "float":
            params[key] = trial.suggest_float(key, float(space.low), float(space.high), step=float(space.step or 0.1))
        else:
            params[key] = trial.suggest_categorical(key, list(space.choices or []))
    return params
