from __future__ import annotations

import random
import logging
from typing import Callable
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from quantengine.engine.backtest import BacktestEngine
from quantengine.data.loader import DataBundle
from quantengine.strategy.base import BaseStrategy

from .base import OptimizationResult, Optimizer, TrialResult
from .grid import _pick_best
from .parallel import build_signal_tensor, evaluate_batch, evaluate_signal_tensor

logger = logging.getLogger(__name__)

class RandomSearchOptimizer(Optimizer):
    method = "random"

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
        batch_size: int = 128,
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
        self.batch_size = max(1, int(batch_size))
        self.show_progress = show_progress

    def optimize(self) -> OptimizationResult:
        rng = random.Random(self.random_seed)
        spaces = self.strategy_factory().parameters()
        params_list: list[dict] = []
        for _ in range(self.n_trials):
            trial_params = {}
            for key, space in spaces.items():
                trial_params[key] = space.sample(rng)
            params_list.append(trial_params)

        all_trials: list[TrialResult] = []
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
            task = progress.add_task("random search", total=len(params_list))

        try:
            for start in range(0, len(params_list), self.batch_size):
                batch = params_list[start : start + self.batch_size]
                if self.data.backend.active == "gpu":
                    strategy = self.strategy_factory()
                    tensor = build_signal_tensor(strategy, self.data, batch)
                    batch_trials = evaluate_signal_tensor(
                        engine=self.engine,
                        data=self.data,
                        params_list=batch,
                        signal_tensor=tensor,
                        metric=self.metric,
                    )
                else:
                    batch_trials = evaluate_batch(
                        engine=self.engine,
                        data=self.data,
                        strategy_factory=self.strategy_factory,
                        params_list=batch,
                        metric=self.metric,
                    )
                all_trials.extend(batch_trials)
                if task is not None and progress is not None:
                    progress.update(task, advance=len(batch_trials))
        finally:
            if progress is not None:
                progress.stop()

        best_trial = _pick_best(all_trials, maximize=self.maximize)
        return OptimizationResult(
            method=self.method,
            metric=self.metric,
            maximize=self.maximize,
            best_params=best_trial.params,
            best_score=best_trial.score,
            best_report=best_trial.report,
            trials=all_trials,
        )
