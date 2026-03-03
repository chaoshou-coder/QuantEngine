from __future__ import annotations

import random
import logging
from typing import Any, Callable
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from quantengine.engine.backtest import BacktestEngine
from quantengine.data.loader import DataBundle
from quantengine.strategy.base import BaseStrategy, ParameterSpace

from .base import OptimizationResult, Optimizer, TrialResult, score_from_report
from .grid import _pick_best
from .random_search import RandomSearchOptimizer

logger = logging.getLogger(__name__)

try:
    import nevergrad as ng  # type: ignore
except Exception:  # pragma: no cover
    ng = None

try:
    from deap import algorithms, base, creator, tools  # type: ignore
except Exception:  # pragma: no cover
    algorithms = None
    base = None
    creator = None
    tools = None


class GeneticOptimizer(Optimizer):
    method = "genetic"

    def __init__(
        self,
        engine: BacktestEngine,
        data: DataBundle,
        strategy_factory: Callable[[], BaseStrategy],
        n_generations: int = 20,
        population_size: int = 50,
        metric: str = "sharpe",
        maximize: bool = True,
        random_seed: int = 42,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        show_progress: bool = False,
    ):
        self.engine = engine
        self.data = data
        self.strategy_factory = strategy_factory
        self.n_generations = max(1, int(n_generations))
        self.population_size = max(2, int(population_size))
        self.metric = metric
        self.maximize = maximize
        self.random_seed = random_seed
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.show_progress = show_progress

    def optimize(self) -> OptimizationResult:
        spaces = self.strategy_factory().parameters()
        if ng is not None:
            trials = self._optimize_with_nevergrad(spaces)
        elif base is not None and creator is not None and tools is not None and algorithms is not None:
            trials = self._optimize_with_deap(spaces)
        else:
            logger.warning("遗传优化缺少 nevergrad/deap 依赖，已降级为随机搜索")
            fallback = RandomSearchOptimizer(
                engine=self.engine,
                data=self.data,
                strategy_factory=self.strategy_factory,
                n_trials=self.n_generations * self.population_size,
                metric=self.metric,
                maximize=self.maximize,
                random_seed=self.random_seed,
                show_progress=self.show_progress,
            )
            return fallback.optimize()

        best_trial = _pick_best(trials, maximize=self.maximize)
        return OptimizationResult(
            method=self.method,
            metric=self.metric,
            maximize=self.maximize,
            best_params=best_trial.params,
            best_score=best_trial.score,
            best_report=best_trial.report,
            trials=trials,
        )

    def _evaluate(self, params: dict[str, Any]) -> TrialResult:
        strategy = self.strategy_factory()
        report = self.engine.run(self.data, strategy, params, record_trades=False)
        score = score_from_report(report, self.metric)
        return TrialResult(params=params, score=score, report=report)

    def _optimize_with_nevergrad(self, spaces: dict[str, ParameterSpace]) -> list[TrialResult]:
        instr_kwargs = {}
        for key, space in spaces.items():
            instr_kwargs[key] = _nevergrad_param(space)
        instrumentation = ng.p.Instrumentation(**instr_kwargs)
        budget = self.n_generations * self.population_size
        optimizer = ng.optimizers.TwoPointsDE(
            parametrization=instrumentation,
            budget=budget,
            num_workers=1,
        )
        trials: list[TrialResult] = []
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
            task = progress.add_task("genetic", total=budget)
        for _ in range(budget):
            candidate = optimizer.ask()
            _, kwargs = candidate.args, candidate.kwargs
            params = dict(kwargs)
            trial = self._evaluate(params)
            objective = trial.score if self.maximize else -trial.score
            optimizer.tell(candidate, -objective)
            trials.append(trial)
            if task is not None and progress is not None:
                progress.update(task, advance=1, description=f"genetic score={trial.score:.6f}")
        if progress is not None:
            progress.stop()
        return trials

    def _optimize_with_deap(self, spaces: dict[str, ParameterSpace]) -> list[TrialResult]:
        rng = random.Random(self.random_seed)
        keys = list(spaces.keys())

        fitness_name = f"FitnessBT_{id(self)}"
        individual_name = f"IndividualBT_{id(self)}"
        if fitness_name in creator.__dict__:
            del creator.__dict__[fitness_name]
        if individual_name in creator.__dict__:
            del creator.__dict__[individual_name]

        weights = (1.0,) if self.maximize else (-1.0,)
        creator.create(fitness_name, base.Fitness, weights=weights)
        creator.create(individual_name, list, fitness=creator.__dict__[fitness_name])

        toolbox = base.Toolbox()
        toolbox.register("individual", _build_individual, individual_name, keys, spaces, rng)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", _mutate_individual, keys, spaces, rng)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", _evaluate_individual, keys, self._evaluate)

        population = toolbox.population(n=self.population_size)
        hall = tools.HallOfFame(1)
        trials: list[TrialResult] = []
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
            task = progress.add_task("genetic", total=self.n_generations * self.population_size)

        for _ in range(self.n_generations):
            offspring = algorithms.varAnd(
                population,
                toolbox,
                cxpb=self.crossover_rate,
                mutpb=self.mutation_rate,
            )
            fitnesses = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses, strict=True):
                ind.fitness.values = fit[0]
                trials.append(fit[1])
                if task is not None and progress is not None:
                    progress.update(task, advance=1)
            population = toolbox.select(offspring, k=self.population_size)
            hall.update(population)
        if progress is not None:
            progress.stop()

        return trials


def _nevergrad_param(space: ParameterSpace):
    if space.kind == "int":
        return ng.p.Scalar(lower=int(space.low), upper=int(space.high)).set_integer_casting()
    if space.kind == "float":
        return ng.p.Scalar(lower=float(space.low), upper=float(space.high))
    return ng.p.Choice(list(space.choices or []))


def _build_individual(individual_name: str, keys, spaces, rng):
    values = []
    for key in keys:
        values.append(spaces[key].sample(rng))
    return creator.__dict__[individual_name](values)


def _mutate_individual(keys, spaces, rng, individual):
    idx = rng.randint(0, len(keys) - 1)
    key = keys[idx]
    individual[idx] = spaces[key].sample(rng)
    return (individual,)


def _evaluate_individual(keys, evaluator, individual):
    params = {key: individual[idx] for idx, key in enumerate(keys)}
    trial = evaluator(params)
    return (trial.score,), trial
