from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from quantengine.engine.backtest import BacktestReport

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    params: dict[str, Any]
    score: float
    report: BacktestReport | None = None


@dataclass
class OptimizationResult:
    method: str
    metric: str
    maximize: bool
    best_params: dict[str, Any]
    best_score: float
    best_report: BacktestReport | None
    trials: list[TrialResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "metric": self.metric,
            "maximize": self.maximize,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trials": [{"params": item.params, "score": item.score} for item in self.trials],
        }


class Optimizer(ABC):
    method: str = "base"

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """执行参数搜索并返回结果。"""


def score_from_report(report: BacktestReport, metric: str) -> float:
    if metric in report.performance:
        return float(report.performance[metric])
    if metric in report.risk:
        return float(report.risk[metric])
    if metric in report.trade_metrics:
        return float(report.trade_metrics[metric])
    raise KeyError(f"未知指标: {metric}")
