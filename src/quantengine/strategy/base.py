from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from quantengine.data import DataBundle
from quantengine.strategy.signal import SignalArray

ParameterType = Literal["int", "float", "choice"]


@dataclass(frozen=True)
class ParameterSpace:
    kind: ParameterType
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: list[Any] | None = None

    def grid_values(self) -> list[Any]:
        if self.kind == "choice":
            return list(self.choices or [])
        if self.low is None or self.high is None:
            return []
        step = self.step or (1 if self.kind == "int" else 0.1)
        if float(step) <= 0:
            raise ValueError("step 必须大于 0")
        if self.kind == "int":
            return list(range(int(self.low), int(self.high) + 1, int(step)))
        return np.arange(float(self.low), float(self.high) + float(step) * 0.5, float(step)).tolist()

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "choice":
            if not self.choices:
                raise ValueError("choices 不能为空")
            return rng.choice(self.choices)
        if self.low is None or self.high is None:
            raise ValueError("low/high 不能为空")
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        return rng.uniform(float(self.low), float(self.high))


def cartesian_from_spaces(spaces: dict[str, ParameterSpace]) -> list[dict[str, Any]]:
    keys = list(spaces.keys())
    values = [spaces[key].grid_values() for key in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append({key: value for key, value in zip(keys, combo, strict=True)})
    return combos


class BaseStrategy(ABC):
    name: str = "base_strategy"

    @abstractmethod
    def parameters(self) -> dict[str, ParameterSpace]:
        """声明可优化参数空间。"""

    @abstractmethod
    def generate_signals(self, data: DataBundle, params: dict[str, Any]) -> SignalArray:
        """根据数据和参数输出信号矩阵，shape=(n_bars, n_assets)。"""

    def on_init(self, engine: Any) -> None:  # noqa: B027
        """可选钩子：用于策略初始化时调整交易模型。"""
