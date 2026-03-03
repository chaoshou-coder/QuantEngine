from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IndicatorResult:
    """指标计算结果的统一返回结构。"""

    values: Any
    name: str


class Indicator(ABC):
    """指标接口。当前主要用于策略或外部接入自定义指标扩展。"""

    @abstractmethod
    def compute(self, series: Any, **kwargs: Any) -> IndicatorResult:
        """计算指标。"""

