from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from quantengine.data.gpu_backend import xp_from_array


class SlippageModel(ABC):
    @abstractmethod
    def adjust_price(
        self,
        price: float,
        side: float,
        quantity: float = 1.0,
        bar_volume: float | None = None,
    ) -> float:
        """单笔成交价滑点调整。"""

    @abstractmethod
    def adjust_price_vector(
        self,
        price: Any,
        side: Any,
        quantity: Any,
        bar_volume: Any | None = None,
    ) -> Any:
        """向量化成交价滑点调整。"""


@dataclass
class FixedSlippage(SlippageModel):
    points: float = 0.0

    def adjust_price(self, price: float, side: float, quantity: float = 1.0, bar_volume=None) -> float:
        direction = 1.0 if side >= 0 else -1.0
        return float(price + direction * self.points)

    def adjust_price_vector(self, price: Any, side: Any, quantity: Any, bar_volume=None) -> Any:
        xp = xp_from_array(price)
        direction = xp.where(side >= 0, 1.0, -1.0)
        return price + direction * self.points


@dataclass
class PercentSlippage(SlippageModel):
    rate: float = 0.0

    def adjust_price(self, price: float, side: float, quantity: float = 1.0, bar_volume=None) -> float:
        direction = 1.0 if side >= 0 else -1.0
        return float(price * (1.0 + direction * self.rate))

    def adjust_price_vector(self, price: Any, side: Any, quantity: Any, bar_volume=None) -> Any:
        xp = xp_from_array(price)
        direction = xp.where(side >= 0, 1.0, -1.0)
        return price * (1.0 + direction * self.rate)


@dataclass
class VolumeSlippage(SlippageModel):
    impact: float = 1.0
    max_ratio: float = 0.05

    def adjust_price(
        self,
        price: float,
        side: float,
        quantity: float = 1.0,
        bar_volume: float | None = None,
    ) -> float:
        volume = max(float(bar_volume or 0.0), 1.0)
        participation = min(float(quantity) / volume, self.max_ratio)
        ratio = self.impact * participation
        direction = 1.0 if side >= 0 else -1.0
        return float(price * (1.0 + direction * ratio))

    def adjust_price_vector(
        self,
        price: Any,
        side: Any,
        quantity: Any,
        bar_volume: Any | None = None,
    ) -> Any:
        xp = xp_from_array(price)
        if bar_volume is None:
            bar_volume = xp.ones_like(price)
        safe_volume = xp.maximum(bar_volume, 1.0)
        participation = xp.minimum(quantity / safe_volume, self.max_ratio)
        ratio = self.impact * participation
        direction = xp.where(side >= 0, 1.0, -1.0)
        return price * (1.0 + direction * ratio)


def build_slippage(model: str, value: float, impact: float = 1.0) -> SlippageModel:
    normalized = model.strip().lower()
    if normalized == "fixed":
        return FixedSlippage(points=value)
    if normalized == "percent":
        return PercentSlippage(rate=value)
    if normalized == "volume":
        return VolumeSlippage(impact=impact, max_ratio=value)
    raise ValueError(f"未知滑点模型: {model}")
