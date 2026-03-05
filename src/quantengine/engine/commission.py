from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from quantengine.data.gpu_backend import xp_from_array


class CommissionModel(ABC):
    @abstractmethod
    def compute(self, notional: float, quantity: float = 1.0) -> float:
        """计算单笔交易手续费。"""

    @abstractmethod
    def compute_vector(self, notional: Any, quantity: Any) -> Any:
        """计算向量化手续费。"""


@dataclass
class FixedCommission(CommissionModel):
    value: float = 0.0

    def compute(self, notional: float, quantity: float = 1.0) -> float:
        return float(self.value * max(quantity, 0.0))

    def compute_vector(self, notional: Any, quantity: Any) -> Any:
        xp = xp_from_array(quantity)
        return self.value * xp.maximum(quantity, 0.0)


@dataclass
class PercentCommission(CommissionModel):
    rate: float = 0.0

    def compute(self, notional: float, quantity: float = 1.0) -> float:
        return float(max(notional, 0.0) * self.rate)

    def compute_vector(self, notional: Any, quantity: Any) -> Any:
        xp = xp_from_array(notional)
        return xp.maximum(notional, 0.0) * self.rate


@dataclass
class TieredCommission(CommissionModel):
    tiers: list[tuple[float, float]] = field(default_factory=list)
    fallback_rate: float = 0.0

    def __post_init__(self) -> None:
        self.tiers = sorted(self.tiers, key=lambda item: item[0])

    def _rate_for_notional(self, notional: float) -> float:
        for threshold, rate in self.tiers:
            if notional <= threshold:
                return rate
        return self.tiers[-1][1] if self.tiers else self.fallback_rate

    def compute(self, notional: float, quantity: float = 1.0) -> float:
        rate = self._rate_for_notional(max(notional, 0.0))
        return float(max(notional, 0.0) * rate)

    def compute_vector(self, notional: Any, quantity: Any) -> Any:
        xp = xp_from_array(notional)
        if not self.tiers:
            return xp.maximum(notional, 0.0) * self.fallback_rate
        out = xp.zeros_like(notional, dtype=float)
        remaining = xp.ones_like(notional, dtype=bool)
        sorted_tiers = list(self.tiers)
        for threshold, rate in sorted_tiers:
            mask = remaining & (notional <= threshold)
            out = xp.where(mask, notional * rate, out)
            remaining = remaining & (~mask)
        out = xp.where(remaining, notional * sorted_tiers[-1][1], out)
        return xp.maximum(out, 0.0)


def build_commission(model: str, value: float, tiers: list[tuple[float, float]] | None = None) -> CommissionModel:
    normalized = model.strip().lower()
    if normalized == "fixed":
        return FixedCommission(value=value)
    if normalized == "percent":
        return PercentCommission(rate=value)
    if normalized == "tiered":
        return TieredCommission(tiers=tiers or [])
    raise ValueError(f"未知手续费模型: {model}")
