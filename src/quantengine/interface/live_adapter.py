from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

OrderSide = Literal["BUY", "SELL"]
OrderKind = Literal["MARKET", "LIMIT"]


@dataclass(frozen=True)
class LiveOrder:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderKind = "MARKET"
    limit_price: float | None = None


@dataclass(frozen=True)
class LiveFill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    avg_price: float
    status: str


class LiveAdapter(ABC):
    """实盘对接抽象接口，供后续接入交易网关。"""

    @abstractmethod
    def connect(self) -> None:
        """建立会话连接。"""

    @abstractmethod
    def disconnect(self) -> None:
        """关闭会话连接。"""

    @abstractmethod
    def submit_order(self, order: LiveOrder) -> str:
        """下单并返回 order_id。"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """撤单。"""

    @abstractmethod
    def query_fill(self, order_id: str) -> LiveFill | None:
        """查询成交状态。"""
