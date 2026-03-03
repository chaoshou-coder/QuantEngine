from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .rules import TradingRules
from .slippage import SlippageModel


OrderType = Literal["market", "limit"]


@dataclass(frozen=True)
class Order:
    symbol: str
    side: float
    quantity: float
    order_type: OrderType = "market"
    limit_price: float | None = None


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: float
    quantity: float
    fill_price: float
    filled: bool
    reason: str = "ok"


def execute_market_order(
    order: Order,
    next_open: float,
    prev_close: float,
    slippage: SlippageModel,
    rules: TradingRules | None = None,
    bar_volume: float | None = None,
) -> Fill:
    price = slippage.adjust_price(next_open, side=order.side, quantity=order.quantity, bar_volume=bar_volume)
    if rules is not None:
        price = rules.apply_price_limit(price, prev_close=prev_close)
    return Fill(
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        fill_price=price,
        filled=True,
    )


def execute_limit_order(
    order: Order,
    next_open: float,
    next_high: float,
    next_low: float,
    prev_close: float,
    slippage: SlippageModel,
    rules: TradingRules | None = None,
    bar_volume: float | None = None,
) -> Fill:
    if order.limit_price is None:
        raise ValueError("limit order 需要 limit_price")

    if order.side >= 0:
        touched = next_low <= order.limit_price
    else:
        touched = next_high >= order.limit_price

    if not touched:
        return Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=0.0,
            filled=False,
            reason="limit_not_touched",
        )

    expected = order.limit_price
    slipped = slippage.adjust_price(expected, side=order.side, quantity=order.quantity, bar_volume=bar_volume)
    if rules is not None:
        slipped = rules.apply_price_limit(slipped, prev_close=prev_close)
    return Fill(
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        fill_price=slipped,
        filled=True,
    )
