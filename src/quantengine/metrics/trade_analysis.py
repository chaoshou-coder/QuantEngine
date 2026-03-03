from __future__ import annotations

from collections import defaultdict
from typing import Any


def _close_positions(
    quantity: float,
    price: float,
    book: list[tuple[float, float]],
    pnl_sign: float,
    realized_pnl: list[float],
) -> float:
    remaining = quantity
    while book and remaining > 0.0:
        open_qty, open_price = book[0]
        close_qty = min(open_qty, remaining)
        realized_pnl.append((price - open_price) * close_qty * pnl_sign)
        remaining -= close_qty
        if close_qty >= open_qty:
            book.pop(0)
        else:
            book[0] = (open_qty - close_qty, open_price)
    return remaining


def calculate_trade_metrics(trades: list[dict[str, Any]]) -> dict[str, float | str]:
    if not trades:
        return {
            "trade_count": 0.0,
            "buy_count": 0.0,
            "sell_count": 0.0,
            "avg_trade_cost": 0.0,
            "avg_trade_size": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "max_consecutive_losses": 0.0,
            "symbol_count": 0.0,
            "most_active_symbol": "",
        }

    buy_count = 0
    sell_count = 0
    total_cost = 0.0
    total_size = 0.0
    symbol_set = set()
    per_symbol = defaultdict(int)
    realized_pnl: list[float] = []
    long_book: dict[str, list[tuple[float, float]]] = defaultdict(list)
    short_book: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for trade in trades:
        side = str(trade.get("side", "")).upper()
        symbol = str(trade.get("symbol", "")).strip()
        if not symbol:
            continue
        quantity = float(trade.get("quantity", 0.0))
        price = float(trade.get("price", 0.0))
        if quantity <= 0.0:
            continue

        total_cost += float(trade.get("cost", 0.0))
        total_size += quantity
        per_symbol[symbol] += 1
        symbol_set.add(symbol)

        if side == "BUY":
            buy_count += 1
            remaining = _close_positions(quantity, price, short_book[symbol], -1.0, realized_pnl)
            if remaining > 0.0:
                long_book[symbol].append((remaining, price))
        elif side == "SELL":
            sell_count += 1
            remaining = _close_positions(quantity, price, long_book[symbol], 1.0, realized_pnl)
            if remaining > 0.0:
                short_book[symbol].append((remaining, price))

    if not realized_pnl:
        return {
            "trade_count": 0.0,
            "buy_count": float(buy_count),
            "sell_count": float(sell_count),
            "avg_trade_cost": total_cost / len(trades),
            "avg_trade_size": total_size / len(trades),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "max_consecutive_losses": 0.0,
            "symbol_count": float(len(symbol_set)),
            "most_active_symbol": max(per_symbol.items(), key=lambda item: item[1])[0] if per_symbol else "",
        }

    win_count = 0
    gross_profit = 0.0
    gross_loss = 0.0
    max_consecutive_losses = 0
    current_loss_streak = 0
    for pnl in realized_pnl:
        if pnl > 0:
            win_count += 1
            gross_profit += pnl
            current_loss_streak = 0
        elif pnl < 0:
            gross_loss += -pnl
            current_loss_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
        else:
            current_loss_streak = 0

    positives = [pnl for pnl in realized_pnl if pnl > 0]
    negatives = [abs(pnl) for pnl in realized_pnl if pnl < 0]

    return {
        "trade_count": float(len(realized_pnl)),
        "buy_count": float(buy_count),
        "sell_count": float(sell_count),
        "avg_trade_cost": total_cost / len(realized_pnl),
        "avg_trade_size": total_size / len(realized_pnl),
        "win_rate": float(win_count) / float(len(realized_pnl)),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "avg_profit": sum(positives) / len(positives) if positives else 0.0,
        "avg_loss": sum(negatives) / len(negatives) if negatives else 0.0,
        "max_consecutive_losses": float(max_consecutive_losses),
        "symbol_count": float(len(symbol_set)),
        "most_active_symbol": max(per_symbol.items(), key=lambda item: item[1])[0] if per_symbol else "",
    }
