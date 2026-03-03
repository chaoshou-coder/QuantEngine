from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantengine.data.gpu_backend import to_numpy
from quantengine.data.gpu_backend import xp_from_array
from quantengine.data.loader import DataBundle

from .commission import CommissionModel
from .rules import TradingRules
from .slippage import SlippageModel
from .commission import FixedCommission, PercentCommission, TieredCommission
from .slippage import FixedSlippage, PercentSlippage, VolumeSlippage

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

@dataclass
class PortfolioResult:
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    turnover: np.ndarray
    trades: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity_curve": self.equity_curve.tolist(),
            "returns": self.returns.tolist(),
            "positions": self.positions.tolist(),
            "turnover": self.turnover.tolist(),
            "trades": self.trades,
            "metadata": self.metadata,
        }


def _normalize_signal_tensor(signal: Any, n_bars: int, n_assets: int, xp):
    signal_arr = xp.asarray(signal, dtype=float)
    if signal_arr.ndim == 2:
        if signal_arr.shape != (n_bars, n_assets):
            raise ValueError(f"signal shape={signal_arr.shape}, expected={(n_bars, n_assets)}")
        signal_arr = signal_arr[:, :, None]
    elif signal_arr.ndim == 3:
        if signal_arr.shape[0] != n_bars or signal_arr.shape[1] != n_assets:
            raise ValueError(
                f"signal shape={signal_arr.shape}, expected=(n_bars, n_assets, n_combos)"
            )
    else:
        raise ValueError(f"signal must be 2D or 3D array, got shape={signal_arr.shape}")
    return xp.clip(signal_arr, -1.0, 1.0)


def simulate_portfolio_batch(
    data: DataBundle,
    signal: Any,
    slippage: SlippageModel,
    commission: CommissionModel,
    rules: TradingRules | None = None,
    initial_cash: float = 1_000_000.0,
    contract_multiplier: float = 1.0,
):
    close = data.close
    n_bars, n_assets = close.shape
    xp = xp_from_array(close)
    if n_bars == 0:
        return (
            xp.empty((0, 0), dtype=float),
            xp.empty((0, 0), dtype=float),
        )
    signal_tensor = _normalize_signal_tensor(signal, n_bars=n_bars, n_assets=n_assets, xp=xp)

    if signal_tensor.ndim != 3:
        raise ValueError(f"signal tensor 维度必须为 3，当前 {signal_tensor.ndim}")
    n_combos = signal_tensor.shape[2]
    if n_combos == 0:
        return (
            xp.zeros((n_bars, 0), dtype=float),
            xp.zeros((n_bars, 0), dtype=float),
        )

    open_arr = data.open
    volume_arr = data.volume

    positions = xp.zeros((n_bars, n_assets, n_combos), dtype=float)
    positions[1:] = signal_tensor[:-1]

    equity = xp.zeros((n_bars, n_combos), dtype=float)
    returns = xp.zeros((n_bars, n_combos), dtype=float)
    equity[0] = float(initial_cash)

    if rules is not None:
        margin_ratio = float(rules.margin_ratio)
        limit_up_ratio = float(rules.limit_up_ratio or 0.0)
        limit_down_ratio = float(rules.limit_down_ratio or 0.0)
    else:
        margin_ratio = 0.0
        limit_up_ratio = 0.0
        limit_down_ratio = 0.0

    if isinstance(slippage, FixedSlippage):
        slippage_points = float(slippage.points)
        slippage_rate = 0.0
        slippage_impact = 0.0
        slippage_kind = 0
    elif isinstance(slippage, PercentSlippage):
        slippage_points = 0.0
        slippage_rate = float(slippage.rate)
        slippage_impact = 0.0
        slippage_kind = 1
    else:
        slippage_points = 0.0
        slippage_rate = 0.0
        slippage_impact = float(slippage.impact)
        slippage_kind = 2

    if isinstance(commission, FixedCommission):
        commission_kind = 0
        commission_rate = float(commission.value)
        tiers_threshold = xp.empty(0, dtype=float)
        tiers_rate = xp.empty(0, dtype=float)
    elif isinstance(commission, PercentCommission):
        commission_kind = 1
        commission_rate = float(commission.rate)
        tiers_threshold = xp.empty(0, dtype=float)
        tiers_rate = xp.empty(0, dtype=float)
    else:
        commission_kind = 2
        commission_rate = 0.0
        tiers_threshold = xp.asarray([float(item[0]) for item in commission.tiers], dtype=float)
        tiers_rate = xp.asarray([float(item[1]) for item in commission.tiers], dtype=float)
        if tiers_threshold.shape[0] != tiers_rate.shape[0]:
            tiers_threshold = xp.empty(0, dtype=float)
            tiers_rate = xp.empty(0, dtype=float)
    volume_cap = 0.05
    if isinstance(slippage, VolumeSlippage):
        volume_cap = float(slippage.max_ratio)

    for t in range(1, n_bars):
        equity_prev = equity[t - 1]
        prev_close = close[t - 1]
        open_t = open_arr[t]
        close_t = close[t]
        volume_t = volume_arr[t]
        prev_pos = positions[t - 1]
        curr_pos = positions[t]

        if margin_ratio > 0.0:
            notional_curr = xp.sum(xp.abs(curr_pos) * open_t[:, None] * contract_multiplier, axis=0)
            required = margin_ratio * notional_curr
            scale = xp.where((required > equity_prev) & (required > 0.0), equity_prev / (required + 1e-12), 1.0)
            scale = xp.where(scale < 0.0, 0.0, scale)
            curr_pos = curr_pos * scale[None, :]
            positions[t] = curr_pos

        delta = curr_pos - prev_pos
        abs_delta = xp.abs(delta)
        side = xp.where(delta >= 0.0, 1.0, -1.0)

        open_vec = open_t[:, None]
        if slippage_kind == 0:
            fill_price = open_vec + side * slippage_points
        elif slippage_kind == 1:
            fill_price = open_vec * (1.0 + side * slippage_rate)
        else:
            vol_t = volume_t[:, None]
            vol = xp.where(vol_t <= 0.0, 1.0, vol_t)
            part = abs_delta / vol
            part = xp.minimum(part, volume_cap)
            fill_price = open_vec * (1.0 + side * slippage_impact * part)

        if limit_up_ratio > 0.0:
            up = prev_close[:, None] * (1.0 + limit_up_ratio)
            fill_price = xp.minimum(fill_price, up)
        if limit_down_ratio > 0.0:
            down = prev_close[:, None] * (1.0 - limit_down_ratio)
            fill_price = xp.maximum(fill_price, down)

        slippage_cash = xp.abs(fill_price - open_vec) * abs_delta * contract_multiplier
        notional = xp.abs(fill_price) * abs_delta * contract_multiplier
        if commission_kind == 0:
            commission_cash = abs_delta * commission_rate
        elif commission_kind == 1:
            commission_cash = notional * commission_rate
        else:
            if tiers_threshold.shape[0] == 0:
                commission_cash = notional * commission_rate
            else:
                commission_cash = xp.zeros_like(notional, dtype=float)
                remain_mask = xp.ones_like(notional, dtype=bool)
                for idx in range(tiers_threshold.shape[0]):
                    threshold = tiers_threshold[idx]
                    rate = tiers_rate[idx]
                    mask = remain_mask & (notional <= threshold)
                    commission_cash = xp.where(mask, notional * rate, commission_cash)
                    remain_mask = remain_mask & (~mask)
                commission_cash = xp.where(remain_mask, notional * tiers_rate[tiers_threshold.shape[0] - 1], commission_cash)
        trade_cost = xp.sum(slippage_cash + commission_cash, axis=0)

        pnl = xp.sum(prev_pos * (open_vec - prev_close[:, None]) * contract_multiplier, axis=0)
        pnl += xp.sum(curr_pos * (close_t[:, None] - open_vec) * contract_multiplier, axis=0)

        equity_t = equity_prev + pnl - trade_cost
        equity_t = xp.where(equity_t > 0.0, equity_t, 0.0)
        equity[t] = equity_t

        base = xp.where(equity_prev > 1.0, equity_prev, 1.0)
        returns[t] = (equity_t - equity_prev) / base

    return equity, returns


def _normalize_signal(signal: Any, n_bars: int, n_assets: int):
    xp = xp_from_array(signal)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    if signal.shape != (n_bars, n_assets):
        raise ValueError(f"signal shape={signal.shape}, expected={(n_bars, n_assets)}")
    return xp.clip(signal, -1.0, 1.0)


if njit is not None:

    @njit
    def _simulate_portfolio_numba(
        open_arr: np.ndarray,
        close_arr: np.ndarray,
        volume_arr: np.ndarray,
        signal_arr: np.ndarray,
        initial_cash: float,
        contract_multiplier: float,
        margin_ratio: float,
        limit_up_ratio: float,
        limit_down_ratio: float,
        slippage_kind: int,
        slippage_points: float,
        slippage_rate: float,
        slippage_impact: float,
        commission_kind: int,
        commission_rate: float,
        volume_cap: float,
        tier_thresholds: np.ndarray,
        tier_rates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_bars, n_assets = open_arr.shape
        positions = np.zeros((n_bars, n_assets), dtype=np.float64)
        for t in range(1, n_bars):
            for a in range(n_assets):
                positions[t, a] = signal_arr[t - 1, a]

        equity = np.zeros(n_bars, dtype=np.float64)
        returns = np.zeros(n_bars, dtype=np.float64)
        turnover = np.zeros(n_bars, dtype=np.float64)
        equity[0] = initial_cash
        bankruptcy_at = -1

        for t in range(1, n_bars):
            equity_prev = equity[t - 1]
            prev_close = close_arr[t - 1]
            open_t = open_arr[t]
            close_t = close_arr[t]
            vol_t = volume_arr[t]
            curr_pos = np.empty(n_assets, dtype=np.float64)
            prev_pos = np.empty(n_assets, dtype=np.float64)
            delta = np.empty(n_assets, dtype=np.float64)
            abs_delta = np.empty(n_assets, dtype=np.float64)
            fill_price = np.empty(n_assets, dtype=np.float64)

            notional_curr = 0.0
            for a in range(n_assets):
                curr_pos[a] = positions[t, a]
                prev_pos[a] = positions[t - 1, a]
                if margin_ratio > 0:
                    notional_curr += np.abs(curr_pos[a]) * open_t[a] * contract_multiplier

            if margin_ratio > 0 and notional_curr > 0:
                required = margin_ratio * notional_curr
                if required > equity_prev:
                    scale = equity_prev / (required + 1e-12)
                    if scale < 0.0:
                        scale = 0.0
                    for a in range(n_assets):
                        curr_pos[a] *= scale
                        positions[t, a] = curr_pos[a]

            trade_cost = 0.0
            turnover_t = 0.0
            pnl = 0.0
            for a in range(n_assets):
                curr = curr_pos[a]
                prev = prev_pos[a]
                d = curr - prev
                delta[a] = d
                abs_d = np.abs(d)
                abs_delta[a] = abs_d
                turnover_t += abs_d

                if d >= 0.0:
                    fill = open_t[a] + slippage_points
                    if slippage_kind == 1:
                        fill = open_t[a] * (1.0 + slippage_rate)
                    elif slippage_kind == 2:
                        vol = vol_t[a]
                        if vol <= 0.0:
                            vol = 1.0
                        part = abs_d / vol
                        if part > volume_cap:
                            part = volume_cap
                        fill = open_t[a] * (1.0 + slippage_impact * part)
                else:
                    fill = open_t[a] - slippage_points
                    if slippage_kind == 1:
                        fill = open_t[a] * (1.0 - slippage_rate)
                    elif slippage_kind == 2:
                        vol = vol_t[a]
                        if vol <= 0.0:
                            vol = 1.0
                        part = abs_d / vol
                        if part > volume_cap:
                            part = volume_cap
                        fill = open_t[a] * (1.0 - slippage_impact * part)

                if limit_up_ratio > 0:
                    up = prev_close[a] * (1.0 + limit_up_ratio)
                    if fill > up:
                        fill = up
                if limit_down_ratio > 0:
                    down = prev_close[a] * (1.0 - limit_down_ratio)
                    if fill < down:
                        fill = down

                fill_price[a] = fill
                notional = np.abs(fill * abs_d * contract_multiplier)
                if commission_kind == 0:
                    trade_cost += abs_d * commission_rate
                elif commission_kind == 1:
                    trade_cost += notional * commission_rate
                else:
                    rate = 0.0
                    if tier_thresholds.shape[0] == 0:
                        rate = commission_rate
                    else:
                        for idx in range(tier_thresholds.shape[0]):
                            if notional <= tier_thresholds[idx]:
                                rate = tier_rates[idx]
                                break
                        if rate == 0.0:
                            rate = tier_rates[tier_thresholds.shape[0] - 1]
                    trade_cost += notional * rate

                pnl += (prev * (open_t[a] - prev_close[a]) + curr * (close_t[a] - open_t[a])) * contract_multiplier

            equity[t] = equity_prev + pnl - trade_cost
            if equity[t] < 0.0:
                equity[t] = 0.0
            if equity_prev > 1.0:
                returns[t] = (equity[t] - equity_prev) / equity_prev
            else:
                returns[t] = 0.0
            turnover[t] = turnover_t
            if bankruptcy_at < 0 and equity[t] <= 0.0:
                bankruptcy_at = t

        return equity, returns, positions, turnover


def simulate_portfolio(
    data: DataBundle,
    signal: Any,
    slippage: SlippageModel,
    commission: CommissionModel,
    rules: TradingRules | None = None,
    initial_cash: float = 1_000_000.0,
    contract_multiplier: float = 1.0,
    record_trades: bool = True,
) -> PortfolioResult:
    close = data.close
    n_bars, n_assets = close.shape
    signal = _normalize_signal(signal, n_bars=n_bars, n_assets=n_assets)
    xp = xp_from_array(close)
    trades: list[dict[str, Any]] = []

    use_cpu_fastpath = (
        xp is np
        and not record_trades
        and njit is not None
        and isinstance(rules, TradingRules)
        and isinstance(slippage, (FixedSlippage, PercentSlippage, VolumeSlippage))
        and isinstance(commission, (FixedCommission, PercentCommission, TieredCommission))
    )

    if use_cpu_fastpath:
        if isinstance(rules, TradingRules):
            limit_up_ratio = float(rules.limit_up_ratio or 0.0)
            limit_down_ratio = float(rules.limit_down_ratio or 0.0)
            margin_ratio = float(rules.margin_ratio)
        else:
            limit_up_ratio = 0.0
            limit_down_ratio = 0.0
            margin_ratio = 0.0

        if isinstance(slippage, FixedSlippage):
            slippage_kind = 0
            slippage_points = float(slippage.points)
            slippage_rate = 0.0
            slippage_impact = 0.0
        elif isinstance(slippage, PercentSlippage):
            slippage_kind = 1
            slippage_points = 0.0
            slippage_rate = float(slippage.rate)
            slippage_impact = 0.0
        else:
            slippage_kind = 2
            slippage_points = 0.0
            slippage_rate = 0.0
            slippage_impact = float(slippage.impact)

        if isinstance(commission, FixedCommission):
            commission_kind = 0
            commission_rate = float(commission.value)
            tiers_threshold = np.empty(0, dtype=np.float64)
            tiers_rate = np.empty(0, dtype=np.float64)
        elif isinstance(commission, PercentCommission):
            commission_kind = 1
            commission_rate = float(commission.rate)
            tiers_threshold = np.empty(0, dtype=np.float64)
            tiers_rate = np.empty(0, dtype=np.float64)
        else:
            commission_kind = 2
            commission_rate = 0.0
            tiers_threshold = np.array([float(item[0]) for item in commission.tiers], dtype=np.float64)
            tiers_rate = np.array([float(item[1]) for item in commission.tiers], dtype=np.float64)
            if tiers_threshold.shape[0] != tiers_rate.shape[0]:
                tiers_threshold = np.empty(0, dtype=np.float64)
                tiers_rate = np.empty(0, dtype=np.float64)

        volume_cap = 0.05
        if isinstance(slippage, VolumeSlippage):
            volume_cap = float(slippage.max_ratio)

        close_np = np.asarray(close, dtype=np.float64)
        open_np = np.asarray(data.open, dtype=np.float64)
        volume_np = np.asarray(data.volume, dtype=np.float64)
        signal_np = np.asarray(signal, dtype=np.float64)
        if signal_np.ndim == 1:
            signal_np = signal_np.reshape(-1, 1)

        equity, returns, positions, turnover = _simulate_portfolio_numba(
            open_np,
            close_np,
            volume_np,
            signal_np,
            float(initial_cash),
            float(contract_multiplier),
            margin_ratio,
            limit_up_ratio,
            limit_down_ratio,
            slippage_kind,
            slippage_points,
            slippage_rate,
            slippage_impact,
            commission_kind,
            commission_rate,
            volume_cap,
            tiers_threshold,
            tiers_rate,
        )
        return PortfolioResult(
            equity_curve=equity,
            returns=returns,
            positions=positions,
            turnover=turnover,
            trades=[],
            metadata={
                "symbols": data.symbols,
                "backend": data.backend.active,
                "initial_cash": initial_cash,
                "contract_multiplier": contract_multiplier,
                "bankrupt": bool(np.any(equity <= 0.0)),
                "bankruptcy_index": int(np.where(equity <= 0.0)[0][0]) if np.any(equity <= 0.0) else None,
            },
        )

    positions = xp.zeros_like(signal, dtype=float)
    positions[1:] = signal[:-1]

    equity = xp.zeros(n_bars, dtype=float)
    returns = xp.zeros(n_bars, dtype=float)
    turnover = xp.zeros(n_bars, dtype=float)
    equity[0] = float(initial_cash)

    bankruptcy_at: int | None = None
    for t in range(1, n_bars):
        prev_pos = positions[t - 1]
        curr_pos = positions[t]

        if rules is not None:
            notional_curr = xp.sum(xp.abs(curr_pos) * data.open[t] * contract_multiplier)
            required = rules.margin_ratio * notional_curr
            if required > equity[t - 1]:
                scale = float(equity[t - 1] / (required + 1e-12))
                curr_pos = curr_pos * max(scale, 0.0)
                positions[t] = curr_pos

        delta = curr_pos - prev_pos
        abs_delta = xp.abs(delta)
        turnover[t] = xp.sum(abs_delta)

        side = xp.where(delta >= 0.0, 1.0, -1.0)
        fill_price = slippage.adjust_price_vector(
            data.open[t],
            side=side,
            quantity=abs_delta,
            bar_volume=data.volume[t],
        )
        if rules is not None:
            fill_price = rules.apply_price_limit_vector(fill_price, prev_close=data.close[t - 1])

        slippage_cash = xp.abs(fill_price - data.open[t]) * abs_delta * contract_multiplier
        notional = xp.abs(fill_price) * abs_delta * contract_multiplier
        commission_cash = commission.compute_vector(notional=notional, quantity=abs_delta)
        trade_cost = xp.sum(slippage_cash + commission_cash)

        pnl = xp.sum(prev_pos * contract_multiplier * (data.open[t] - data.close[t - 1]))
        pnl += xp.sum(curr_pos * contract_multiplier * (data.close[t] - data.open[t]))
        equity[t] = max(float(equity[t - 1] + pnl - trade_cost), 0.0)
        equity_prev = float(equity[t - 1])
        base = equity_prev if equity_prev > 1.0 else 1.0
        returns[t] = (equity[t] - equity_prev) / base
        if bankruptcy_at is None and equity[t] <= 0.0:
            bankruptcy_at = t

        if record_trades:
            delta_np = to_numpy(delta)
            fill_np = to_numpy(fill_price)
            cost_np = to_numpy(commission_cash + slippage_cash)
            for idx, qty_delta in enumerate(delta_np):
                if abs(float(qty_delta)) < 1e-12:
                    continue
                ts = data.timestamps[t]
                trades.append(
                    {
                        "timestamp": str(ts),
                        "symbol": data.symbols[idx],
                        "side": "BUY" if qty_delta > 0 else "SELL",
                        "quantity": float(abs(qty_delta)),
                        "price": float(fill_np[idx]),
                        "cost": float(cost_np[idx]),
                    }
                )

    return PortfolioResult(
        equity_curve=to_numpy(equity),
        returns=to_numpy(returns),
        positions=to_numpy(positions),
        turnover=to_numpy(turnover),
        trades=trades,
        metadata={
            "symbols": data.symbols,
            "backend": data.backend.active,
            "initial_cash": initial_cash,
            "contract_multiplier": contract_multiplier,
            "bankrupt": bankruptcy_at is not None,
            "bankruptcy_index": bankruptcy_at,
        },
    )
