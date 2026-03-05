from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantengine.data.gpu_backend import to_numpy, xp_from_array
from quantengine.data.loader import DataBundle

from .commission import CommissionModel, FixedCommission, PercentCommission, TieredCommission
from .rules import TradingRules
from .slippage import FixedSlippage, PercentSlippage, SlippageModel, VolumeSlippage

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
    risk_events: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity_curve": self.equity_curve.tolist(),
            "returns": self.returns.tolist(),
            "positions": self.positions.tolist(),
            "turnover": self.turnover.tolist(),
            "trades": self.trades,
            "risk_events": self.risk_events,
            "metadata": self.metadata,
        }


def _calendar_day_ids(timestamps: Any) -> np.ndarray:
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if ts.size == 0:
        return np.empty(0, dtype=np.int64)
    return ts.astype("datetime64[D]").astype(np.int64)


def _calendar_week_ids(timestamps: Any) -> np.ndarray:
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if ts.size == 0:
        return np.empty(0, dtype=np.int64)
    return ts.astype("datetime64[W]").astype(np.int64)


def _drawdown_action_code(action: str) -> int:
    normalized = (action or "stop").strip().lower()
    if normalized == "reduce":
        return 1
    if normalized == "alert":
        return 2
    return 0


def _collect_risk_events(
    daily_flags: np.ndarray,
    weekly_flags: np.ndarray,
    drawdown_flags: np.ndarray,
    position_flags: np.ndarray,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for bar, flag in enumerate(daily_flags):
        if int(flag) > 0:
            events.append({"bar": int(bar), "type": "daily_loss_breach", "detail": "daily loss limit breached"})
    for bar, flag in enumerate(weekly_flags):
        if int(flag) > 0:
            events.append({"bar": int(bar), "type": "weekly_loss_breach", "detail": "weekly loss limit breached"})
    for bar, flag in enumerate(drawdown_flags):
        if int(flag) > 0:
            events.append({"bar": int(bar), "type": "drawdown_breach", "detail": "drawdown guardrail triggered"})
    for bar, flag in enumerate(position_flags):
        if int(flag) > 0:
            events.append({"bar": int(bar), "type": "position_limit", "detail": "position adjusted by risk rules"})
    events.sort(key=lambda item: (int(item["bar"]), str(item["type"])))
    return events


def _normalize_signal_tensor(signal: Any, n_bars: int, n_assets: int, xp):
    signal_arr = xp.asarray(signal, dtype=float)
    if signal_arr.ndim == 2:
        if signal_arr.shape != (n_bars, n_assets):
            raise ValueError(f"signal shape={signal_arr.shape}, expected={(n_bars, n_assets)}")
        signal_arr = signal_arr[:, :, None]
    elif signal_arr.ndim == 3:
        if signal_arr.shape[0] != n_bars or signal_arr.shape[1] != n_assets:
            raise ValueError(f"signal shape={signal_arr.shape}, expected=(n_bars, n_assets, n_combos)")
    else:
        raise ValueError(f"signal must be 2D or 3D array, got shape={signal_arr.shape}")
    return xp.clip(signal_arr, -1.0, 1.0)


def _running_min_accumulate(values: Any, xp):
    try:
        return xp.minimum.accumulate(values, axis=0)
    except Exception:
        values_np = np.asarray(to_numpy(values), dtype=np.float64)
        mins_np = np.minimum.accumulate(values_np, axis=0)
        return xp.asarray(mins_np, dtype=float)


def _batch_sim_no_margin(
    open_arr: Any,
    close_arr: Any,
    volume_arr: Any,
    signal_tensor: Any,
    initial_cash: float,
    contract_multiplier: float,
    slippage_kind: int,
    slippage_points: float,
    slippage_rate: float,
    slippage_impact: float,
    volume_cap: float,
    commission_kind: int,
    commission_rate: float,
    tiers_threshold: Any,
    tiers_rate: Any,
    limit_up_ratio: float,
    limit_down_ratio: float,
):
    xp = xp_from_array(signal_tensor)
    n_bars, n_assets, n_combos = signal_tensor.shape
    equity = xp.zeros((n_bars, n_combos), dtype=float)
    returns = xp.zeros((n_bars, n_combos), dtype=float)
    if n_bars == 0 or n_combos == 0:
        return equity, returns

    equity[0] = float(initial_cash)
    if n_bars == 1:
        return equity, returns

    if n_assets == 1:
        curr_pos_2d = signal_tensor[:-1, 0, :]
        prev_pos_2d = xp.zeros_like(curr_pos_2d, dtype=float)
        if n_bars > 2:
            prev_pos_2d[1:] = signal_tensor[:-2, 0, :]

        delta_2d = curr_pos_2d - prev_pos_2d
        abs_delta_2d = xp.abs(delta_2d)
        side_2d = xp.where(delta_2d >= 0.0, 1.0, -1.0)

        open_2d = open_arr[1:, 0][:, None]
        close_2d = close_arr[1:, 0][:, None]
        prev_close_2d = close_arr[:-1, 0][:, None]
        volume_2d = volume_arr[1:, 0][:, None]

        if slippage_kind == 0:
            fill_2d = open_2d + side_2d * slippage_points
        elif slippage_kind == 1:
            fill_2d = open_2d * (1.0 + side_2d * slippage_rate)
        else:
            safe_volume_2d = xp.where(volume_2d <= 0.0, 1.0, volume_2d)
            participation_2d = xp.minimum(abs_delta_2d / safe_volume_2d, volume_cap)
            fill_2d = open_2d * (1.0 + side_2d * slippage_impact * participation_2d)

        if limit_up_ratio > 0.0:
            fill_2d = xp.minimum(fill_2d, prev_close_2d * (1.0 + limit_up_ratio))
        if limit_down_ratio > 0.0:
            fill_2d = xp.maximum(fill_2d, prev_close_2d * (1.0 - limit_down_ratio))

        slippage_cash_2d = xp.abs(fill_2d - open_2d) * abs_delta_2d * contract_multiplier
        notional_2d = xp.abs(fill_2d) * abs_delta_2d * contract_multiplier
        if commission_kind == 0:
            commission_cash_2d = abs_delta_2d * commission_rate
        elif commission_kind == 1:
            commission_cash_2d = notional_2d * commission_rate
        else:
            if tiers_threshold.shape[0] == 0:
                commission_cash_2d = notional_2d * commission_rate
            else:
                commission_cash_2d = xp.zeros_like(notional_2d, dtype=float)
                remain_mask = xp.ones_like(notional_2d, dtype=bool)
                for idx in range(int(tiers_threshold.shape[0])):
                    threshold = tiers_threshold[idx]
                    rate = tiers_rate[idx]
                    mask = remain_mask & (notional_2d <= threshold)
                    commission_cash_2d = xp.where(mask, notional_2d * rate, commission_cash_2d)
                    remain_mask = remain_mask & (~mask)
                commission_cash_2d = xp.where(
                    remain_mask, notional_2d * tiers_rate[int(tiers_threshold.shape[0]) - 1], commission_cash_2d
                )

        delta_equity = (
            prev_pos_2d * (open_2d - prev_close_2d) * contract_multiplier
            + curr_pos_2d * (close_2d - open_2d) * contract_multiplier
            - (slippage_cash_2d + commission_cash_2d)
        )
    else:
        curr_pos = signal_tensor[:-1]
        if n_bars > 2:
            prev_pos = xp.concatenate(
                [
                    xp.zeros((1, n_assets, n_combos), dtype=float),
                    signal_tensor[:-2],
                ],
                axis=0,
            )
        else:
            prev_pos = xp.zeros_like(curr_pos, dtype=float)

        delta = curr_pos - prev_pos
        abs_delta = xp.abs(delta)
        side = xp.where(delta >= 0.0, 1.0, -1.0)

        open_t = open_arr[1:, :, None]
        close_t = close_arr[1:, :, None]
        prev_close = close_arr[:-1, :, None]
        volume_t = volume_arr[1:, :, None]

        if slippage_kind == 0:
            fill_price = open_t + side * slippage_points
        elif slippage_kind == 1:
            fill_price = open_t * (1.0 + side * slippage_rate)
        else:
            safe_volume = xp.where(volume_t <= 0.0, 1.0, volume_t)
            participation = xp.minimum(abs_delta / safe_volume, volume_cap)
            fill_price = open_t * (1.0 + side * slippage_impact * participation)

        if limit_up_ratio > 0.0:
            fill_price = xp.minimum(fill_price, prev_close * (1.0 + limit_up_ratio))
        if limit_down_ratio > 0.0:
            fill_price = xp.maximum(fill_price, prev_close * (1.0 - limit_down_ratio))

        slippage_cash = xp.abs(fill_price - open_t) * abs_delta * contract_multiplier
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
                for idx in range(int(tiers_threshold.shape[0])):
                    threshold = tiers_threshold[idx]
                    rate = tiers_rate[idx]
                    mask = remain_mask & (notional <= threshold)
                    commission_cash = xp.where(mask, notional * rate, commission_cash)
                    remain_mask = remain_mask & (~mask)
                commission_cash = xp.where(
                    remain_mask, notional * tiers_rate[int(tiers_threshold.shape[0]) - 1], commission_cash
                )

        trade_cost = xp.sum(slippage_cash + commission_cash, axis=1)
        pnl = xp.sum(
            prev_pos * (open_t - prev_close) * contract_multiplier
            + curr_pos * (close_t - open_t) * contract_multiplier,
            axis=1,
        )
        delta_equity = pnl - trade_cost

    equity_path = float(initial_cash) + xp.cumsum(delta_equity, axis=0)
    running_min = _running_min_accumulate(equity_path, xp=xp)
    reflected_floor = xp.minimum(running_min, 0.0)
    equity[1:] = equity_path - reflected_floor

    base = xp.where(equity[:-1] > 1.0, equity[:-1], 1.0)
    returns[1:] = (equity[1:] - equity[:-1]) / base
    return equity, returns


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

    if margin_ratio == 0.0:
        return _batch_sim_no_margin(
            open_arr=open_arr,
            close_arr=close,
            volume_arr=volume_arr,
            signal_tensor=signal_tensor,
            initial_cash=float(initial_cash),
            contract_multiplier=float(contract_multiplier),
            slippage_kind=slippage_kind,
            slippage_points=slippage_points,
            slippage_rate=slippage_rate,
            slippage_impact=slippage_impact,
            volume_cap=volume_cap,
            commission_kind=commission_kind,
            commission_rate=commission_rate,
            tiers_threshold=tiers_threshold,
            tiers_rate=tiers_rate,
            limit_up_ratio=limit_up_ratio,
            limit_down_ratio=limit_down_ratio,
        )

    positions = xp.zeros((n_bars, n_assets, n_combos), dtype=float)
    positions[1:] = signal_tensor[:-1]

    equity = xp.zeros((n_bars, n_combos), dtype=float)
    returns = xp.zeros((n_bars, n_combos), dtype=float)
    equity[0] = float(initial_cash)

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
                commission_cash = xp.where(
                    remain_mask, notional * tiers_rate[tiers_threshold.shape[0] - 1], commission_cash
                )
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
        day_ids: np.ndarray,
        week_ids: np.ndarray,
        max_risk_per_trade: float,
        max_daily_loss: float,
        max_weekly_loss: float,
        max_drawdown_limit: float,
        drawdown_action_code: int,
        max_position: float,
        max_addon_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_bars, n_assets = open_arr.shape
        positions = np.zeros((n_bars, n_assets), dtype=np.float64)
        for t in range(1, n_bars):
            for a in range(n_assets):
                positions[t, a] = signal_arr[t - 1, a]

        equity = np.zeros(n_bars, dtype=np.float64)
        returns = np.zeros(n_bars, dtype=np.float64)
        turnover = np.zeros(n_bars, dtype=np.float64)
        daily_flags = np.zeros(n_bars, dtype=np.int8)
        weekly_flags = np.zeros(n_bars, dtype=np.int8)
        drawdown_flags = np.zeros(n_bars, dtype=np.int8)
        position_flags = np.zeros(n_bars, dtype=np.int8)
        equity[0] = initial_cash
        current_day = 0
        current_week = 0
        if n_bars > 0:
            current_day = day_ids[0]
            current_week = week_ids[0]
        day_anchor_equity = initial_cash
        week_anchor_equity = initial_cash
        daily_stop = False
        weekly_stop = False
        drawdown_stop = False
        drawdown_reduce = False
        drawdown_logged = False
        peak_equity = initial_cash
        addon_counts = np.zeros(n_assets, dtype=np.int64)

        for t in range(1, n_bars):
            equity_prev = equity[t - 1]
            day_id = day_ids[t]
            if day_id != current_day:
                current_day = day_id
                day_anchor_equity = equity_prev
                daily_stop = False
            week_id = week_ids[t]
            if week_id != current_week:
                current_week = week_id
                week_anchor_equity = equity_prev
                weekly_stop = False
            prev_close = close_arr[t - 1]
            open_t = open_arr[t]
            close_t = close_arr[t]
            vol_t = volume_arr[t]
            curr_pos = np.empty(n_assets, dtype=np.float64)
            prev_pos = np.empty(n_assets, dtype=np.float64)
            abs_delta = np.empty(n_assets, dtype=np.float64)
            fill_price = np.empty(n_assets, dtype=np.float64)

            for a in range(n_assets):
                curr_pos[a] = positions[t, a]
                prev_pos[a] = positions[t - 1, a]

            if drawdown_reduce:
                for a in range(n_assets):
                    curr_pos[a] = curr_pos[a] * 0.5
                position_flags[t] = 1

            if drawdown_stop or daily_stop or weekly_stop:
                for a in range(n_assets):
                    curr_pos[a] = 0.0
                position_flags[t] = 1

            if max_position > 0.0:
                limited = False
                for a in range(n_assets):
                    pos = curr_pos[a]
                    if np.abs(pos) > max_position:
                        curr_pos[a] = max_position if pos > 0.0 else -max_position
                        limited = True
                if limited:
                    position_flags[t] = 1

            if max_addon_count > 0:
                addon_limited = False
                for a in range(n_assets):
                    prev = prev_pos[a]
                    curr = curr_pos[a]
                    if np.abs(prev) < 1e-12 or prev * curr <= 0.0:
                        addon_counts[a] = 0
                    else:
                        if np.abs(curr) > np.abs(prev) + 1e-12:
                            addon_counts[a] += 1
                            if addon_counts[a] > max_addon_count:
                                curr_pos[a] = prev
                                addon_counts[a] = max_addon_count
                                addon_limited = True
                        elif np.abs(curr) + 1e-12 < np.abs(prev):
                            addon_counts[a] = 0
                if addon_limited:
                    position_flags[t] = 1

            if max_risk_per_trade > 0.0 and equity_prev > 0.0:
                risk_notional = 0.0
                for a in range(n_assets):
                    risk_notional += np.abs(curr_pos[a] - prev_pos[a]) * open_t[a] * contract_multiplier
                allowed = max_risk_per_trade * equity_prev
                if risk_notional > allowed:
                    scale = allowed / (risk_notional + 1e-12)
                    if scale < 0.0:
                        scale = 0.0
                    for a in range(n_assets):
                        curr_pos[a] = prev_pos[a] + (curr_pos[a] - prev_pos[a]) * scale
                    position_flags[t] = 1

            if margin_ratio > 0:
                notional_curr = 0.0
                for a in range(n_assets):
                    notional_curr += np.abs(curr_pos[a]) * open_t[a] * contract_multiplier
                if notional_curr > 0:
                    required = margin_ratio * notional_curr
                    if required > equity_prev:
                        scale = equity_prev / (required + 1e-12)
                        if scale < 0.0:
                            scale = 0.0
                        for a in range(n_assets):
                            curr_pos[a] *= scale
                        position_flags[t] = 1

            for a in range(n_assets):
                positions[t, a] = curr_pos[a]

            trade_cost = 0.0
            turnover_t = 0.0
            pnl = 0.0
            for a in range(n_assets):
                curr = curr_pos[a]
                prev = prev_pos[a]
                d = curr - prev
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

            equity_t = equity_prev + pnl - trade_cost
            if equity_t < 0.0:
                equity_t = 0.0
            equity[t] = equity_t
            if equity_prev > 1.0:
                returns[t] = (equity_t - equity_prev) / equity_prev
            else:
                returns[t] = 0.0
            turnover[t] = turnover_t
            if equity_t > peak_equity:
                peak_equity = equity_t

            if max_daily_loss > 0.0 and day_anchor_equity > 0.0:
                daily_loss = (day_anchor_equity - equity_t) / day_anchor_equity
                if daily_loss >= max_daily_loss and not daily_stop:
                    daily_stop = True
                    daily_flags[t] = 1

            if max_weekly_loss > 0.0 and week_anchor_equity > 0.0:
                weekly_loss = (week_anchor_equity - equity_t) / week_anchor_equity
                if weekly_loss >= max_weekly_loss and not weekly_stop:
                    weekly_stop = True
                    weekly_flags[t] = 1

            if max_drawdown_limit > 0.0 and peak_equity > 0.0:
                drawdown = (peak_equity - equity_t) / peak_equity
                if drawdown >= max_drawdown_limit:
                    if drawdown_action_code == 0:
                        drawdown_stop = True
                        if not drawdown_logged:
                            drawdown_flags[t] = 1
                            drawdown_logged = True
                    elif drawdown_action_code == 1:
                        drawdown_reduce = True
                        if not drawdown_logged:
                            drawdown_flags[t] = 1
                            drawdown_logged = True
                    else:
                        if not drawdown_logged:
                            drawdown_flags[t] = 1
                            drawdown_logged = True

        return equity, returns, positions, turnover, daily_flags, weekly_flags, drawdown_flags, position_flags


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

    if isinstance(rules, TradingRules):
        limit_up_ratio = float(rules.limit_up_ratio or 0.0)
        limit_down_ratio = float(rules.limit_down_ratio or 0.0)
        margin_ratio = float(rules.margin_ratio)
        max_risk_per_trade = float(rules.max_risk_per_trade)
        max_daily_loss = float(rules.max_daily_loss)
        max_weekly_loss = float(rules.max_weekly_loss)
        max_drawdown_limit = float(rules.max_drawdown_limit)
        max_drawdown_action = str(rules.max_drawdown_action or "stop")
        max_position = float(rules.max_position)
        max_addon_count = int(max(rules.max_addon_count, 0))
    else:
        limit_up_ratio = 0.0
        limit_down_ratio = 0.0
        margin_ratio = 0.0
        max_risk_per_trade = 0.0
        max_daily_loss = 0.0
        max_weekly_loss = 0.0
        max_drawdown_limit = 0.0
        max_drawdown_action = "stop"
        max_position = 0.0
        max_addon_count = 0
    drawdown_action_code = _drawdown_action_code(max_drawdown_action)

    use_cpu_fastpath = (
        xp is np
        and not record_trades
        and njit is not None
        and isinstance(rules, TradingRules)
        and isinstance(slippage, (FixedSlippage, PercentSlippage, VolumeSlippage))
        and isinstance(commission, (FixedCommission, PercentCommission, TieredCommission))
    )

    if use_cpu_fastpath:
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
        day_ids = _calendar_day_ids(data.timestamps)
        week_ids = _calendar_week_ids(data.timestamps)

        equity, returns, positions, turnover, daily_flags, weekly_flags, drawdown_flags, position_flags = (
            _simulate_portfolio_numba(
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
                day_ids,
                week_ids,
                max_risk_per_trade,
                max_daily_loss,
                max_weekly_loss,
                max_drawdown_limit,
                drawdown_action_code,
                max_position,
                max_addon_count,
            )
        )
        risk_events = _collect_risk_events(
            daily_flags=daily_flags,
            weekly_flags=weekly_flags,
            drawdown_flags=drawdown_flags,
            position_flags=position_flags,
        )
        return PortfolioResult(
            equity_curve=equity,
            returns=returns,
            positions=positions,
            turnover=turnover,
            trades=[],
            risk_events=risk_events if risk_events else None,
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

    day_ids = _calendar_day_ids(data.timestamps)
    week_ids = _calendar_week_ids(data.timestamps)
    current_day = int(day_ids[0]) if day_ids.size > 0 else 0
    current_week = int(week_ids[0]) if week_ids.size > 0 else 0
    day_anchor_equity = float(initial_cash)
    week_anchor_equity = float(initial_cash)
    daily_stop = False
    weekly_stop = False
    drawdown_stop = False
    drawdown_reduce = False
    drawdown_logged = False
    peak_equity = float(initial_cash)
    addon_counts = np.zeros(n_assets, dtype=np.int64)
    risk_events: list[dict[str, Any]] = []

    bankruptcy_at: int | None = None
    for t in range(1, n_bars):
        equity_prev = float(equity[t - 1])

        if day_ids.size > t and int(day_ids[t]) != current_day:
            current_day = int(day_ids[t])
            day_anchor_equity = equity_prev
            daily_stop = False
        if week_ids.size > t and int(week_ids[t]) != current_week:
            current_week = int(week_ids[t])
            week_anchor_equity = equity_prev
            weekly_stop = False

        prev_pos = positions[t - 1]
        curr_pos = positions[t].copy()
        position_limited = False

        if drawdown_reduce:
            curr_pos = curr_pos * 0.5
            position_limited = True

        if drawdown_stop or daily_stop or weekly_stop:
            curr_pos = curr_pos * 0.0
            position_limited = True

        curr_np = np.asarray(to_numpy(curr_pos), dtype=np.float64)
        prev_np = np.asarray(to_numpy(prev_pos), dtype=np.float64)

        if max_position > 0.0:
            clipped = np.clip(curr_np, -max_position, max_position)
            if not np.allclose(clipped, curr_np):
                curr_np = clipped
                position_limited = True

        if max_addon_count > 0:
            for idx in range(n_assets):
                prev = prev_np[idx]
                curr = curr_np[idx]
                if abs(prev) < 1e-12 or prev * curr <= 0.0:
                    addon_counts[idx] = 0
                    continue
                if abs(curr) > abs(prev) + 1e-12:
                    addon_counts[idx] += 1
                    if addon_counts[idx] > max_addon_count:
                        curr_np[idx] = prev
                        addon_counts[idx] = max_addon_count
                        position_limited = True
                elif abs(curr) + 1e-12 < abs(prev):
                    addon_counts[idx] = 0

        if max_risk_per_trade > 0.0 and equity_prev > 0.0:
            open_np = np.asarray(to_numpy(data.open[t]), dtype=np.float64)
            risk_notional = float(np.sum(np.abs(curr_np - prev_np) * open_np * float(contract_multiplier)))
            allowed = max_risk_per_trade * equity_prev
            if risk_notional > allowed:
                scale = max(allowed / (risk_notional + 1e-12), 0.0)
                curr_np = prev_np + (curr_np - prev_np) * scale
                position_limited = True

        curr_pos = xp.asarray(curr_np, dtype=float)

        if rules is not None:
            notional_curr = xp.sum(xp.abs(curr_pos) * data.open[t] * contract_multiplier)
            required = margin_ratio * notional_curr
            if required > equity[t - 1]:
                scale = float(equity[t - 1] / (required + 1e-12))
                curr_pos = curr_pos * max(scale, 0.0)
                position_limited = True

        positions[t] = curr_pos
        if position_limited:
            risk_events.append({"bar": int(t), "type": "position_limit", "detail": "position adjusted by risk rules"})

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
        equity_now = float(equity[t])
        base = equity_prev if equity_prev > 1.0 else 1.0
        returns[t] = (equity_now - equity_prev) / base
        if bankruptcy_at is None and equity_now <= 0.0:
            bankruptcy_at = t
        if equity_now > peak_equity:
            peak_equity = equity_now

        if max_daily_loss > 0.0 and day_anchor_equity > 0.0:
            daily_loss = (day_anchor_equity - equity_now) / day_anchor_equity
            if daily_loss >= max_daily_loss and not daily_stop:
                daily_stop = True
                risk_events.append({"bar": int(t), "type": "daily_loss_breach", "detail": "daily loss limit breached"})

        if max_weekly_loss > 0.0 and week_anchor_equity > 0.0:
            weekly_loss = (week_anchor_equity - equity_now) / week_anchor_equity
            if weekly_loss >= max_weekly_loss and not weekly_stop:
                weekly_stop = True
                risk_events.append(
                    {"bar": int(t), "type": "weekly_loss_breach", "detail": "weekly loss limit breached"}
                )

        if max_drawdown_limit > 0.0 and peak_equity > 0.0:
            drawdown = (peak_equity - equity_now) / peak_equity
            if drawdown >= max_drawdown_limit:
                if drawdown_action_code == 0:
                    drawdown_stop = True
                elif drawdown_action_code == 1:
                    drawdown_reduce = True
                if not drawdown_logged:
                    drawdown_logged = True
                    risk_events.append(
                        {"bar": int(t), "type": "drawdown_breach", "detail": "drawdown guardrail triggered"}
                    )

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
        risk_events=risk_events if risk_events else None,
        metadata={
            "symbols": data.symbols,
            "backend": data.backend.active,
            "initial_cash": initial_cash,
            "contract_multiplier": contract_multiplier,
            "bankrupt": bankruptcy_at is not None,
            "bankruptcy_index": bankruptcy_at,
        },
    )
