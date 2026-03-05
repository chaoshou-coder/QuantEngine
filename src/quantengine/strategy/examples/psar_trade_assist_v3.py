from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from quantengine.data.gpu_backend import to_numpy
from quantengine.indicators.technical import adx, atr, ema, parabolic_sar, rsi
from quantengine.strategy.base import BaseStrategy, ParameterSpace
from quantengine.strategy.registry import register_strategy

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


def _get_xp(values: Any):
    if cp is not None and isinstance(values, cp.ndarray):
        return cp
    return np


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _shift(values: Any, periods: int, xp, fill_value: float = 0.0):
    arr = xp.asarray(values, dtype=float)
    out = xp.empty_like(arr, dtype=float)
    out[...] = fill_value
    if periods <= 0:
        return arr
    if arr.ndim == 1:
        if periods < arr.shape[0]:
            out[periods:] = arr[:-periods]
        return out
    if periods < arr.shape[0]:
        out[periods:, :] = arr[:-periods, :]
    return out


def _ffill_nonzero(values: Any, xp):
    arr = xp.asarray(values, dtype=float)
    if xp is np:
        if arr.ndim == 1:
            idx = np.where(arr != 0.0, np.arange(arr.shape[0]), 0)
            idx = np.maximum.accumulate(idx)
            return arr[idx]
        n_bars, n_assets = arr.shape
        rows = np.arange(n_bars).reshape(-1, 1)
        idx = np.where(arr != 0.0, rows, 0)
        idx = np.maximum.accumulate(idx, axis=0)
        cols = np.arange(n_assets).reshape(1, -1)
        return arr[idx, cols]

    # CuPy path: stay on GPU via cumsum(group-id) + gather.
    if arr.ndim == 1:
        mask = arr != 0.0
        group = xp.cumsum(mask.astype(xp.int32))
        out = xp.zeros_like(arr, dtype=float)
        if not bool(xp.any(mask)):
            return out
        vals = arr[mask]
        valid = group > 0
        out[valid] = vals[group[valid] - 1]
        return out

    n_bars, n_assets = arr.shape
    out = xp.zeros_like(arr, dtype=float)
    for col in range(n_assets):
        col_arr = arr[:, col]
        mask = col_arr != 0.0
        if not bool(xp.any(mask)):
            continue
        group = xp.cumsum(mask.astype(xp.int32))
        vals = col_arr[mask]
        valid = group > 0
        out_col = xp.zeros_like(col_arr, dtype=float)
        out_col[valid] = vals[group[valid] - 1]
        out[:, col] = out_col
    return out


def _rolling_median_1d(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    if window <= 1:
        out[:] = values
        return out
    if values.shape[0] < window:
        return out
    windows = sliding_window_view(values, window_shape=window)
    out[window - 1 :] = np.median(windows, axis=-1)
    return out


def _calc_atr_ratio(atr_vals: Any, window: int, xp) -> Any:
    atr_np = np.asarray(to_numpy(atr_vals), dtype=np.float64)
    if atr_np.ndim == 1:
        atr_np = atr_np.reshape(-1, 1)
    n_bars, n_assets = atr_np.shape
    ratio = np.ones((n_bars, n_assets), dtype=np.float64)
    if window <= 1:
        return xp.asarray(ratio)
    for col in range(n_assets):
        median = _rolling_median_1d(atr_np[:, col], window)
        valid = median > 0.0
        ratio[valid, col] = atr_np[valid, col] / median[valid]
    return xp.asarray(ratio)


@dataclass
class _IndicatorCache:
    values: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def get_or_compute(self, key: tuple[Any, ...], compute):
        if key in self.values:
            return self.values[key]
        value = compute()
        self.values[key] = value
        return value


@register_strategy("psar_trade_assist_v3")
class PsarTradeAssistV3Strategy(BaseStrategy):
    """Adaptive PSAR strategy with vectorized GPU-ready signal generation."""

    def parameters(self) -> dict[str, ParameterSpace]:
        return {
            "enable_adx": ParameterSpace(kind="choice", choices=[0, 1]),
            "enable_rsi": ParameterSpace(kind="choice", choices=[0, 1]),
            "enable_ma": ParameterSpace(kind="choice", choices=[0, 1]),
            "enable_body": ParameterSpace(kind="choice", choices=[0, 1]),
            "enable_adaptive_step": ParameterSpace(kind="choice", choices=[0, 1]),
            "require_adx_slope": ParameterSpace(kind="choice", choices=[0, 1]),
            "atr_period": ParameterSpace(kind="int", low=10, high=20, step=2),
            "atr_ratio_window": ParameterSpace(kind="int", low=50, high=150, step=25),
            "atr_ratio_low": ParameterSpace(kind="float", low=0.5, high=0.9, step=0.1),
            "atr_ratio_high": ParameterSpace(kind="float", low=1.2, high=2.0, step=0.1),
            "psar_step_low": ParameterSpace(kind="float", low=0.010, high=0.020, step=0.005),
            "psar_step_mid": ParameterSpace(kind="float", low=0.020, high=0.035, step=0.005),
            "psar_step_high": ParameterSpace(kind="float", low=0.035, high=0.050, step=0.005),
            "psar_max": ParameterSpace(kind="float", low=0.15, high=0.25, step=0.05),
            "adx_period": ParameterSpace(kind="int", low=10, high=20, step=2),
            "adx_threshold": ParameterSpace(kind="float", low=15.0, high=25.0, step=2.5),
            "ma_fast": ParameterSpace(kind="int", low=10, high=30, step=5),
            "ma_slow": ParameterSpace(kind="int", low=30, high=90, step=10),
            "rsi_period": ParameterSpace(kind="int", low=10, high=20, step=2),
            "rsi_upper": ParameterSpace(kind="float", low=70.0, high=85.0, step=5.0),
            "rsi_lower": ParameterSpace(kind="float", low=15.0, high=35.0, step=5.0),
            "body_atr_mult": ParameterSpace(kind="float", low=2.0, high=3.5, step=0.5),
        }

    def _resolve_params(self, params: dict[str, Any]) -> dict[str, Any]:
        resolved = {
            "enable_adx": _as_bool(params.get("enable_adx", 1), True),
            "enable_rsi": _as_bool(params.get("enable_rsi", 1), True),
            "enable_ma": _as_bool(params.get("enable_ma", 1), True),
            "enable_body": _as_bool(params.get("enable_body", 1), True),
            "enable_adaptive_step": _as_bool(params.get("enable_adaptive_step", 1), True),
            "require_adx_slope": _as_bool(params.get("require_adx_slope", 1), True),
            "atr_period": int(params.get("atr_period", 14)),
            "atr_ratio_window": int(params.get("atr_ratio_window", 100)),
            "atr_ratio_low": float(params.get("atr_ratio_low", 0.70)),
            "atr_ratio_high": float(params.get("atr_ratio_high", 1.50)),
            "psar_step_low": float(params.get("psar_step_low", 0.015)),
            "psar_step_mid": float(params.get("psar_step_mid", 0.025)),
            "psar_step_high": float(params.get("psar_step_high", 0.040)),
            "psar_max": float(params.get("psar_max", 0.20)),
            "adx_period": int(params.get("adx_period", 14)),
            "adx_threshold": float(params.get("adx_threshold", 20.0)),
            "ma_fast": int(params.get("ma_fast", 20)),
            "ma_slow": int(params.get("ma_slow", 50)),
            "rsi_period": int(params.get("rsi_period", 14)),
            "rsi_upper": float(params.get("rsi_upper", 75.0)),
            "rsi_lower": float(params.get("rsi_lower", 25.0)),
            "body_atr_mult": float(params.get("body_atr_mult", 2.5)),
        }
        return resolved

    def generate_signals(self, data, params: dict[str, Any], cache: _IndicatorCache | None = None):
        cfg = self._resolve_params(params)
        xp = _get_xp(data.close)
        close_arr = xp.asarray(data.close, dtype=float)
        high_arr = xp.asarray(data.high, dtype=float)
        low_arr = xp.asarray(data.low, dtype=float)
        open_arr = xp.asarray(data.open, dtype=float)
        if close_arr.ndim == 1:
            close_arr = close_arr.reshape(-1, 1)
            high_arr = high_arr.reshape(-1, 1)
            low_arr = low_arr.reshape(-1, 1)
            open_arr = open_arr.reshape(-1, 1)

        n_bars, n_assets = close_arr.shape
        if n_bars == 0:
            return xp.zeros((0, n_assets), dtype=float)

        warmup = (
            max(
                cfg["atr_ratio_window"],
                cfg["ma_slow"],
                cfg["atr_period"],
                cfg["adx_period"],
                cfg["rsi_period"],
            )
            + 10
        )
        if n_bars < max(warmup, 3):
            return xp.zeros((n_bars, n_assets), dtype=float)

        data_key = id(data)

        def _cached(name: str, key_parts: tuple[Any, ...], compute):
            if cache is None:
                return compute()
            return cache.get_or_compute((name, data_key) + key_parts, compute)

        atr_vals = xp.asarray(
            _cached(
                "atr",
                (cfg["atr_period"],),
                lambda: atr(high_arr, low_arr, close_arr, window=cfg["atr_period"]),
            ),
            dtype=float,
        )
        atr_ratio = xp.asarray(
            _cached(
                "atr_ratio",
                (cfg["atr_period"], cfg["atr_ratio_window"]),
                lambda: _calc_atr_ratio(atr_vals, cfg["atr_ratio_window"], xp=xp),
            ),
            dtype=float,
        )
        adx_line, plus_di, minus_di = _cached(
            "adx",
            (cfg["adx_period"],),
            lambda: adx(high_arr, low_arr, close_arr, window=cfg["adx_period"]),
        )
        adx_line = xp.asarray(adx_line, dtype=float)
        plus_di = xp.asarray(plus_di, dtype=float)
        minus_di = xp.asarray(minus_di, dtype=float)
        rsi_vals = xp.asarray(
            _cached("rsi", (cfg["rsi_period"],), lambda: rsi(close_arr, window=cfg["rsi_period"])),
            dtype=float,
        )
        ema_fast = xp.asarray(
            _cached("ema", (cfg["ma_fast"],), lambda: ema(close_arr, span=cfg["ma_fast"])),
            dtype=float,
        )
        ema_slow = xp.asarray(
            _cached("ema", (cfg["ma_slow"],), lambda: ema(close_arr, span=cfg["ma_slow"])),
            dtype=float,
        )

        sar_mid = xp.asarray(
            _cached(
                "psar",
                (cfg["psar_step_mid"], cfg["psar_max"]),
                lambda: parabolic_sar(high_arr, low_arr, close_arr, step=cfg["psar_step_mid"], maximum=cfg["psar_max"]),
            ),
            dtype=float,
        )
        sar1_mid = _shift(sar_mid, 1, xp)
        sar2_mid = _shift(sar_mid, 2, xp)
        if cfg["enable_adaptive_step"]:
            sar_low = xp.asarray(
                _cached(
                    "psar",
                    (cfg["psar_step_low"], cfg["psar_max"]),
                    lambda: parabolic_sar(
                        high_arr, low_arr, close_arr, step=cfg["psar_step_low"], maximum=cfg["psar_max"]
                    ),
                ),
                dtype=float,
            )
            sar_high = xp.asarray(
                _cached(
                    "psar",
                    (cfg["psar_step_high"], cfg["psar_max"]),
                    lambda: parabolic_sar(
                        high_arr, low_arr, close_arr, step=cfg["psar_step_high"], maximum=cfg["psar_max"]
                    ),
                ),
                dtype=float,
            )
            sar1_low = _shift(sar_low, 1, xp)
            sar2_low = _shift(sar_low, 2, xp)
            sar1_high = _shift(sar_high, 1, xp)
            sar2_high = _shift(sar_high, 2, xp)
            low_mask = atr_ratio < cfg["atr_ratio_low"]
            high_mask = atr_ratio > cfg["atr_ratio_high"]
            sar1 = xp.where(low_mask, sar1_low, xp.where(high_mask, sar1_high, sar1_mid))
            sar2 = xp.where(low_mask, sar2_low, xp.where(high_mask, sar2_high, sar2_mid))
        else:
            sar1 = sar1_mid
            sar2 = sar2_mid

        c1 = _shift(close_arr, 1, xp)
        c2 = _shift(close_arr, 2, xp)
        o1 = _shift(open_arr, 1, xp)
        adx1 = _shift(adx_line, 1, xp)
        adx2 = _shift(adx_line, 2, xp)
        pdi1 = _shift(plus_di, 1, xp)
        mdi1 = _shift(minus_di, 1, xp)
        rsi1 = _shift(rsi_vals, 1, xp)
        atr1 = _shift(atr_vals, 1, xp)
        ef1 = _shift(ema_fast, 1, xp)
        es1 = _shift(ema_slow, 1, xp)

        flip_long = (sar2 >= c2) & (sar1 < c1)
        flip_short = (sar2 <= c2) & (sar1 > c1)
        flip_any = flip_long | flip_short
        direction = xp.where(flip_long, 1.0, xp.where(flip_short, -1.0, 0.0))

        gate = xp.ones_like(direction, dtype=bool)
        if cfg["enable_adx"]:
            adx_ok = adx1 >= cfg["adx_threshold"]
            if cfg["require_adx_slope"]:
                adx_ok = adx_ok & (adx1 > adx2)
            di_ok = xp.where(direction > 0.0, pdi1 > mdi1, mdi1 > pdi1)
            gate = gate & adx_ok & di_ok
        if cfg["enable_rsi"]:
            gate = gate & (rsi1 > cfg["rsi_lower"]) & (rsi1 < cfg["rsi_upper"])
        if cfg["enable_body"]:
            gate = gate & (atr1 > 0.0) & (xp.abs(c1 - o1) <= cfg["body_atr_mult"] * atr1)
        if cfg["enable_ma"]:
            ma_long = (c1 > ef1) & (ef1 >= es1)
            ma_short = (c1 < ef1) & (ef1 <= es1)
            gate = gate & xp.where(direction > 0.0, ma_long, ma_short)

        warm_mask = xp.arange(n_bars).reshape(-1, 1) >= warmup
        accepted_flip = flip_any & gate & warm_mask
        trigger = xp.where(accepted_flip, direction, 0.0)
        signal = _ffill_nonzero(trigger, xp)
        signal = xp.where(warm_mask, signal, 0.0)
        signal = xp.clip(xp.nan_to_num(signal, nan=0.0), -1.0, 1.0)
        return signal

    def generate_signals_reference(self, data, params: dict[str, Any]) -> np.ndarray:
        """Reference loop implementation for acceptance tests."""
        cfg = self._resolve_params(params)
        close_np = np.asarray(to_numpy(data.close), dtype=np.float64)
        high_np = np.asarray(to_numpy(data.high), dtype=np.float64)
        low_np = np.asarray(to_numpy(data.low), dtype=np.float64)
        open_np = np.asarray(to_numpy(data.open), dtype=np.float64)
        if close_np.ndim == 1:
            close_np = close_np.reshape(-1, 1)
            high_np = high_np.reshape(-1, 1)
            low_np = low_np.reshape(-1, 1)
            open_np = open_np.reshape(-1, 1)
        n_bars, n_assets = close_np.shape
        warmup = (
            max(
                cfg["atr_ratio_window"],
                cfg["ma_slow"],
                cfg["atr_period"],
                cfg["adx_period"],
                cfg["rsi_period"],
            )
            + 10
        )
        if n_bars < max(warmup, 3):
            return np.zeros((n_bars, n_assets), dtype=np.float64)

        atr_vals = np.asarray(atr(high_np, low_np, close_np, window=cfg["atr_period"]), dtype=np.float64)
        atr_ratio = np.asarray(_calc_atr_ratio(atr_vals, cfg["atr_ratio_window"], xp=np), dtype=np.float64)
        adx_line, plus_di, minus_di = adx(high_np, low_np, close_np, window=cfg["adx_period"])
        adx_np = np.asarray(adx_line, dtype=np.float64)
        pdi_np = np.asarray(plus_di, dtype=np.float64)
        mdi_np = np.asarray(minus_di, dtype=np.float64)
        rsi_np = np.asarray(rsi(close_np, window=cfg["rsi_period"]), dtype=np.float64)
        ema_fast = np.asarray(ema(close_np, span=cfg["ma_fast"]), dtype=np.float64)
        ema_slow = np.asarray(ema(close_np, span=cfg["ma_slow"]), dtype=np.float64)
        sar_mid = np.asarray(
            parabolic_sar(high_np, low_np, close_np, step=cfg["psar_step_mid"], maximum=cfg["psar_max"]),
            dtype=np.float64,
        )
        if cfg["enable_adaptive_step"]:
            sar_low = np.asarray(
                parabolic_sar(high_np, low_np, close_np, step=cfg["psar_step_low"], maximum=cfg["psar_max"]),
                dtype=np.float64,
            )
            sar_high = np.asarray(
                parabolic_sar(high_np, low_np, close_np, step=cfg["psar_step_high"], maximum=cfg["psar_max"]),
                dtype=np.float64,
            )
        else:
            sar_low = sar_mid
            sar_high = sar_mid

        out = np.zeros((n_bars, n_assets), dtype=np.float64)
        for col in range(n_assets):
            pos = 0.0
            for i in range(warmup, n_bars):
                ratio = atr_ratio[i, col]
                if ratio < cfg["atr_ratio_low"]:
                    sar = sar_low
                elif ratio > cfg["atr_ratio_high"]:
                    sar = sar_high
                else:
                    sar = sar_mid
                s1 = sar[i - 1, col]
                s2 = sar[i - 2, col]
                c1 = close_np[i - 1, col]
                c2 = close_np[i - 2, col]
                flip_long = (s2 >= c2) and (s1 < c1)
                flip_short = (s2 <= c2) and (s1 > c1)
                if not (flip_long or flip_short):
                    out[i, col] = pos
                    continue

                direction = 1.0 if flip_long else -1.0
                allowed = True
                if cfg["enable_adx"]:
                    if adx_np[i - 1, col] < cfg["adx_threshold"]:
                        allowed = False
                    if cfg["require_adx_slope"] and adx_np[i - 1, col] <= adx_np[i - 2, col]:
                        allowed = False
                    if direction > 0 and pdi_np[i - 1, col] <= mdi_np[i - 1, col]:
                        allowed = False
                    if direction < 0 and mdi_np[i - 1, col] <= pdi_np[i - 1, col]:
                        allowed = False
                if cfg["enable_rsi"] and (
                    rsi_np[i - 1, col] <= cfg["rsi_lower"] or rsi_np[i - 1, col] >= cfg["rsi_upper"]
                ):
                    allowed = False
                if cfg["enable_body"] and (
                    atr_vals[i - 1, col] <= 0
                    or abs(close_np[i - 1, col] - open_np[i - 1, col]) > cfg["body_atr_mult"] * atr_vals[i - 1, col]
                ):
                    allowed = False
                if cfg["enable_ma"]:
                    if direction > 0 and not (
                        close_np[i - 1, col] > ema_fast[i - 1, col] and ema_fast[i - 1, col] >= ema_slow[i - 1, col]
                    ):
                        allowed = False
                    if direction < 0 and not (
                        close_np[i - 1, col] < ema_fast[i - 1, col] and ema_fast[i - 1, col] <= ema_slow[i - 1, col]
                    ):
                        allowed = False

                if allowed:
                    pos = direction
                out[i, col] = pos
        return np.nan_to_num(out, nan=0.0)
