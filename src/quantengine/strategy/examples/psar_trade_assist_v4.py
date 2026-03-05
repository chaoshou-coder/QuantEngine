"""PSAR V4 trend-following strategy with structured entry frameworks,
trade management state machine, and risk mode presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from quantengine.data.gpu_backend import to_numpy
from quantengine.indicators.technical import (
    adx,
    atr,
    cci,
    donchian,
    ema,
    macd,
    parabolic_sar,
    rsi,
    sma,
)
from quantengine.strategy.base import BaseStrategy, ParameterSpace
from quantengine.strategy.registry import register_strategy

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

# ---------------------------------------------------------------------------
# Flag / param index constants for the Numba state-machine
# ---------------------------------------------------------------------------
F_ADX = 0
F_RSI = 1
F_MA = 2
F_BODY = 3
F_ADAPTIVE = 4
F_ADX_SLOPE = 5
F_MACD = 6
F_VOL = 7
F_SESSION = 8
F_MOMENTUM = 9
F_ATR_SL = 10
F_STRUCT_SL = 11
F_TP = 12
F_TRAIL_PSAR = 13
F_TRAIL_ATR = 14
F_PARTIAL = 15
F_BE = 16
F_ADDON = 17
F_TIMESTOP = 18
N_FLAGS = 19

P_ADX_THRESH = 0
P_RSI_LO = 1
P_RSI_HI = 2
P_BODY_MULT = 3
P_VOL_MULT = 4
P_CCI_THRESH = 5
P_ATR_SL_MULT = 6
P_STRUCT_LOOK = 7
P_STRUCT_PAD = 8
P_TP1_R = 9
P_TP2_R = 10
P_TRAIL_ATR_MULT = 11
P_PARTIAL_RATIO = 12
P_BE_TRIG = 13
P_BE_BUF = 14
P_ADDON_SIZE = 15
P_TIMESTOP_BARS = 16
P_NOPROG_R = 17
P_ENTRY_SIZE = 18
P_ASIAN_START = 19
P_ASIAN_END = 20
N_PARAMS = 21

# ---------------------------------------------------------------------------
# Framework / risk-mode definitions
# ---------------------------------------------------------------------------
FRAMEWORKS: dict[str, dict[str, int]] = {
    "F1": {},
    "F2": {"enable_adx": 1},
    "F3": {"enable_ma": 1},
    "F4": {"enable_adx": 1, "enable_rsi": 1},
    "F5": {"enable_ma": 1, "enable_body": 1},
}

RISK_MODES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "conservative": {
        "enable_atr_sl": 1,
        "enable_trailing_psar": 1,
        "enable_tp": 1,
    },
    "standard": {
        "enable_atr_sl": 1,
        "enable_structure_sl": 1,
        "enable_tp": 1,
        "enable_trailing_psar": 1,
        "enable_partial_close": 1,
        "enable_breakeven": 1,
    },
    "aggressive": {
        "enable_atr_sl": 1,
        "enable_structure_sl": 1,
        "enable_tp": 1,
        "enable_trailing_psar": 1,
        "enable_partial_close": 1,
        "enable_breakeven": 1,
        "enable_addon": 1,
        "enable_time_stop": 1,
        "entry_size": 0.6,
    },
}

# ---------------------------------------------------------------------------
# Indicator cache (reused from V3)
# ---------------------------------------------------------------------------


@dataclass
class _IndicatorCache:
    values: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def get_or_compute(self, key: tuple[Any, ...], compute):  # noqa: ANN001
        if key in self.values:
            return self.values[key]
        value = compute()
        self.values[key] = value
        return value


# ---------------------------------------------------------------------------
# Rolling-median helper (reused from V3)
# ---------------------------------------------------------------------------


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


def _calc_atr_ratio(atr_vals: np.ndarray, window: int) -> np.ndarray:
    if atr_vals.ndim == 1:
        atr_vals = atr_vals.reshape(-1, 1)
    n, m = atr_vals.shape
    ratio = np.ones((n, m), dtype=np.float64)
    if window <= 1:
        return ratio
    for col in range(m):
        median = _rolling_median_1d(atr_vals[:, col], window)
        valid = median > 0.0
        ratio[valid, col] = atr_vals[valid, col] / median[valid]
    return ratio


# ---------------------------------------------------------------------------
# Numba state-machine loop (pure-Python version, optionally @njit)
# ---------------------------------------------------------------------------


def _v4_loop_py(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_arr: np.ndarray,
    volume: np.ndarray,
    sar: np.ndarray,
    atr_v: np.ndarray,
    adx_v: np.ndarray,
    pdi: np.ndarray,
    mdi: np.ndarray,
    rsi_v: np.ndarray,
    ema_f: np.ndarray,
    ema_s: np.ndarray,
    macd_h: np.ndarray,
    vol_ma: np.ndarray,
    donch_hi: np.ndarray,
    donch_lo: np.ndarray,
    cci_v: np.ndarray,
    hour: np.ndarray,
    flags: np.ndarray,
    params: np.ndarray,
    warmup: int,
) -> np.ndarray:
    n = close.shape[0]
    signal = np.zeros(n, dtype=np.float64)
    if n < max(warmup, 3):
        return signal

    # Unpack frequently-used params once
    adx_thr = params[P_ADX_THRESH]
    rsi_lo = params[P_RSI_LO]
    rsi_hi = params[P_RSI_HI]
    body_m = params[P_BODY_MULT]
    vol_m = params[P_VOL_MULT]
    cci_thr = params[P_CCI_THRESH]
    atr_sl_m = params[P_ATR_SL_MULT]
    s_look = int(params[P_STRUCT_LOOK])
    s_pad = params[P_STRUCT_PAD]
    tp1r = params[P_TP1_R]
    tp2r = params[P_TP2_R]
    trail_atr_m = params[P_TRAIL_ATR_MULT]
    p_ratio = params[P_PARTIAL_RATIO]
    be_trig = params[P_BE_TRIG]
    be_buf = params[P_BE_BUF]
    addon_sz = params[P_ADDON_SIZE]
    ts_bars = int(params[P_TIMESTOP_BARS])
    noprog = params[P_NOPROG_R]
    entry_sz = params[P_ENTRY_SIZE]
    asian_s = int(params[P_ASIAN_START])
    asian_e = int(params[P_ASIAN_END])

    f_adx = flags[F_ADX] != 0
    f_rsi = flags[F_RSI] != 0
    f_ma = flags[F_MA] != 0
    f_body = flags[F_BODY] != 0
    f_adx_slope = flags[F_ADX_SLOPE] != 0
    f_macd = flags[F_MACD] != 0
    f_vol = flags[F_VOL] != 0
    f_session = flags[F_SESSION] != 0
    f_momentum = flags[F_MOMENTUM] != 0
    f_atr_sl = flags[F_ATR_SL] != 0
    f_struct_sl = flags[F_STRUCT_SL] != 0
    f_tp = flags[F_TP] != 0
    f_trail_psar = flags[F_TRAIL_PSAR] != 0
    f_trail_atr = flags[F_TRAIL_ATR] != 0
    f_partial = flags[F_PARTIAL] != 0
    f_be = flags[F_BE] != 0
    f_addon = flags[F_ADDON] != 0
    f_timestop = flags[F_TIMESTOP] != 0
    has_risk = f_atr_sl or f_struct_sl or f_tp or f_trail_psar or f_trail_atr

    # State
    pos = 0.0
    entry_px = 0.0
    sl = 0.0
    initial_sl = 0.0
    tp1_px = 0.0
    tp2_px = 0.0
    bars_in = 0
    tp1_hit = False
    be_done = False
    addon_ct = 0
    best_px = 0.0  # highest (long) / lowest (short) since entry

    for i in range(warmup, n):
        # ---- Phase 1: Exit checks (if in trade) ----
        if pos != 0.0:
            bars_in += 1
            direction = 1.0 if pos > 0.0 else -1.0

            if direction > 0.0:
                if high[i] > best_px:
                    best_px = high[i]
            else:
                if low[i] < best_px:
                    best_px = low[i]

            exited = False

            # SL check
            if has_risk and sl != 0.0:
                if direction > 0.0 and low[i] <= sl or direction < 0.0 and high[i] >= sl:
                    exited = True

            # TP2 check
            if not exited and f_tp and tp2_px != 0.0:
                if direction > 0.0 and high[i] >= tp2_px or direction < 0.0 and low[i] <= tp2_px:
                    exited = True

            if exited:
                pos = 0.0
                entry_px = 0.0
                sl = 0.0
                initial_sl = 0.0
                tp1_px = 0.0
                tp2_px = 0.0
                bars_in = 0
                tp1_hit = False
                be_done = False
                addon_ct = 0
                best_px = 0.0
                signal[i] = 0.0
                continue

            # TP1 partial close
            if f_partial and f_tp and not tp1_hit and tp1_px != 0.0:
                if direction > 0.0 and high[i] >= tp1_px or direction < 0.0 and low[i] <= tp1_px:
                    tp1_hit = True
                    pos = pos * (1.0 - p_ratio)
                    if abs(pos) < 0.01:
                        pos = 0.0

            if pos == 0.0:
                entry_px = 0.0
                sl = 0.0
                initial_sl = 0.0
                tp1_px = 0.0
                tp2_px = 0.0
                bars_in = 0
                tp1_hit = False
                be_done = False
                addon_ct = 0
                best_px = 0.0
                signal[i] = 0.0
                continue

            # Trailing PSAR
            if f_trail_psar and sar[i] > 0.0:
                if direction > 0.0 and sar[i] < close[i]:
                    if sl == 0.0 or sar[i] > sl:
                        sl = sar[i]
                elif direction < 0.0 and sar[i] > close[i] and (sl == 0.0 or sar[i] < sl):
                    sl = sar[i]

            # Trailing ATR
            if f_trail_atr and atr_v[i] > 0.0:
                if direction > 0.0:
                    t_sl = best_px - trail_atr_m * atr_v[i]
                    if sl == 0.0 or t_sl > sl:
                        sl = t_sl
                else:
                    t_sl = best_px + trail_atr_m * atr_v[i]
                    if sl == 0.0 or t_sl < sl:
                        sl = t_sl

            # Breakeven
            if f_be and not be_done and entry_px > 0.0 and initial_sl != 0.0:
                r = abs(entry_px - initial_sl)
                if r > 0.0:
                    if direction > 0.0 and close[i] >= entry_px + be_trig * r:
                        new_sl = entry_px + be_buf * r
                        if new_sl > sl:
                            sl = new_sl
                        be_done = True
                    elif direction < 0.0 and close[i] <= entry_px - be_trig * r:
                        new_sl = entry_px - be_buf * r
                        if new_sl < sl:
                            sl = new_sl
                        be_done = True

            # Time stop
            if f_timestop and bars_in >= ts_bars:
                r = abs(entry_px - initial_sl) if initial_sl != 0.0 else atr_v[i]
                if r > 0.0:
                    if direction > 0.0:
                        progress = (close[i] - entry_px) / r
                    else:
                        progress = (entry_px - close[i]) / r
                    if progress < noprog:
                        pos = 0.0
                        entry_px = 0.0
                        sl = 0.0
                        initial_sl = 0.0
                        tp1_px = 0.0
                        tp2_px = 0.0
                        bars_in = 0
                        tp1_hit = False
                        be_done = False
                        addon_ct = 0
                        best_px = 0.0
                        signal[i] = 0.0
                        continue

            # Add-on
            if f_addon and addon_ct < 1 and tp1_hit:
                if direction > 0.0 and close[i] > ema_f[i] and close[i] < tp1_px:
                    addon_ct += 1
                    pos = min(abs(pos) + addon_sz, 1.0) * direction
                elif direction < 0.0 and close[i] < ema_f[i] and close[i] > tp1_px:
                    addon_ct += 1
                    pos = -min(abs(pos) + addon_sz, 1.0)

        # ---- Phase 2: PSAR flip check (entry or reversal) ----
        if i < 2:
            signal[i] = pos
            continue

        s1 = sar[i - 1]
        s2 = sar[i - 2]
        c_1 = close[i - 1]
        c_2 = close[i - 2]

        flip_long = (s2 >= c_2) and (s1 < c_1)
        flip_short = (s2 <= c_2) and (s1 > c_1)
        if not (flip_long or flip_short):
            signal[i] = pos
            continue

        d = 1.0 if flip_long else -1.0

        # If already in same direction, keep current trade (no re-entry)
        if pos != 0.0:
            cur_dir = 1.0 if pos > 0.0 else -1.0
            if cur_dir == d:
                signal[i] = pos
                continue

        # Gate: ADX
        gate_pass = True
        if f_adx:
            if (
                adx_v[i - 1] < adx_thr
                or f_adx_slope
                and adx_v[i - 1] <= adx_v[i - 2]
                or d > 0.0
                and pdi[i - 1] <= mdi[i - 1]
                or d < 0.0
                and mdi[i - 1] <= pdi[i - 1]
            ):
                gate_pass = False

        # Gate: RSI
        if gate_pass and f_rsi and (rsi_v[i - 1] <= rsi_lo or rsi_v[i - 1] >= rsi_hi):
            gate_pass = False

        # Gate: MA
        if gate_pass and f_ma:
            if d > 0.0:
                if not (c_1 > ema_f[i - 1] and ema_f[i - 1] >= ema_s[i - 1]):
                    gate_pass = False
            else:
                if not (c_1 < ema_f[i - 1] and ema_f[i - 1] <= ema_s[i - 1]):
                    gate_pass = False

        # Gate: Body
        if gate_pass and f_body:
            a1 = atr_v[i - 1]
            if a1 <= 0.0 or abs(c_1 - open_arr[i - 1]) > body_m * a1:
                gate_pass = False

        # Confirmation: MACD
        if gate_pass and f_macd and i >= 3:
            h1 = macd_h[i - 1]
            h2 = macd_h[i - 2]
            if d > 0.0:
                if not (h1 > 0.0 and h1 > h2):
                    gate_pass = False
            else:
                if not (h1 < 0.0 and h1 < h2):
                    gate_pass = False

        # Confirmation: Volume
        if gate_pass and f_vol and vol_ma[i - 1] > 0.0 and volume[i - 1] <= vol_m * vol_ma[i - 1]:
            gate_pass = False

        # Confirmation: Session
        if gate_pass and f_session:
            h = hour[i]
            if asian_s <= asian_e:
                if h >= asian_s and h < asian_e:
                    gate_pass = False
            else:
                if h >= asian_s or h < asian_e:
                    gate_pass = False

        # Confirmation: Momentum (Donchian/CCI)
        if gate_pass and f_momentum and i >= 3:
            donch_ok = False
            cci_ok = False
            if d > 0.0:
                if donch_hi[i - 2] > 0.0 and c_1 > donch_hi[i - 2]:
                    donch_ok = True
                if cci_v[i - 1] > cci_thr:
                    cci_ok = True
            else:
                if donch_lo[i - 2] > 0.0 and c_1 < donch_lo[i - 2]:
                    donch_ok = True
                if cci_v[i - 1] < -cci_thr:
                    cci_ok = True
            if not (donch_ok or cci_ok):
                gate_pass = False

        if not gate_pass:
            signal[i] = pos
            continue

        # ---- Entry/Reversal accepted ----
        pos = entry_sz * d
        entry_px = c_1
        bars_in = 0
        tp1_hit = False
        be_done = False
        addon_ct = 0
        best_px = high[i] if d > 0.0 else low[i]

        a_val = atr_v[i - 1] if atr_v[i - 1] > 0.0 else 1.0

        # Compute SL
        sl = 0.0
        if f_atr_sl:
            if d > 0.0:
                sl = entry_px - atr_sl_m * a_val
            else:
                sl = entry_px + atr_sl_m * a_val

        if f_struct_sl and s_look >= 1:
            start_j = max(0, i - s_look)
            if d > 0.0:
                swing = low[start_j]
                for j in range(start_j + 1, i):
                    if low[j] < swing:
                        swing = low[j]
                struct_sl = swing - s_pad * a_val
                if sl == 0.0 or struct_sl < sl:
                    sl = struct_sl
            else:
                swing = high[start_j]
                for j in range(start_j + 1, i):
                    if high[j] > swing:
                        swing = high[j]
                struct_sl = swing + s_pad * a_val
                if sl == 0.0 or struct_sl > sl:
                    sl = struct_sl

        initial_sl = sl

        # Compute TP
        tp1_px = 0.0
        tp2_px = 0.0
        if f_tp and sl != 0.0:
            r = abs(entry_px - sl)
            if r > 0.0:
                if d > 0.0:
                    tp1_px = entry_px + tp1r * r
                    tp2_px = entry_px + tp2r * r
                else:
                    tp1_px = entry_px - tp1r * r
                    tp2_px = entry_px - tp2r * r

        signal[i] = pos

    return signal


if njit is not None:
    _v4_loop_numba = njit(cache=True)(_v4_loop_py)
else:
    _v4_loop_numba = None


def _v4_loop(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_arr: np.ndarray,
    volume: np.ndarray,
    sar: np.ndarray,
    atr_v: np.ndarray,
    adx_v: np.ndarray,
    pdi: np.ndarray,
    mdi: np.ndarray,
    rsi_v: np.ndarray,
    ema_f: np.ndarray,
    ema_s: np.ndarray,
    macd_h: np.ndarray,
    vol_ma: np.ndarray,
    donch_hi: np.ndarray,
    donch_lo: np.ndarray,
    cci_v: np.ndarray,
    hour: np.ndarray,
    flags: np.ndarray,
    params: np.ndarray,
    warmup: int,
) -> np.ndarray:
    if _v4_loop_numba is not None:
        return _v4_loop_numba(
            close,
            high,
            low,
            open_arr,
            volume,
            sar,
            atr_v,
            adx_v,
            pdi,
            mdi,
            rsi_v,
            ema_f,
            ema_s,
            macd_h,
            vol_ma,
            donch_hi,
            donch_lo,
            cci_v,
            hour,
            flags,
            params,
            warmup,
        )
    return _v4_loop_py(
        close,
        high,
        low,
        open_arr,
        volume,
        sar,
        atr_v,
        adx_v,
        pdi,
        mdi,
        rsi_v,
        ema_f,
        ema_s,
        macd_h,
        vol_ma,
        donch_hi,
        donch_lo,
        cci_v,
        hour,
        flags,
        params,
        warmup,
    )


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


def _extract_hours(timestamps: np.ndarray) -> np.ndarray:
    """Extract UTC hour from a timestamp array."""
    try:
        import pandas as pd

        idx = pd.DatetimeIndex(timestamps)
        return idx.hour.to_numpy(dtype=np.int32)
    except Exception:
        try:
            ts_ns = timestamps.astype("datetime64[ns]")
            ts_h = ts_ns.astype("datetime64[h]")
            ts_d = ts_ns.astype("datetime64[D]")
            return ((ts_h - ts_d) / np.timedelta64(1, "h")).astype(np.int32)
        except Exception:
            return np.zeros(timestamps.shape[0], dtype=np.int32)


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lo = value.strip().lower()
        if lo in {"1", "true", "yes", "on"}:
            return True
        if lo in {"0", "false", "no", "off"}:
            return False
    return default


@register_strategy("psar_trade_assist_v4")
class PsarTradeAssistV4Strategy(BaseStrategy):
    """Structured PSAR strategy with framework selection, risk-mode presets,
    and Numba-accelerated trade-management state machine."""

    def parameters(self) -> dict[str, ParameterSpace]:
        return {
            "framework": ParameterSpace(kind="choice", choices=["F1", "F2", "F3", "F4", "F5"]),
            "risk_mode": ParameterSpace(kind="choice", choices=["baseline", "conservative", "standard", "aggressive"]),
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
            "enable_macd": ParameterSpace(kind="choice", choices=[0, 1]),
            "macd_fast": ParameterSpace(kind="int", low=8, high=16, step=2),
            "macd_slow": ParameterSpace(kind="int", low=20, high=30, step=2),
            "macd_signal": ParameterSpace(kind="int", low=7, high=12, step=1),
            "enable_vol_filter": ParameterSpace(kind="choice", choices=[0, 1]),
            "vol_ma_period": ParameterSpace(kind="int", low=10, high=30, step=5),
            "vol_ma_mult": ParameterSpace(kind="float", low=1.0, high=2.0, step=0.25),
            "enable_session": ParameterSpace(kind="choice", choices=[0, 1]),
            "enable_momentum": ParameterSpace(kind="choice", choices=[0, 1]),
            "donchian_period": ParameterSpace(kind="int", low=10, high=30, step=5),
            "cci_period": ParameterSpace(kind="int", low=10, high=20, step=2),
            "cci_threshold": ParameterSpace(kind="float", low=50.0, high=150.0, step=25.0),
            "atr_sl_mult": ParameterSpace(kind="float", low=1.0, high=3.0, step=0.25),
            "structure_lookback": ParameterSpace(kind="int", low=3, high=10, step=1),
            "structure_pad": ParameterSpace(kind="float", low=0.1, high=0.5, step=0.1),
            "tp1_r": ParameterSpace(kind="float", low=0.8, high=1.5, step=0.1),
            "tp2_r": ParameterSpace(kind="float", low=1.5, high=3.0, step=0.25),
            "trailing_atr_mult": ParameterSpace(kind="float", low=1.0, high=2.5, step=0.25),
            "partial_ratio": ParameterSpace(kind="float", low=0.3, high=0.6, step=0.1),
            "be_trigger_r": ParameterSpace(kind="float", low=0.3, high=0.8, step=0.1),
            "be_buffer_r": ParameterSpace(kind="float", low=0.05, high=0.2, step=0.05),
            "addon_size": ParameterSpace(kind="float", low=0.3, high=0.5, step=0.1),
            "entry_size": ParameterSpace(kind="float", low=0.5, high=1.0, step=0.1),
            "time_stop_bars": ParameterSpace(kind="int", low=15, high=60, step=5),
            "no_progress_r": ParameterSpace(kind="float", low=0.1, high=0.5, step=0.1),
        }

    def _resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        fw = str(params.get("framework", "F1")).upper()
        rm = str(params.get("risk_mode", "baseline")).lower()

        fw_flags = FRAMEWORKS.get(fw, {})
        rm_flags = RISK_MODES.get(rm, {})

        cfg: dict[str, Any] = {
            "framework": fw,
            "risk_mode": rm,
            # Entry (defaults off, overridden by framework)
            "enable_adx": _as_bool(fw_flags.get("enable_adx", params.get("enable_adx", 0)), False),
            "enable_rsi": _as_bool(fw_flags.get("enable_rsi", params.get("enable_rsi", 0)), False),
            "enable_ma": _as_bool(fw_flags.get("enable_ma", params.get("enable_ma", 0)), False),
            "enable_body": _as_bool(fw_flags.get("enable_body", params.get("enable_body", 0)), False),
            "enable_adaptive_step": _as_bool(params.get("enable_adaptive_step", 1), True),
            "require_adx_slope": _as_bool(params.get("require_adx_slope", 1), True),
            # Confirmations
            "enable_macd": _as_bool(params.get("enable_macd", 0), False),
            "enable_vol_filter": _as_bool(params.get("enable_vol_filter", 0), False),
            "enable_session": _as_bool(params.get("enable_session", 0), False),
            "enable_momentum": _as_bool(params.get("enable_momentum", 0), False),
            # Risk mgmt (defaults off, overridden by risk_mode)
            "enable_atr_sl": _as_bool(rm_flags.get("enable_atr_sl", params.get("enable_atr_sl", 0)), False),
            "enable_structure_sl": _as_bool(
                rm_flags.get("enable_structure_sl", params.get("enable_structure_sl", 0)), False
            ),
            "enable_tp": _as_bool(rm_flags.get("enable_tp", params.get("enable_tp", 0)), False),
            "enable_trailing_psar": _as_bool(
                rm_flags.get("enable_trailing_psar", params.get("enable_trailing_psar", 0)), False
            ),
            "enable_trailing_atr": _as_bool(
                rm_flags.get("enable_trailing_atr", params.get("enable_trailing_atr", 0)), False
            ),
            "enable_partial_close": _as_bool(
                rm_flags.get("enable_partial_close", params.get("enable_partial_close", 0)), False
            ),
            "enable_breakeven": _as_bool(rm_flags.get("enable_breakeven", params.get("enable_breakeven", 0)), False),
            "enable_addon": _as_bool(rm_flags.get("enable_addon", params.get("enable_addon", 0)), False),
            "enable_time_stop": _as_bool(rm_flags.get("enable_time_stop", params.get("enable_time_stop", 0)), False),
            # Continuous params
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
            "macd_fast": int(params.get("macd_fast", 12)),
            "macd_slow": int(params.get("macd_slow", 26)),
            "macd_signal": int(params.get("macd_signal", 9)),
            "vol_ma_period": int(params.get("vol_ma_period", 20)),
            "vol_ma_mult": float(params.get("vol_ma_mult", 1.5)),
            "donchian_period": int(params.get("donchian_period", 20)),
            "cci_period": int(params.get("cci_period", 14)),
            "cci_threshold": float(params.get("cci_threshold", 100.0)),
            "atr_sl_mult": float(params.get("atr_sl_mult", rm_flags.get("atr_sl_mult", 1.5))),
            "structure_lookback": int(params.get("structure_lookback", 6)),
            "structure_pad": float(params.get("structure_pad", 0.20)),
            "tp1_r": float(params.get("tp1_r", 1.0)),
            "tp2_r": float(params.get("tp2_r", 2.0)),
            "trailing_atr_mult": float(params.get("trailing_atr_mult", 2.0)),
            "partial_ratio": float(params.get("partial_ratio", 0.5)),
            "be_trigger_r": float(params.get("be_trigger_r", 0.5)),
            "be_buffer_r": float(params.get("be_buffer_r", 0.1)),
            "addon_size": float(params.get("addon_size", 0.4)),
            "entry_size": float(rm_flags.get("entry_size", params.get("entry_size", 1.0))),
            "time_stop_bars": int(params.get("time_stop_bars", 30)),
            "no_progress_r": float(params.get("no_progress_r", 0.3)),
        }
        return cfg

    def _pack(self, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        flags = np.zeros(N_FLAGS, dtype=np.int32)
        flags[F_ADX] = int(cfg["enable_adx"])
        flags[F_RSI] = int(cfg["enable_rsi"])
        flags[F_MA] = int(cfg["enable_ma"])
        flags[F_BODY] = int(cfg["enable_body"])
        flags[F_ADAPTIVE] = int(cfg["enable_adaptive_step"])
        flags[F_ADX_SLOPE] = int(cfg["require_adx_slope"])
        flags[F_MACD] = int(cfg["enable_macd"])
        flags[F_VOL] = int(cfg["enable_vol_filter"])
        flags[F_SESSION] = int(cfg["enable_session"])
        flags[F_MOMENTUM] = int(cfg["enable_momentum"])
        flags[F_ATR_SL] = int(cfg["enable_atr_sl"])
        flags[F_STRUCT_SL] = int(cfg["enable_structure_sl"])
        flags[F_TP] = int(cfg["enable_tp"])
        flags[F_TRAIL_PSAR] = int(cfg["enable_trailing_psar"])
        flags[F_TRAIL_ATR] = int(cfg["enable_trailing_atr"])
        flags[F_PARTIAL] = int(cfg["enable_partial_close"])
        flags[F_BE] = int(cfg["enable_breakeven"])
        flags[F_ADDON] = int(cfg["enable_addon"])
        flags[F_TIMESTOP] = int(cfg["enable_time_stop"])

        p = np.zeros(N_PARAMS, dtype=np.float64)
        p[P_ADX_THRESH] = cfg["adx_threshold"]
        p[P_RSI_LO] = cfg["rsi_lower"]
        p[P_RSI_HI] = cfg["rsi_upper"]
        p[P_BODY_MULT] = cfg["body_atr_mult"]
        p[P_VOL_MULT] = cfg["vol_ma_mult"]
        p[P_CCI_THRESH] = cfg["cci_threshold"]
        p[P_ATR_SL_MULT] = cfg["atr_sl_mult"]
        p[P_STRUCT_LOOK] = float(cfg["structure_lookback"])
        p[P_STRUCT_PAD] = cfg["structure_pad"]
        p[P_TP1_R] = cfg["tp1_r"]
        p[P_TP2_R] = cfg["tp2_r"]
        p[P_TRAIL_ATR_MULT] = cfg["trailing_atr_mult"]
        p[P_PARTIAL_RATIO] = cfg["partial_ratio"]
        p[P_BE_TRIG] = cfg["be_trigger_r"]
        p[P_BE_BUF] = cfg["be_buffer_r"]
        p[P_ADDON_SIZE] = cfg["addon_size"]
        p[P_TIMESTOP_BARS] = float(cfg["time_stop_bars"])
        p[P_NOPROG_R] = cfg["no_progress_r"]
        p[P_ENTRY_SIZE] = cfg["entry_size"]
        p[P_ASIAN_START] = 0.0
        p[P_ASIAN_END] = 8.0
        return flags, p

    # -----------------------------------------------------------------------

    def generate_signals(
        self,
        data: Any,
        params: dict[str, Any],
        cache: _IndicatorCache | None = None,
    ) -> np.ndarray:
        cfg = self._resolve(params)

        close_np = np.asarray(to_numpy(data.close), dtype=np.float64)
        high_np = np.asarray(to_numpy(data.high), dtype=np.float64)
        low_np = np.asarray(to_numpy(data.low), dtype=np.float64)
        open_np = np.asarray(to_numpy(data.open), dtype=np.float64)
        vol_np = np.asarray(to_numpy(data.volume), dtype=np.float64)
        was_1d = close_np.ndim == 1
        if was_1d:
            close_np = close_np.reshape(-1, 1)
            high_np = high_np.reshape(-1, 1)
            low_np = low_np.reshape(-1, 1)
            open_np = open_np.reshape(-1, 1)
            vol_np = vol_np.reshape(-1, 1)

        n_bars, n_assets = close_np.shape
        warmup = (
            max(
                cfg["atr_ratio_window"],
                cfg["ma_slow"],
                cfg["atr_period"],
                cfg["adx_period"],
                cfg["rsi_period"],
                cfg["donchian_period"],
                cfg["cci_period"],
                cfg["vol_ma_period"],
                cfg["macd_slow"],
            )
            + 10
        )
        if n_bars < max(warmup, 3):
            out = np.zeros((n_bars, n_assets), dtype=np.float64)
            return out.ravel() if was_1d else out

        data_key = id(data)

        def _cached(name: str, key_parts: tuple[Any, ...], compute):  # noqa: ANN001
            if cache is None:
                return compute()
            return cache.get_or_compute((name, data_key) + key_parts, compute)

        # -- Compute indicators --
        atr_v = np.asarray(
            _cached("atr", (cfg["atr_period"],), lambda: atr(high_np, low_np, close_np, window=cfg["atr_period"])),
            dtype=np.float64,
        )

        atr_ratio = np.asarray(
            _cached(
                "atr_ratio",
                (cfg["atr_period"], cfg["atr_ratio_window"]),
                lambda: _calc_atr_ratio(np.asarray(to_numpy(atr_v), dtype=np.float64), cfg["atr_ratio_window"]),
            ),
            dtype=np.float64,
        )

        sar_mid = np.asarray(
            _cached(
                "psar",
                (cfg["psar_step_mid"], cfg["psar_max"]),
                lambda: parabolic_sar(high_np, low_np, close_np, step=cfg["psar_step_mid"], maximum=cfg["psar_max"]),
            ),
            dtype=np.float64,
        )

        if cfg["enable_adaptive_step"]:
            sar_low = np.asarray(
                _cached(
                    "psar",
                    (cfg["psar_step_low"], cfg["psar_max"]),
                    lambda: parabolic_sar(
                        high_np, low_np, close_np, step=cfg["psar_step_low"], maximum=cfg["psar_max"]
                    ),
                ),
                dtype=np.float64,
            )
            sar_high = np.asarray(
                _cached(
                    "psar",
                    (cfg["psar_step_high"], cfg["psar_max"]),
                    lambda: parabolic_sar(
                        high_np, low_np, close_np, step=cfg["psar_step_high"], maximum=cfg["psar_max"]
                    ),
                ),
                dtype=np.float64,
            )
            low_mask = atr_ratio < cfg["atr_ratio_low"]
            high_mask = atr_ratio > cfg["atr_ratio_high"]
            sar_sel = np.where(low_mask, sar_low, np.where(high_mask, sar_high, sar_mid))
        else:
            sar_sel = sar_mid

        adx_v, pdi_v, mdi_v = _cached(
            "adx", (cfg["adx_period"],), lambda: adx(high_np, low_np, close_np, window=cfg["adx_period"])
        )
        adx_v = np.asarray(adx_v, dtype=np.float64)
        pdi_v = np.asarray(pdi_v, dtype=np.float64)
        mdi_v = np.asarray(mdi_v, dtype=np.float64)

        rsi_v = np.asarray(
            _cached("rsi", (cfg["rsi_period"],), lambda: rsi(close_np, window=cfg["rsi_period"])), dtype=np.float64
        )

        ema_f = np.asarray(
            _cached("ema", (cfg["ma_fast"],), lambda: ema(close_np, span=cfg["ma_fast"])), dtype=np.float64
        )
        ema_s = np.asarray(
            _cached("ema", (cfg["ma_slow"],), lambda: ema(close_np, span=cfg["ma_slow"])), dtype=np.float64
        )

        _, _, macd_h = _cached(
            "macd",
            (cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"]),
            lambda: macd(close_np, fast=cfg["macd_fast"], slow=cfg["macd_slow"], signal=cfg["macd_signal"]),
        )
        macd_h = np.asarray(macd_h, dtype=np.float64)

        vol_ma_arr = np.asarray(
            _cached("sma_vol", (cfg["vol_ma_period"],), lambda: sma(vol_np, window=cfg["vol_ma_period"])),
            dtype=np.float64,
        )

        d_hi, d_lo = _cached(
            "donchian", (cfg["donchian_period"],), lambda: donchian(high_np, low_np, period=cfg["donchian_period"])
        )
        d_hi = np.asarray(d_hi, dtype=np.float64)
        d_lo = np.asarray(d_lo, dtype=np.float64)

        cci_v = np.asarray(
            _cached("cci", (cfg["cci_period"],), lambda: cci(high_np, low_np, close_np, period=cfg["cci_period"])),
            dtype=np.float64,
        )

        hour_arr = _extract_hours(data.timestamps)

        # NaN -> 0 for safety in Numba
        for arr in (atr_v, adx_v, pdi_v, mdi_v, rsi_v, ema_f, ema_s, macd_h, vol_ma_arr, d_hi, d_lo, cci_v, sar_sel):
            np.nan_to_num(arr, copy=False, nan=0.0)

        flags, p = self._pack(cfg)

        # Per-column loop (usually n_assets == 1)
        out = np.zeros((n_bars, n_assets), dtype=np.float64)
        for col in range(n_assets):
            out[:, col] = _v4_loop(
                close_np[:, col],
                high_np[:, col],
                low_np[:, col],
                open_np[:, col],
                vol_np[:, col],
                sar_sel[:, col],
                atr_v[:, col],
                adx_v[:, col],
                pdi_v[:, col],
                mdi_v[:, col],
                rsi_v[:, col],
                ema_f[:, col],
                ema_s[:, col],
                macd_h[:, col],
                vol_ma_arr[:, col],
                d_hi[:, col],
                d_lo[:, col],
                cci_v[:, col],
                hour_arr,
                flags,
                p,
                warmup,
            )

        return out.ravel() if was_1d else out
