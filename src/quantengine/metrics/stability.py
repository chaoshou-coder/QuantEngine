"""Walk-forward stability evaluation, permutation testing, and regime analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantengine.data.gpu_backend import estimate_max_batch_size, to_numpy, xp_from_array
from quantengine.data.loader import DataBundle
from quantengine.metrics.batch import batch_score
from quantengine.metrics.performance import (
    bar_win_rate,
    max_drawdown,
    sharpe_ratio,
)


@dataclass
class WindowResult:
    window_id: int
    train_sharpe: float
    test_sharpe: float
    train_win_rate: float
    test_win_rate: float
    train_max_dd: float
    test_max_dd: float
    decay_sharpe: float


@dataclass
class StabilityReport:
    windows: list[WindowResult]
    sharpe_mean: float
    sharpe_std: float
    stability_ratio: float
    worst_window_sharpe: float
    decay_mean: float
    is_stable: bool
    threshold_used: float


@dataclass
class PermutationResult:
    actual_sharpe: float
    p_value: float
    n_permutations: int
    is_significant: bool


@dataclass
class MultipleTestingResult:
    method: str
    alpha: float
    raw_p_values: list[float]
    adjusted_p_values: list[float]
    rejected: list[bool]
    significant_count: int
    summary: str


@dataclass
class ParameterSensitivityPoint:
    param_name: str
    base_value: float
    minus_value: float
    plus_value: float
    minus_score: float
    base_score: float
    plus_score: float


@dataclass
class RegimeResult:
    regime_name: str
    n_bars: int
    sharpe: float
    win_rate: float
    max_dd: float


SignalFn = Callable[[DataBundle, dict[str, Any]], np.ndarray]
SimFn = Callable[[DataBundle, np.ndarray], tuple[np.ndarray, np.ndarray]]
SimBatchFn = Callable[[DataBundle, Any], tuple[Any, Any]]


def build_walk_forward_windows(
    n_bars: int,
    train_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[tuple[int, int, int, int]]:
    """Return list of (train_start, train_end, test_start, test_end) index tuples."""
    windows: list[tuple[int, int, int, int]] = []
    start = 0
    while start + train_bars + test_bars <= n_bars:
        t_start = start
        t_end = start + train_bars
        v_start = t_end
        v_end = min(t_end + test_bars, n_bars)
        windows.append((t_start, t_end, v_start, v_end))
        start += step_bars
    return windows


def _eval_one_window(
    data: DataBundle,
    signal_fn: SignalFn,
    sim_fn: SimFn,
    params: dict[str, Any],
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    window_id: int,
    risk_free_rate: float,
    periods_per_year: int,
) -> WindowResult:
    train_data = data.slice_by_index(train_start, train_end)
    test_data = data.slice_by_index(test_start, test_end)

    train_signal = signal_fn(train_data, params)
    train_eq, train_ret = sim_fn(train_data, train_signal)

    test_signal = signal_fn(test_data, params)
    test_eq, test_ret = sim_fn(test_data, test_signal)

    train_eq_np = np.asarray(train_eq, dtype=np.float64).ravel()
    train_ret_np = np.asarray(train_ret, dtype=np.float64).ravel()
    test_eq_np = np.asarray(test_eq, dtype=np.float64).ravel()
    test_ret_np = np.asarray(test_ret, dtype=np.float64).ravel()

    tr_sharpe = sharpe_ratio(train_ret_np, risk_free_rate, periods_per_year)
    te_sharpe = sharpe_ratio(test_ret_np, risk_free_rate, periods_per_year)
    tr_wr = bar_win_rate(train_ret_np)
    te_wr = bar_win_rate(test_ret_np)
    tr_dd = max_drawdown(train_eq_np)
    te_dd = max_drawdown(test_eq_np)

    if abs(tr_sharpe) > 1e-12:
        decay = (tr_sharpe - te_sharpe) / abs(tr_sharpe)
    else:
        decay = 0.0

    return WindowResult(
        window_id=window_id,
        train_sharpe=tr_sharpe,
        test_sharpe=te_sharpe,
        train_win_rate=tr_wr,
        test_win_rate=te_wr,
        train_max_dd=tr_dd,
        test_max_dd=te_dd,
        decay_sharpe=decay,
    )


def walk_forward_evaluate(
    data: DataBundle,
    signal_fn: SignalFn,
    sim_fn: SimFn,
    params: dict[str, Any],
    train_bars: int,
    test_bars: int,
    step_bars: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 525_600,
    stability_threshold: float = 1.0,
) -> StabilityReport:
    """Run walk-forward validation and compute stability metrics.

    Args:
        data: Full data bundle.
        signal_fn: (data_slice, params) -> signal array (n_bars, n_assets).
        sim_fn: (data_slice, signal) -> (equity_1d, returns_1d).
        params: Strategy parameters dict.
        train_bars: Training window size in bars.
        test_bars: Test window size in bars.
        step_bars: Step size between windows.
        stability_threshold: Minimum stability_ratio to be considered stable.

    Returns:
        StabilityReport with per-window results and aggregate stability metrics.
    """
    n_bars = data.close.shape[0]
    windows = build_walk_forward_windows(n_bars, train_bars, test_bars, step_bars)

    if not windows:
        return StabilityReport(
            windows=[],
            sharpe_mean=0.0,
            sharpe_std=0.0,
            stability_ratio=0.0,
            worst_window_sharpe=0.0,
            decay_mean=0.0,
            is_stable=False,
            threshold_used=stability_threshold,
        )

    results: list[WindowResult] = []
    for idx, (ts, te, vs, ve) in enumerate(windows):
        wr = _eval_one_window(
            data,
            signal_fn,
            sim_fn,
            params,
            ts,
            te,
            vs,
            ve,
            idx,
            risk_free_rate,
            periods_per_year,
        )
        results.append(wr)

    test_sharpes = np.array([w.test_sharpe for w in results], dtype=np.float64)
    decays = np.array([w.decay_sharpe for w in results], dtype=np.float64)

    s_mean = float(np.mean(test_sharpes))
    s_std = float(np.std(test_sharpes, ddof=1)) if len(test_sharpes) > 1 else 0.0
    s_ratio = s_mean / (s_std + 1e-12)
    worst = float(np.min(test_sharpes))
    d_mean = float(np.mean(decays))

    return StabilityReport(
        windows=results,
        sharpe_mean=s_mean,
        sharpe_std=s_std,
        stability_ratio=s_ratio,
        worst_window_sharpe=worst,
        decay_mean=d_mean,
        is_stable=(s_ratio >= stability_threshold and worst > 0.0),
        threshold_used=stability_threshold,
    )


def _as_tensor(signals: list[Any] | np.ndarray, n_bars: int, n_assets: int) -> Any:
    xp = xp_from_array(signals) if hasattr(signals, "__array__") else np
    if hasattr(signals, "ndim"):
        signal_arr = xp.asarray(signals, dtype=float)
        if signal_arr.ndim == 2:
            signal_arr = signal_arr[:, :, None]
        if signal_arr.ndim != 3:
            raise ValueError(f"signals 维度错误: shape={signal_arr.shape}, expected 2D or 3D")
        if signal_arr.shape[0] != n_bars or signal_arr.shape[1] != n_assets:
            raise ValueError(
                f"signals 形状错误: received={signal_arr.shape}, expected=({n_bars}, {n_assets}, n_params)"
            )
        return signal_arr

    if isinstance(signals, (list, tuple)) and len(signals) == 0:
        return np.empty((n_bars, n_assets, 0), dtype=np.float64)

    first = xp_from_array(signals[0]).asarray(signals[0], dtype=float)
    if first.ndim != 2:
        raise ValueError(f"signal[0] 维度错误: shape={first.shape}, expected 2D")
    if first.shape != (n_bars, n_assets):
        raise ValueError(f"signal[0] 形状错误: received={first.shape}, expected=({n_bars}, {n_assets})")
    stacked = xp_from_array(first).stack([xp_from_array(s).asarray(s, dtype=float) for s in signals], axis=2)
    if stacked.shape[2] != len(signals):
        raise ValueError(f"signal 堆叠失败: expected={len(signals)}，actual={stacked.shape[2]}")
    return stacked


def _to_2d_results(result: Any, n_bars: int) -> Any:
    arr = xp_from_array(result).asarray(result)
    if arr.ndim == 1:
        if arr.shape[0] != n_bars:
            raise ValueError(f"仿真结果维度错误: shape={arr.shape}, expected bars={n_bars}")
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"仿真结果维度错误: shape={arr.shape}, expected 2D")
    return arr


def walk_forward_evaluate_batched(
    data: DataBundle,
    signal_tensor: list[np.ndarray] | np.ndarray,
    params_list: list[dict[str, Any]],
    sim_batch_fn: SimBatchFn,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 525_600,
    stability_threshold: float = 1.0,
) -> list[StabilityReport]:
    """Batch evaluate multiple parameter sets by window and combo tensor.

    The function slices a pre-generated signal tensor into walk-forward windows and
    runs one `simulate` call per window, returning one `StabilityReport` per
    parameter set.
    """
    n_bars, n_assets = data.close.shape
    n_combos = len(params_list)
    if n_bars == 0:
        return []
    if n_combos == 0:
        return []

    sig_tensor = _as_tensor(signals=signal_tensor, n_bars=n_bars, n_assets=n_assets)
    if sig_tensor.shape[2] != n_combos:
        raise ValueError(f"signals 与参数数量不一致: signals={sig_tensor.shape[2]}, params={n_combos}")

    windows = build_walk_forward_windows(n_bars, train_bars, test_bars, step_bars)
    if not windows:
        return [
            StabilityReport(
                windows=[],
                sharpe_mean=0.0,
                sharpe_std=0.0,
                stability_ratio=0.0,
                worst_window_sharpe=0.0,
                decay_mean=0.0,
                is_stable=False,
                threshold_used=stability_threshold,
            )
            for _ in range(n_combos)
        ]

    window_histories: list[list[WindowResult]] = [[] for _ in range(n_combos)]
    combo_batch = max(
        1,
        min(
            n_combos,
            estimate_max_batch_size(n_bars=n_bars, n_assets=n_assets),
        ),
    )

    for window_id, (ts, te, vs, ve) in enumerate(windows):
        train_data = data.slice_by_index(ts, te)
        test_data = data.slice_by_index(vs, ve)
        train_window_signal = sig_tensor[ts:te]
        test_window_signal = sig_tensor[vs:ve]

        if train_window_signal.shape[2] == 0 or test_window_signal.shape[2] == 0:
            continue

        for start in range(0, n_combos, combo_batch):
            end = min(start + combo_batch, n_combos)
            train_signals = train_window_signal[:, :, start:end]
            test_signals = test_window_signal[:, :, start:end]

            train_eq, train_ret = sim_batch_fn(train_data, train_signals)
            test_eq, test_ret = sim_batch_fn(test_data, test_signals)

            train_eq_2d = _to_2d_results(train_eq, train_data.close.shape[0])
            train_ret_2d = _to_2d_results(train_ret, train_data.close.shape[0])
            test_eq_2d = _to_2d_results(test_eq, test_data.close.shape[0])
            test_ret_2d = _to_2d_results(test_ret, test_data.close.shape[0])

            market_train = None
            market_test = None
            if n_combos > 0:
                train_close_np = np.asarray(to_numpy(train_data.close), dtype=np.float64).reshape(-1)
                test_close_np = np.asarray(to_numpy(test_data.close), dtype=np.float64).reshape(-1)
                market_train = np.diff(np.log(train_close_np), prepend=train_close_np[0])
                market_test = np.diff(np.log(test_close_np), prepend=test_close_np[0])

            train_sharpe = batch_score(
                equity_2d=train_eq_2d,
                returns_2d=train_ret_2d,
                metric="sharpe",
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                market_returns=market_train,
            )
            test_sharpe = batch_score(
                equity_2d=test_eq_2d,
                returns_2d=test_ret_2d,
                metric="sharpe",
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                market_returns=market_test,
            )
            train_wr = batch_score(
                equity_2d=train_eq_2d,
                returns_2d=train_ret_2d,
                metric="win_rate",
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
            test_wr = batch_score(
                equity_2d=test_eq_2d,
                returns_2d=test_ret_2d,
                metric="win_rate",
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
            train_dd = batch_score(
                equity_2d=train_eq_2d,
                returns_2d=train_ret_2d,
                metric="max_drawdown",
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
            test_dd = batch_score(
                equity_2d=test_eq_2d,
                returns_2d=test_ret_2d,
                metric="max_drawdown",
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )

            for local_idx in range(start, end):
                local = local_idx - start
                tr_sharpe = float(train_sharpe[local])
                te_sharpe = float(test_sharpe[local])
                tr_wr = float(train_wr[local])
                te_wr = float(test_wr[local])
                tr_dd = float(train_dd[local])
                te_dd = float(test_dd[local])
                decay = (tr_sharpe - te_sharpe) / abs(tr_sharpe) if abs(tr_sharpe) > 1e-12 else 0.0
                window_histories[local_idx].append(
                    WindowResult(
                        window_id=window_id,
                        train_sharpe=tr_sharpe,
                        test_sharpe=te_sharpe,
                        train_win_rate=tr_wr,
                        test_win_rate=te_wr,
                        train_max_dd=tr_dd,
                        test_max_dd=te_dd,
                        decay_sharpe=float(decay),
                    )
                )

    reports: list[StabilityReport] = []
    for idx in range(n_combos):
        windows = window_histories[idx]
        if not windows:
            reports.append(
                StabilityReport(
                    windows=[],
                    sharpe_mean=0.0,
                    sharpe_std=0.0,
                    stability_ratio=0.0,
                    worst_window_sharpe=0.0,
                    decay_mean=0.0,
                    is_stable=False,
                    threshold_used=stability_threshold,
                )
            )
            continue

        test_sharpes = np.array([w.test_sharpe for w in windows], dtype=np.float64)
        decays = np.array([w.decay_sharpe for w in windows], dtype=np.float64)
        s_mean = float(np.mean(test_sharpes))
        s_std = float(np.std(test_sharpes, ddof=1)) if len(test_sharpes) > 1 else 0.0
        s_ratio = s_mean / (s_std + 1e-12)
        worst = float(np.min(test_sharpes))
        d_mean = float(np.mean(decays))
        reports.append(
            StabilityReport(
                windows=windows,
                sharpe_mean=s_mean,
                sharpe_std=s_std,
                stability_ratio=s_ratio,
                worst_window_sharpe=worst,
                decay_mean=d_mean,
                is_stable=(s_ratio >= stability_threshold and worst > 0.0),
                threshold_used=stability_threshold,
            )
        )

    return reports


def benjamini_hochberg_correction(
    p_values: np.ndarray | list[float],
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Apply Benjamini-Hochberg correction and return detailed result."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha 必须在 (0, 1) 区间内，当前={alpha}")

    pvals = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if pvals.size == 0:
        return MultipleTestingResult(
            method="benjamini-hochberg",
            alpha=alpha,
            raw_p_values=[],
            adjusted_p_values=[],
            rejected=[],
            significant_count=0,
            summary=f"Benjamini-Hochberg 校正完成：无可用假设检验结果（alpha={alpha:.4f}）。",
        )
    if np.any(np.isnan(pvals)):
        raise ValueError("p_values 包含 NaN")
    if np.any((pvals < 0.0) | (pvals > 1.0)):
        raise ValueError("p_values 必须在 [0, 1] 区间内")

    m = pvals.shape[0]
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adjusted_sorted = np.empty_like(sorted_p)
    running_min = 1.0
    for idx in range(m - 1, -1, -1):
        rank = idx + 1
        candidate = float(sorted_p[idx] * m / rank)
        running_min = min(running_min, candidate)
        adjusted_sorted[idx] = min(max(running_min, 0.0), 1.0)

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    rejected = adjusted <= alpha
    significant_count = int(np.sum(rejected))
    summary = f"Benjamini-Hochberg 校正完成：显著 {significant_count}/{m}，alpha={alpha:.4f}。"

    return MultipleTestingResult(
        method="benjamini-hochberg",
        alpha=alpha,
        raw_p_values=[float(item) for item in pvals.tolist()],
        adjusted_p_values=[float(item) for item in adjusted.tolist()],
        rejected=[bool(item) for item in rejected.tolist()],
        significant_count=significant_count,
        summary=summary,
    )


def parameter_sensitivity_analysis(
    base_params: dict[str, Any],
    evaluate_fn: Callable[[dict[str, Any]], float],
    perturbation: float = 0.1,
) -> list[ParameterSensitivityPoint]:
    """Evaluate +/- perturbation sensitivity for numeric parameters.

    Notes:
        - Uses multiplicative +/- perturbation for non-zero values.
        - Uses additive +/- perturbation for zero values to avoid no-op changes.
    """
    if perturbation <= 0.0:
        raise ValueError(f"perturbation 必须大于 0，当前={perturbation}")

    base_snapshot = dict(base_params)
    base_score = float(evaluate_fn(dict(base_snapshot)))
    points: list[ParameterSensitivityPoint] = []
    for key, value in base_snapshot.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            continue

        base_value = float(value)
        if abs(base_value) > 1e-12:
            minus_value = base_value * (1.0 - perturbation)
            plus_value = base_value * (1.0 + perturbation)
        else:
            minus_value = -perturbation
            plus_value = perturbation

        minus_params = dict(base_snapshot)
        plus_params = dict(base_snapshot)
        minus_params[key] = minus_value
        plus_params[key] = plus_value

        minus_score = float(evaluate_fn(minus_params))
        plus_score = float(evaluate_fn(plus_params))
        points.append(
            ParameterSensitivityPoint(
                param_name=str(key),
                base_value=base_value,
                minus_value=float(minus_value),
                plus_value=float(plus_value),
                minus_score=minus_score,
                base_score=base_score,
                plus_score=plus_score,
            )
        )

    return points


def permutation_test(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 525_600,
    n_permutations: int = 1000,
    seed: int = 42,
    significance: float = 0.05,
) -> PermutationResult:
    """Monte Carlo permutation test: shuffle returns and compare Sharpe."""
    ret = np.asarray(returns, dtype=np.float64).ravel()
    actual = sharpe_ratio(ret, risk_free_rate, periods_per_year)

    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(ret)
        s = sharpe_ratio(shuffled, risk_free_rate, periods_per_year)
        if s >= actual:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    return PermutationResult(
        actual_sharpe=actual,
        p_value=p_value,
        n_permutations=n_permutations,
        is_significant=p_value < significance,
    )


def regime_analysis(
    equity: np.ndarray,
    returns: np.ndarray,
    atr_values: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 525_600,
    n_regimes: int = 3,
) -> list[RegimeResult]:
    """Split data by ATR quantiles and compute per-regime metrics."""
    eq = np.asarray(equity, dtype=np.float64).ravel()
    ret = np.asarray(returns, dtype=np.float64).ravel()
    atr_v = np.asarray(atr_values, dtype=np.float64).ravel()

    n = min(eq.shape[0], ret.shape[0], atr_v.shape[0])
    eq, ret, atr_v = eq[:n], ret[:n], atr_v[:n]

    valid = ~np.isnan(atr_v)
    if valid.sum() < n_regimes * 10:
        return []

    quantiles = np.linspace(0, 1, n_regimes + 1)
    thresholds = np.nanquantile(atr_v[valid], quantiles)

    labels = ["low_vol", "mid_vol", "high_vol"] if n_regimes == 3 else [f"regime_{i}" for i in range(n_regimes)]

    results: list[RegimeResult] = []
    for i in range(n_regimes):
        lo = thresholds[i]
        hi = thresholds[i + 1] if i < n_regimes - 1 else np.inf
        if i == 0:
            mask = (atr_v >= lo) & (atr_v <= hi)
        else:
            mask = (atr_v > lo) & (atr_v <= hi)

        regime_ret = ret[mask]
        regime_eq = eq[mask]
        if regime_ret.shape[0] < 10:
            results.append(RegimeResult(labels[i], int(regime_ret.shape[0]), 0.0, 0.0, 0.0))
            continue

        s = sharpe_ratio(regime_ret, risk_free_rate, periods_per_year)
        wr = bar_win_rate(regime_ret)
        dd = max_drawdown(regime_eq)
        results.append(RegimeResult(labels[i], int(regime_ret.shape[0]), s, wr, dd))

    return results
