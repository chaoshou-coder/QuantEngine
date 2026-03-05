from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from quantengine.data.gpu_backend import get_backend_info
from quantengine.data.loader import DataBundle
from quantengine.engine.backtest import BacktestEngine
from quantengine.engine.portfolio import simulate_portfolio, simulate_portfolio_batch
from quantengine.metrics.batch import batch_score
from quantengine.metrics.performance import calculate_performance_metrics, market_returns_from_close
from quantengine.metrics.risk import calculate_risk_metrics
from quantengine.metrics.trade_analysis import calculate_trade_metrics
from quantengine.strategy.base import BaseStrategy

from .base import TrialResult, score_from_report

logger = logging.getLogger(__name__)

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


def _require_cupy() -> Any:
    if cp is None:
        raise RuntimeError("CuPy required for Sprint 0 Track B GPU batch path")
    return cp


def _data_to_gpu(data: DataBundle) -> DataBundle:
    cp_module = _require_cupy()
    backend = get_backend_info(requested="gpu", use_gpu=True)
    if backend.active != "gpu":
        raise RuntimeError("GPU 请求失败：未检测到可用 CUDA 设备")
    if data.backend.active == "gpu":
        return data
    return DataBundle(
        symbols=list(data.symbols),
        timestamps=data.timestamps,
        open=cp_module.asarray(data.open),
        high=cp_module.asarray(data.high),
        low=cp_module.asarray(data.low),
        close=cp_module.asarray(data.close),
        volume=cp_module.asarray(data.volume),
        backend=backend,
    )


def _signals_to_tensor(signals: list[Any] | tuple[Any, ...], dtype: Any = float):
    cp_module = _require_cupy()
    if not signals:
        return cp_module.empty((0, 0, 0), dtype=dtype)
    signal_arrays = [cp_module.asarray(signal, dtype=dtype) for signal in signals]
    first = signal_arrays[0]
    if first.ndim != 2:
        raise ValueError(f"signal shape must be 2D, got {first.shape}")
    for idx, signal in enumerate(signal_arrays):
        if signal.ndim != 2:
            raise ValueError(f"signal[{idx}] shape must be 2D, got {signal.shape}")
    return cp_module.stack(signal_arrays, axis=2)


def cleanup_gpu_memory() -> None:
    if cp is None:
        return
    if hasattr(cp, "get_default_memory_pool"):
        cp.get_default_memory_pool().free_all_blocks()
    pinned = getattr(cp, "get_default_pinned_memory_pool", None)
    if callable(pinned):
        pinned().free_all_blocks()


@dataclass
class _ReportProxy:
    performance: dict[str, float]
    risk: dict[str, float]
    trade_metrics: dict[str, float | str]


def _run_single_trial_worker(
    engine: BacktestEngine,
    data: DataBundle,
    strategy_type: type[BaseStrategy],
    params: dict[str, Any],
    metric: str,
) -> TrialResult:
    strategy = strategy_type()
    report = engine.run(data=data, strategy=strategy, params=params, record_trades=False)
    score = score_from_report(report, metric=metric)
    return TrialResult(params=params, score=score, report=report)


def _evaluate_signal_combo_worker(
    engine: BacktestEngine,
    data: DataBundle,
    signal: Any,
    params: dict[str, Any],
    metric: str,
) -> TrialResult:
    portfolio = simulate_portfolio(
        data=data,
        signal=signal,
        slippage=engine.slippage,
        commission=engine.commission,
        rules=engine.rules,
        initial_cash=engine.initial_cash,
        contract_multiplier=engine.contract_multiplier,
        record_trades=False,
    )
    perf = calculate_performance_metrics(
        returns=portfolio.returns,
        equity_curve=portfolio.equity_curve,
        risk_free_rate=engine.risk_free_rate,
        periods_per_year=engine.periods_per_year,
        market_returns=market_returns_from_close(data.close),
    )
    risk = calculate_risk_metrics(
        returns=portfolio.returns,
        equity_curve=portfolio.equity_curve,
    )
    trade = calculate_trade_metrics(portfolio.trades)
    score = score_from_report(
        _ReportProxy(performance=perf, risk=risk, trade_metrics=trade),
        metric=metric,
    )
    return TrialResult(params=params, score=score, report=None)


def evaluate_batch(
    engine: BacktestEngine,
    data: DataBundle,
    strategy_factory: Callable[[], BaseStrategy],
    params_list: list[dict[str, Any]],
    metric: str,
) -> list[TrialResult]:
    if not params_list:
        return []

    strategy = strategy_factory()
    signal_tensor = build_signal_tensor(strategy, data, params_list)
    if signal_tensor.ndim != 3:
        raise ValueError("signal_tensor 维度必须为 3")

    equity_2d, returns_2d = simulate_portfolio_batch(
        data=data,
        signal=signal_tensor,
        slippage=engine.slippage,
        commission=engine.commission,
        rules=engine.rules,
        initial_cash=engine.initial_cash,
        contract_multiplier=engine.contract_multiplier,
    )
    scores = batch_score(
        equity_2d=equity_2d,
        returns_2d=returns_2d,
        metric=metric,
        risk_free_rate=engine.risk_free_rate,
        periods_per_year=engine.periods_per_year,
        market_returns=market_returns_from_close(data.close),
    )
    return [
        TrialResult(params=params_list[idx], score=float(scores[idx]), report=None) for idx in range(len(params_list))
    ]


def evaluate_trials_parallel(
    engine: BacktestEngine,
    data: DataBundle,
    strategy_factory: Callable[[], BaseStrategy],
    params_list: list[dict[str, Any]],
    metric: str,
    max_workers: int | None = None,
) -> list[TrialResult]:
    if not params_list:
        return []

    if max_workers is None or max_workers <= 1:
        results: list[TrialResult] = []
        for params in params_list:
            strategy = strategy_factory()
            report = engine.run(data=data, strategy=strategy, params=params, record_trades=False)
            score = score_from_report(report, metric=metric)
            results.append(TrialResult(params=params, score=score, report=report))
        return results

    try:
        pickle.dumps(strategy_factory)
        strategy_type = type(strategy_factory())
    except Exception:
        logger.warning("strategy_factory 不可序列化，已退回为串行执行 evaluate_trials_parallel")
        return evaluate_trials_parallel(
            engine=engine,
            data=data,
            strategy_factory=strategy_factory,
            params_list=params_list,
            metric=metric,
            max_workers=1,
        )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_trial_worker,
                engine=engine,
                data=data,
                strategy_type=strategy_type,
                params=params,
                metric=metric,
            ): idx
            for idx, params in enumerate(params_list)
        }
        ordered: list[TrialResult | None] = [None] * len(params_list)
        for future in as_completed(futures):
            idx = futures[future]
            ordered[idx] = future.result()
        return [cast(TrialResult, result) for result in ordered]


def build_signal_tensor(
    strategy: BaseStrategy,
    data: DataBundle,
    params_list: list[dict[str, Any]],
):
    signals = [strategy.generate_signals(data, params) for params in params_list]
    if not signals:
        return np.empty((0, 0, 0))
    first = signals[0]
    if cp is not None and isinstance(first, cp.ndarray):
        return cp.stack(signals, axis=2)
    return np.stack(signals, axis=2)


def evaluate_signal_tensor(
    engine: BacktestEngine,
    data: DataBundle,
    params_list: list[dict[str, Any]],
    signal_tensor,
    metric: str,
) -> list[TrialResult]:
    if signal_tensor.ndim != 3:
        raise ValueError("signal_tensor 维度必须为 3")
    n_combo = signal_tensor.shape[2]
    max_workers = min(os.cpu_count() or 1, max(1, n_combo))
    if cp is not None and isinstance(signal_tensor, cp.ndarray):
        max_workers = 1
    if max_workers <= 1:
        return [
            _evaluate_signal_combo_worker(
                engine=engine,
                data=data,
                signal=signal_tensor[:, :, idx],
                params=params_list[idx],
                metric=metric,
            )
            for idx in range(n_combo)
        ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_signal_combo_worker,
                engine=engine,
                data=data,
                signal=signal_tensor[:, :, idx],
                params=params_list[idx],
                metric=metric,
            ): idx
            for idx in range(n_combo)
        }
        ordered: list[TrialResult | None] = [None] * n_combo
        for future in as_completed(futures):
            idx = futures[future]
            ordered[idx] = future.result()
        return [cast(TrialResult, result) for result in ordered if result is not None]
