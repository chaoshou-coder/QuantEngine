from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantengine.data.gpu_backend import get_backend_info, to_numpy
from quantengine.data.loader import DataBundle
from quantengine.engine.backtest import BacktestEngine
from quantengine.engine.commission import PercentCommission
from quantengine.engine.portfolio import simulate_portfolio, simulate_portfolio_batch
from quantengine.engine.rules import TradingRules
from quantengine.engine.slippage import PercentSlippage
from quantengine.indicators import technical
from quantengine.strategy.examples.psar_trade_assist_v3 import PsarTradeAssistV3Strategy

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = ROOT / "test_data"


def _gpu_ready() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _load_real_slice(rows: int = 5000) -> pd.DataFrame:
    candidates = [
        TEST_DATA_DIR / "XAUUSD_1m_20190101_20200101.csv",
        TEST_DATA_DIR / "XAUUSD_M1.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue

        if path.name == "XAUUSD_M1.csv":
            frame = pd.read_csv(
                path,
                header=None,
                names=["date", "time", "open", "high", "low", "close", "volume"],
                usecols=[0, 1, 2, 3, 4, 5, 6],
            )
            frame["datetime"] = pd.to_datetime(
                frame["date"].astype(str) + " " + frame["time"].astype(str),
                format="%Y.%m.%d %H:%M",
                errors="coerce",
                utc=True,
            )
        else:
            frame = pd.read_csv(path)
            if "datetime" not in frame.columns:
                continue
            frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
        frame = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)]
        if len(frame) >= rows:
            return frame.iloc[:rows].copy()

    raise RuntimeError("未找到可用真实数据切片，请检查 test_data 目录。")


def _bundle_from_frame(frame: pd.DataFrame, backend: str = "cpu") -> DataBundle:
    symbols = ["XAUUSD"]
    ts = frame["datetime"].to_numpy()
    open_arr = frame["open"].to_numpy(dtype=float).reshape(-1, 1)
    high_arr = frame["high"].to_numpy(dtype=float).reshape(-1, 1)
    low_arr = frame["low"].to_numpy(dtype=float).reshape(-1, 1)
    close_arr = frame["close"].to_numpy(dtype=float).reshape(-1, 1)
    volume_arr = frame["volume"].to_numpy(dtype=float).reshape(-1, 1)

    backend_info = get_backend_info(requested=backend, use_gpu=True)
    if backend_info.active == "gpu" and cp is not None:
        open_arr = cp.asarray(open_arr)
        high_arr = cp.asarray(high_arr)
        low_arr = cp.asarray(low_arr)
        close_arr = cp.asarray(close_arr)
        volume_arr = cp.asarray(volume_arr)

    return DataBundle(
        symbols=symbols,
        timestamps=ts,
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        volume=volume_arr,
        backend=backend_info,
    )


def _engine() -> BacktestEngine:
    return BacktestEngine(
        slippage=PercentSlippage(rate=0.0),
        commission=PercentCommission(rate=0.0),
        risk_free_rate=0.0,
        periods_per_year=358800,
    )


def test_psar_numba_matches_python() -> None:
    if technical._parabolic_sar_1d_numba is None:  # type: ignore[attr-defined]
        pytest.skip("numba 不可用，跳过")
    rng = np.random.default_rng(7)
    close = 1900 + np.cumsum(rng.normal(0, 0.8, 512))
    high = close + rng.uniform(0.1, 0.7, close.shape[0])
    low = close - rng.uniform(0.1, 0.7, close.shape[0])

    py = technical._parabolic_sar_1d_py(high, low, close, 0.02, 0.2)  # type: ignore[attr-defined]
    nj = technical._parabolic_sar_1d_numba(high, low, close, 0.02, 0.2)  # type: ignore[attr-defined]
    assert np.allclose(py, nj, atol=1e-10)


def test_adx_numba_matches_python() -> None:
    if technical._adx_1d_numba is None:  # type: ignore[attr-defined]
        pytest.skip("numba 不可用，跳过")
    rng = np.random.default_rng(11)
    close = 1900 + np.cumsum(rng.normal(0, 0.8, 512))
    high = close + rng.uniform(0.1, 0.7, close.shape[0])
    low = close - rng.uniform(0.1, 0.7, close.shape[0])

    py_adx, py_pdi, py_mdi = technical._adx_1d_py(high, low, close, 14)  # type: ignore[attr-defined]
    nj_adx, nj_pdi, nj_mdi = technical._adx_1d_numba(high, low, close, 14)  # type: ignore[attr-defined]
    assert np.allclose(py_adx, nj_adx, atol=1e-10)
    assert np.allclose(py_pdi, nj_pdi, atol=1e-10)
    assert np.allclose(py_mdi, nj_mdi, atol=1e-10)


def test_ema_matches_pandas_reference() -> None:
    rng = np.random.default_rng(13)
    close = 1900 + np.cumsum(rng.normal(0, 0.8, 256))
    got = np.asarray(technical.ema(close, span=14), dtype=float)
    exp = pd.Series(close).ewm(span=14, adjust=False).mean().to_numpy(dtype=float)
    assert np.allclose(got, exp, atol=1e-10)


def test_atr_matches_reference_ema_tr() -> None:
    rng = np.random.default_rng(17)
    close = 1900 + np.cumsum(rng.normal(0, 0.8, 256))
    high = close + rng.uniform(0.1, 0.7, close.shape[0])
    low = close - rng.uniform(0.1, 0.7, close.shape[0])

    got = np.asarray(technical.atr(high, low, close, window=14), dtype=float)
    prev_close = np.concatenate([close[:1], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    exp = pd.Series(tr).ewm(span=14, adjust=False).mean().to_numpy(dtype=float)
    assert np.allclose(got, exp, atol=1e-10)


def test_signal_vectorized_matches_reference_loop() -> None:
    frame = _load_real_slice(rows=5000)
    bundle = _bundle_from_frame(frame, backend="cpu")
    strategy = PsarTradeAssistV3Strategy()
    params: dict[str, int | float] = {}

    vec = np.asarray(strategy.generate_signals(bundle, params), dtype=float)
    ref = np.asarray(strategy.generate_signals_reference(bundle, params), dtype=float)
    assert np.array_equal(vec, ref)


def test_gpu_cpu_signal_and_equity_consistency() -> None:
    if not _gpu_ready():
        pytest.skip("无可用 CUDA GPU，跳过")

    frame = _load_real_slice(rows=4000)
    cpu_bundle = _bundle_from_frame(frame, backend="cpu")
    gpu_bundle = _bundle_from_frame(frame, backend="gpu")
    if gpu_bundle.backend.active != "gpu":
        pytest.skip("GPU backend 未激活，跳过")

    strategy = PsarTradeAssistV3Strategy()
    params: dict[str, int | float] = {}
    cpu_sig = np.asarray(strategy.generate_signals(cpu_bundle, params), dtype=float)
    gpu_sig = np.asarray(to_numpy(strategy.generate_signals(gpu_bundle, params)), dtype=float)
    assert np.allclose(cpu_sig, gpu_sig, atol=1e-10)

    engine = _engine()
    cpu_report = engine.run(cpu_bundle, strategy, params, record_trades=True)
    gpu_report = engine.run(gpu_bundle, strategy, params, record_trades=True)
    eq_cpu = np.asarray(cpu_report.portfolio.equity_curve, dtype=float)
    eq_gpu = np.asarray(gpu_report.portfolio.equity_curve, dtype=float)
    denom = np.maximum(np.abs(eq_cpu), 1.0)
    rel_err = np.max(np.abs(eq_cpu - eq_gpu) / denom)
    assert rel_err < 1e-6


def test_batch_vectorized_matches_serial_no_margin() -> None:
    frame = _load_real_slice(rows=4000)
    bundle = _bundle_from_frame(frame, backend="cpu")
    strategy = PsarTradeAssistV3Strategy()

    params_a: dict[str, int | float] = {}
    params_b: dict[str, int | float] = {
        "enable_adx": 1,
        "enable_rsi": 0,
        "enable_ma": 1,
        "enable_body": 1,
        "enable_adaptive_step": 1,
        "adx_threshold": 22.5,
        "ma_fast": 15,
        "ma_slow": 50,
        "body_atr_mult": 2.5,
    }

    sig_a = np.asarray(strategy.generate_signals(bundle, params_a), dtype=float)
    sig_b = np.asarray(strategy.generate_signals(bundle, params_b), dtype=float)
    signal_tensor = np.stack([sig_a, sig_b], axis=2)

    # NOTE: numba fastpath in simulate_portfolio currently does not model non-zero slippage cost,
    # so keep slippage=0 here to validate vectorized path against the same execution semantics.
    slippage = PercentSlippage(rate=0.0)
    commission = PercentCommission(rate=0.0002)
    rules = TradingRules(margin_ratio=0.0)
    equity_2d, returns_2d = simulate_portfolio_batch(
        data=bundle,
        signal=signal_tensor,
        slippage=slippage,
        commission=commission,
        rules=rules,
        initial_cash=1_000_000.0,
        contract_multiplier=1.0,
    )
    equity_2d = np.asarray(equity_2d, dtype=float)
    returns_2d = np.asarray(returns_2d, dtype=float)

    for idx, sig in enumerate([sig_a, sig_b]):
        serial = simulate_portfolio(
            data=bundle,
            signal=sig,
            slippage=slippage,
            commission=commission,
            rules=rules,
            initial_cash=1_000_000.0,
            contract_multiplier=1.0,
            record_trades=False,
        )
        eq = np.asarray(serial.equity_curve, dtype=float)
        rt = np.asarray(serial.returns, dtype=float)
        assert np.allclose(equity_2d[:, idx], eq, atol=1e-6, rtol=1e-6)
        assert np.allclose(returns_2d[:, idx], rt, atol=1e-6, rtol=1e-6)


def test_edge_cases_do_not_crash() -> None:
    strategy = PsarTradeAssistV3Strategy()

    backend = get_backend_info("cpu", use_gpu=False)
    empty = DataBundle(
        symbols=["XAUUSD"],
        timestamps=np.array([], dtype="datetime64[ns]"),
        open=np.zeros((0, 1), dtype=float),
        high=np.zeros((0, 1), dtype=float),
        low=np.zeros((0, 1), dtype=float),
        close=np.zeros((0, 1), dtype=float),
        volume=np.zeros((0, 1), dtype=float),
        backend=backend,
    )
    out_empty = np.asarray(strategy.generate_signals(empty, {}), dtype=float)
    assert out_empty.shape == (0, 1)

    one = DataBundle(
        symbols=["XAUUSD"],
        timestamps=np.array([np.datetime64("2020-01-01T00:00:00")]),
        open=np.array([[1900.0]]),
        high=np.array([[1901.0]]),
        low=np.array([[1899.0]]),
        close=np.array([[1900.5]]),
        volume=np.array([[1.0]]),
        backend=backend,
    )
    out_one = np.asarray(strategy.generate_signals(one, {}), dtype=float)
    assert out_one.shape == (1, 1)

    frame = _load_real_slice(rows=2000)
    frame.loc[:10, ["open", "high", "low", "close"]] = np.nan
    frame = frame.ffill().bfill()
    bundle = _bundle_from_frame(frame, backend="cpu")

    off_params = {
        "enable_adx": 0,
        "enable_rsi": 0,
        "enable_ma": 0,
        "enable_body": 0,
        "enable_adaptive_step": 0,
    }
    on_params = {
        "enable_adx": 1,
        "enable_rsi": 1,
        "enable_ma": 1,
        "enable_body": 1,
        "enable_adaptive_step": 1,
    }

    sig_off = np.asarray(strategy.generate_signals(bundle, off_params), dtype=float)
    sig_on = np.asarray(strategy.generate_signals(bundle, on_params), dtype=float)
    assert sig_off.shape == sig_on.shape
    assert np.any(sig_off != 0.0)

    fixed_step = {
        "enable_adaptive_step": 1,
        "psar_step_low": 0.02,
        "psar_step_mid": 0.02,
        "psar_step_high": 0.02,
    }
    sig_fixed = np.asarray(strategy.generate_signals(bundle, fixed_step), dtype=float)
    assert sig_fixed.shape == sig_off.shape


def test_backtest_sanity_with_real_slice() -> None:
    frame = _load_real_slice(rows=2000)
    bundle = _bundle_from_frame(frame, backend="cpu")
    strategy = PsarTradeAssistV3Strategy()
    params = {
        "enable_adx": 0,
        "enable_rsi": 0,
        "enable_ma": 0,
        "enable_body": 0,
        "enable_adaptive_step": 1,
    }
    report = _engine().run(bundle, strategy, params, record_trades=True)

    equity = np.asarray(report.portfolio.equity_curve, dtype=float)
    assert equity.size == len(frame)
    assert np.any(np.diff(equity) != 0.0)

    trade_count = float(report.trade_metrics.get("trade_count", 0.0))
    assert trade_count > 0.0

    trade_win_rate = float(report.trade_metrics.get("win_rate", 0.0))
    assert 0.0 <= trade_win_rate <= 1.0

    beta = float(report.performance.get("beta", 0.0))
    assert -2.0 <= beta <= 2.0

    profit_factor = float(report.trade_metrics.get("profit_factor", 0.0))
    assert profit_factor >= 0.0
