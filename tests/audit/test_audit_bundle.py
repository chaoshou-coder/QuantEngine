from __future__ import annotations

import zipfile

import numpy as np
import pytest

from quantengine.audit import replay_from_bundle, save_audit_bundle, verify_audit_bundle
from quantengine.data.gpu_backend import BackendInfo
from quantengine.data.loader import DataBundle
from quantengine.engine.backtest import BacktestEngine
from quantengine.engine.commission import FixedCommission
from quantengine.engine.rules import TradingRules
from quantengine.engine.slippage import FixedSlippage
from quantengine.strategy.base import BaseStrategy
from quantengine.strategy.registry import register_strategy


@register_strategy("audit_static_test")
class AuditStaticStrategy(BaseStrategy):
    name = "audit_static_test"

    def parameters(self):
        return {}

    def generate_signals(self, data: DataBundle, params: dict):
        _ = params
        return np.array([[0.0], [1.0], [1.0], [0.0], [0.0]], dtype=float)


def _build_data() -> DataBundle:
    timestamps = np.array(
        [
            "2026-03-02T09:30:00",
            "2026-03-02T09:31:00",
            "2026-03-02T09:32:00",
            "2026-03-02T09:33:00",
            "2026-03-02T09:34:00",
        ],
        dtype="datetime64[ns]",
    )
    close = np.array([[100.0], [101.0], [103.0], [102.0], [104.0]], dtype=float)
    return DataBundle(
        symbols=["AAA"],
        timestamps=timestamps,
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close.copy(),
        volume=np.array([[1000.0], [900.0], [1100.0], [1200.0], [1000.0]], dtype=float),
        backend=BackendInfo(
            requested="auto",
            active="cpu",
            reason="test",
            gpu_available=False,
            cudf_available=False,
            cupy_available=False,
        ),
    )


def _build_engine() -> BacktestEngine:
    return BacktestEngine(
        slippage=FixedSlippage(points=0.0),
        commission=FixedCommission(value=0.0),
        rules=TradingRules(margin_ratio=0.0),
        initial_cash=10_000.0,
        contract_multiplier=1.0,
        risk_free_rate=0.0,
        periods_per_year=252,
    )


def test_engine_run_attaches_audit_bundle():
    engine = _build_engine()
    data = _build_data()
    strategy = AuditStaticStrategy()

    report = engine.run(data=data, strategy=strategy, params={"seed": 123}, record_trades=True)

    assert report.audit_bundle is not None
    assert report.audit_bundle.data_hash
    assert report.audit_bundle.config["strategy"]["name"] == "audit_static_test"
    assert report.audit_bundle.config["strategy"]["params"] == {"seed": 123}
    assert report.audit_bundle.trade_log == report.portfolio.trades
    assert report.audit_bundle.risk_events == (report.portfolio.risk_events or [])


def test_save_verify_audit_bundle_roundtrip(tmp_path):
    engine = _build_engine()
    data = _build_data()
    strategy = AuditStaticStrategy()
    report = engine.run(data=data, strategy=strategy, params={"seed": 7}, record_trades=True)

    output_zip = tmp_path / "audit_bundle.zip"
    save_audit_bundle(report.audit_bundle, output_zip)
    verification = verify_audit_bundle(output_zip, data=data)

    assert verification["ok"] is True
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = set(zf.namelist())
    assert {"sha256.json", "config.json", "env.json", "trades.csv", "risk_events.csv"}.issubset(names)


def test_verify_audit_bundle_raises_on_data_hash_mismatch(tmp_path):
    engine = _build_engine()
    data = _build_data()
    strategy = AuditStaticStrategy()
    report = engine.run(data=data, strategy=strategy, params={}, record_trades=True)
    output_zip = tmp_path / "audit_bundle.zip"
    save_audit_bundle(report.audit_bundle, output_zip)

    changed = DataBundle(
        symbols=data.symbols,
        timestamps=data.timestamps,
        open=data.open.copy(),
        high=data.high.copy(),
        low=data.low.copy(),
        close=(data.close + 1.0).copy(),
        volume=data.volume.copy(),
        backend=data.backend,
    )
    with pytest.raises(ValueError, match="数据哈希不一致"):
        verify_audit_bundle(output_zip, data=changed)


def test_replay_from_bundle_is_bit_identical(tmp_path):
    engine = _build_engine()
    data = _build_data()
    strategy = AuditStaticStrategy()
    report = engine.run(data=data, strategy=strategy, params={"seed": 99}, record_trades=True)
    output_zip = tmp_path / "audit_bundle.zip"
    save_audit_bundle(report.audit_bundle, output_zip)

    replay_report = replay_from_bundle(output_zip, data=data)

    np.testing.assert_array_equal(report.portfolio.equity_curve, replay_report.portfolio.equity_curve)
    np.testing.assert_array_equal(report.portfolio.returns, replay_report.portfolio.returns)
