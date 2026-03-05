"""Tests for audit/io.py — save, load, verify edge cases."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from quantengine.audit.bundle import AuditBundle, capture_environment, hash_data_bundle
from quantengine.audit.io import load_audit_bundle, save_audit_bundle, verify_audit_bundle
from quantengine.data.gpu_backend import BackendInfo
from quantengine.data.loader import DataBundle


def _make_bundle(seed: int = 42) -> AuditBundle:
    return AuditBundle(
        created_at="2026-03-05T00:00:00+00:00",
        data_hash="abc123",
        config={"strategy": {"name": "test", "params": {"x": 1}}},
        env=capture_environment(),
        seed=seed,
        trade_log=[],
        risk_events=[],
        equity_curve=[1000.0, 1001.0, 999.0],
        returns=[0.0, 0.001, -0.002],
        performance={"sharpe": 0.5, "win_rate": 0.55},
        risk={"var_95": -0.01},
        trade_metrics={"trade_count": 0.0},
    )


def _make_data() -> DataBundle:
    n = 3
    ts = np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[ns]")
    c = np.array([[100.0], [101.0], [102.0]])
    return DataBundle(
        symbols=["X"],
        timestamps=ts,
        open=c.copy(), high=c + 1, low=c - 1, close=c.copy(),
        volume=np.ones((n, 1)) * 1000,
        backend=BackendInfo("auto", "cpu", "test", False, False, False),
    )


def test_save_and_load_roundtrip(tmp_path):
    bundle = _make_bundle()
    path = tmp_path / "test.zip"
    save_audit_bundle(bundle, path)
    loaded = load_audit_bundle(path)
    assert loaded.data_hash == bundle.data_hash
    assert loaded.config == bundle.config
    assert loaded.equity_curve == bundle.equity_curve
    assert loaded.returns == bundle.returns
    assert loaded.seed == bundle.seed


def test_save_none_raises():
    with pytest.raises(ValueError, match="不能为空"):
        save_audit_bundle(None, "/tmp/x.zip")


def test_load_missing_file_raises(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", "{}")
    with pytest.raises(ValueError, match="缺少必要文件"):
        load_audit_bundle(path)


def test_verify_strict_integrity_error(tmp_path):
    bundle = _make_bundle()
    path = tmp_path / "tampered.zip"
    save_audit_bundle(bundle, path)

    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("config.json", '{"tampered": true}')

    with pytest.raises(ValueError):
        verify_audit_bundle(path, strict=True)


def test_verify_non_strict_returns_result(tmp_path):
    bundle = _make_bundle()
    path = tmp_path / "tampered2.zip"
    save_audit_bundle(bundle, path)

    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("config.json", '{"tampered": true}')

    result = verify_audit_bundle(path, strict=False)
    assert result["ok"] is False
    assert result["integrity_ok"] is False


def test_verify_data_hash_match(tmp_path):
    data = _make_data()
    bundle = _make_bundle()
    bundle.data_hash = hash_data_bundle(data)
    path = tmp_path / "ok.zip"
    save_audit_bundle(bundle, path)
    result = verify_audit_bundle(path, data=data, strict=False)
    assert result["data_hash_match"] is True


def test_save_with_trades_and_risk_events(tmp_path):
    bundle = _make_bundle()
    bundle.trade_log = [
        {"timestamp": "2026-01-01", "symbol": "X", "side": "BUY",
         "quantity": 1.0, "price": 100.0, "cost": 0.5},
    ]
    bundle.risk_events = [
        {"bar": 5, "type": "daily_loss_breach", "detail": "loss=-0.03"},
    ]
    path = tmp_path / "with_trades.zip"
    save_audit_bundle(bundle, path)
    loaded = load_audit_bundle(path)
    assert len(loaded.trade_log) == 1
    assert loaded.trade_log[0]["symbol"] == "X"
    assert len(loaded.risk_events) == 1
    assert loaded.risk_events[0]["type"] == "daily_loss_breach"


def test_zip_contains_all_required_files(tmp_path):
    bundle = _make_bundle()
    path = tmp_path / "complete.zip"
    save_audit_bundle(bundle, path)
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
    required = {
        "sha256.json", "config.json", "env.json", "metadata.json",
        "trades.csv", "risk_events.csv", "equity_curve.csv",
        "returns.csv", "performance.json", "risk.json", "trade_metrics.json",
    }
    assert required.issubset(names)
