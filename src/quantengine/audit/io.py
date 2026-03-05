from __future__ import annotations

import csv
import io
import json
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from quantengine.data.loader import DataBundle

from .bundle import AuditBundle, capture_environment, hash_data_bundle

REQUIRED_BUNDLE_FILES = {
    "sha256.json",
    "config.json",
    "env.json",
    "metadata.json",
    "trades.csv",
    "risk_events.csv",
    "equity_curve.csv",
    "returns.csv",
    "performance.json",
    "risk.json",
    "trade_metrics.json",
}


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _series_to_csv(values: list[float], column: str) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["index", column])
    for idx, value in enumerate(values):
        writer.writerow([idx, format(float(value), ".17g")])
    return buffer.getvalue().encode("utf-8")


def _trades_to_csv(rows: list[dict[str, Any]]) -> bytes:
    headers = ["timestamp", "symbol", "side", "quantity", "price", "cost"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "timestamp": row.get("timestamp", ""),
                "symbol": row.get("symbol", ""),
                "side": row.get("side", ""),
                "quantity": format(float(row.get("quantity", 0.0)), ".17g"),
                "price": format(float(row.get("price", 0.0)), ".17g"),
                "cost": format(float(row.get("cost", 0.0)), ".17g"),
            }
        )
    return buffer.getvalue().encode("utf-8")


def _risk_events_to_csv(rows: list[dict[str, Any]]) -> bytes:
    headers = ["bar", "type", "detail"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "bar": int(row.get("bar", 0)),
                "type": row.get("type", ""),
                "detail": row.get("detail", ""),
            }
        )
    return buffer.getvalue().encode("utf-8")


def save_audit_bundle(bundle: AuditBundle | None, output_path: str | Path) -> Path:
    if bundle is None:
        raise ValueError("bundle 不能为空")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_bytes: dict[str, bytes] = {
        "config.json": _json_bytes(bundle.config),
        "env.json": _json_bytes(bundle.env),
        "metadata.json": _json_bytes(
            {"created_at": bundle.created_at, "seed": bundle.seed, "data_hash": bundle.data_hash}
        ),
        "trades.csv": _trades_to_csv(bundle.trade_log),
        "risk_events.csv": _risk_events_to_csv(bundle.risk_events),
        "equity_curve.csv": _series_to_csv(bundle.equity_curve, column="equity"),
        "returns.csv": _series_to_csv(bundle.returns, column="return"),
        "performance.json": _json_bytes(bundle.performance),
        "risk.json": _json_bytes(bundle.risk),
        "trade_metrics.json": _json_bytes(bundle.trade_metrics),
    }
    manifest = {
        "data_hash": bundle.data_hash,
        "files": {name: _sha256_hex(payload) for name, payload in file_bytes.items()},
    }
    file_bytes["sha256.json"] = _json_bytes(manifest)

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in file_bytes.items():
            zf.writestr(name, payload)
    return path


def _parse_series_csv(payload: bytes, value_column: str) -> list[float]:
    text = payload.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    values: list[float] = []
    for row in reader:
        token = row.get(value_column, "")
        if token == "":
            continue
        values.append(float(token))
    return values


def _parse_trades_csv(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for row in reader:
        rows.append(
            {
                "timestamp": row.get("timestamp", ""),
                "symbol": row.get("symbol", ""),
                "side": row.get("side", ""),
                "quantity": float(row.get("quantity", "0") or 0.0),
                "price": float(row.get("price", "0") or 0.0),
                "cost": float(row.get("cost", "0") or 0.0),
            }
        )
    return rows


def _parse_risk_events_csv(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for row in reader:
        rows.append(
            {
                "bar": int(row.get("bar", "0") or 0),
                "type": row.get("type", ""),
                "detail": row.get("detail", ""),
            }
        )
    return rows


def load_audit_bundle(bundle_path: str | Path) -> AuditBundle:
    path = Path(bundle_path)
    with zipfile.ZipFile(path, mode="r") as zf:
        names = set(zf.namelist())
        missing = sorted(REQUIRED_BUNDLE_FILES - names)
        if missing:
            raise ValueError(f"审计包缺少必要文件: {', '.join(missing)}")

        config = json.loads(zf.read("config.json").decode("utf-8"))
        env = json.loads(zf.read("env.json").decode("utf-8"))
        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        performance = json.loads(zf.read("performance.json").decode("utf-8"))
        risk = json.loads(zf.read("risk.json").decode("utf-8"))
        trade_metrics = json.loads(zf.read("trade_metrics.json").decode("utf-8"))
        trades = _parse_trades_csv(zf.read("trades.csv"))
        risk_events = _parse_risk_events_csv(zf.read("risk_events.csv"))
        equity_curve = _parse_series_csv(zf.read("equity_curve.csv"), value_column="equity")
        returns = _parse_series_csv(zf.read("returns.csv"), value_column="return")

    return AuditBundle(
        created_at=str(metadata.get("created_at", "")),
        data_hash=str(metadata.get("data_hash", "")),
        config=config,
        env=env,
        seed=metadata.get("seed"),
        trade_log=trades,
        risk_events=risk_events,
        equity_curve=equity_curve,
        returns=returns,
        performance={str(k): float(v) for k, v in dict(performance).items()},
        risk={str(k): float(v) for k, v in dict(risk).items()},
        trade_metrics={str(k): v for k, v in dict(trade_metrics).items()},
    )


def _environment_mismatches(bundle_env: dict[str, Any]) -> dict[str, dict[str, str]]:
    current = capture_environment()
    mismatches: dict[str, dict[str, str]] = {}
    for key in ("python_version", "numpy_version", "quantengine_version"):
        expected = str(bundle_env.get(key, ""))
        actual = str(current.get(key, ""))
        if expected and actual and expected != actual:
            mismatches[key] = {"expected": expected, "actual": actual}
    return mismatches


def verify_audit_bundle(
    bundle_path: str | Path,
    *,
    data: DataBundle | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    path = Path(bundle_path)
    with zipfile.ZipFile(path, mode="r") as zf:
        names = set(zf.namelist())
        missing = sorted(REQUIRED_BUNDLE_FILES - names)
        if missing:
            if strict:
                raise ValueError(f"审计包缺少必要文件: {', '.join(missing)}")
            return {
                "ok": False,
                "missing_files": missing,
                "integrity_ok": False,
                "data_hash_match": None,
                "env_match": False,
                "env_mismatches": {},
            }

        manifest = json.loads(zf.read("sha256.json").decode("utf-8"))
        expected_files: dict[str, str] = dict(manifest.get("files", {}))
        integrity_errors: list[str] = []
        for file_name, expected_digest in expected_files.items():
            if file_name not in names:
                integrity_errors.append(f"missing:{file_name}")
                continue
            actual_digest = _sha256_hex(zf.read(file_name))
            if actual_digest != str(expected_digest):
                integrity_errors.append(f"mismatch:{file_name}")
        manifest_data_hash = str(manifest.get("data_hash", ""))

    bundle = load_audit_bundle(path)
    env_mismatches = _environment_mismatches(bundle.env)
    env_match = len(env_mismatches) == 0
    data_hash_match: bool | None = None
    if data is not None:
        current_data_hash = hash_data_bundle(data)
        data_hash_match = current_data_hash == manifest_data_hash == bundle.data_hash

    ok = (len(integrity_errors) == 0) and env_match and (data_hash_match in (None, True))
    result = {
        "ok": ok,
        "missing_files": [],
        "integrity_ok": len(integrity_errors) == 0,
        "integrity_errors": integrity_errors,
        "data_hash_match": data_hash_match,
        "env_match": env_match,
        "env_mismatches": env_mismatches,
    }
    if strict and not ok:
        if data_hash_match is False:
            raise ValueError("数据哈希不一致")
        raise ValueError(f"审计包校验失败: {result}")
    return result
