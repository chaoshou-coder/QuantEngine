from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from quantengine.data.gpu_backend import to_numpy
from quantengine.data.loader import DataBundle
from quantengine.engine.commission import FixedCommission, PercentCommission, TieredCommission
from quantengine.engine.portfolio import PortfolioResult
from quantengine.engine.rules import TradingRules
from quantengine.engine.slippage import FixedSlippage, PercentSlippage, VolumeSlippage


@dataclass
class AuditBundle:
    created_at: str
    data_hash: str
    config: dict[str, Any]
    env: dict[str, Any]
    seed: int | None
    trade_log: list[dict[str, Any]]
    risk_events: list[dict[str, Any]]
    equity_curve: list[float]
    returns: list[float]
    performance: dict[str, float]
    risk: dict[str, float]
    trade_metrics: dict[str, float | str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AuditBundle:
        return cls(
            created_at=str(payload["created_at"]),
            data_hash=str(payload["data_hash"]),
            config=dict(payload.get("config", {})),
            env=dict(payload.get("env", {})),
            seed=_to_seed(payload.get("seed")),
            trade_log=list(payload.get("trade_log", [])),
            risk_events=list(payload.get("risk_events", [])),
            equity_curve=[float(item) for item in payload.get("equity_curve", [])],
            returns=[float(item) for item in payload.get("returns", [])],
            performance={str(k): float(v) for k, v in dict(payload.get("performance", {})).items()},
            risk={str(k): float(v) for k, v in dict(payload.get("risk", {})).items()},
            trade_metrics={str(k): _to_builtin(v) for k, v in dict(payload.get("trade_metrics", {})).items()},
        )


def _to_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if token and token.lstrip("-").isdigit():
            return int(token)
    return None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hash_array(hasher, values: Any) -> None:
    array = np.asarray(to_numpy(values))
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(str(contiguous.shape).encode("utf-8"))
    hasher.update(contiguous.tobytes())


def hash_data_bundle(data: DataBundle) -> str:
    hasher = sha256()
    for symbol in data.symbols:
        hasher.update(symbol.encode("utf-8"))
        hasher.update(b"\x00")
    _hash_array(hasher, np.asarray(data.timestamps, dtype="datetime64[ns]").astype(np.int64))
    _hash_array(hasher, data.open)
    _hash_array(hasher, data.high)
    _hash_array(hasher, data.low)
    _hash_array(hasher, data.close)
    _hash_array(hasher, data.volume)
    return hasher.hexdigest()


def capture_environment() -> dict[str, str]:
    try:
        quantengine_version = version("quantengine")
    except PackageNotFoundError:
        quantengine_version = "0.1.0"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "quantengine_version": quantengine_version,
    }


def _serialize_slippage(slippage: Any) -> dict[str, Any]:
    if isinstance(slippage, FixedSlippage):
        return {"model": "fixed", "value": float(slippage.points), "impact": 0.0}
    if isinstance(slippage, PercentSlippage):
        return {"model": "percent", "value": float(slippage.rate), "impact": 0.0}
    if isinstance(slippage, VolumeSlippage):
        return {"model": "volume", "value": float(slippage.max_ratio), "impact": float(slippage.impact)}
    raise TypeError(f"不支持的滑点模型类型: {type(slippage).__name__}")


def _serialize_commission(commission: Any) -> dict[str, Any]:
    if isinstance(commission, FixedCommission):
        return {"model": "fixed", "value": float(commission.value), "tiers": []}
    if isinstance(commission, PercentCommission):
        return {"model": "percent", "value": float(commission.rate), "tiers": []}
    if isinstance(commission, TieredCommission):
        return {
            "model": "tiered",
            "value": float(commission.fallback_rate),
            "tiers": [[float(item[0]), float(item[1])] for item in commission.tiers],
        }
    raise TypeError(f"不支持的手续费模型类型: {type(commission).__name__}")


def _serialize_rules(rules: TradingRules | None) -> dict[str, Any]:
    if rules is None:
        return {}
    return _to_builtin(asdict(rules))


def _extract_seed(params: dict[str, Any]) -> int | None:
    if "seed" in params:
        seed = _to_seed(params.get("seed"))
        if seed is not None:
            return seed
    if "random_seed" in params:
        seed = _to_seed(params.get("random_seed"))
        if seed is not None:
            return seed
    return None


def build_audit_bundle(
    *,
    data: DataBundle,
    strategy_name: str,
    params: dict[str, Any],
    slippage: Any,
    commission: Any,
    rules: TradingRules | None,
    initial_cash: float,
    contract_multiplier: float,
    risk_free_rate: float,
    periods_per_year: int,
    record_trades: bool,
    portfolio: PortfolioResult,
    performance: dict[str, float],
    risk: dict[str, float],
    trade_metrics: dict[str, float | str],
) -> AuditBundle:
    normalized_params = _to_builtin(params)
    created_at = datetime.now(UTC).isoformat()
    return AuditBundle(
        created_at=created_at,
        data_hash=hash_data_bundle(data),
        config={
            "strategy": {"name": str(strategy_name), "params": normalized_params},
            "engine": {
                "slippage": _serialize_slippage(slippage),
                "commission": _serialize_commission(commission),
                "rules": _serialize_rules(rules),
                "initial_cash": float(initial_cash),
                "contract_multiplier": float(contract_multiplier),
                "risk_free_rate": float(risk_free_rate),
                "periods_per_year": int(periods_per_year),
            },
            "run": {"record_trades": bool(record_trades)},
        },
        env=capture_environment(),
        seed=_extract_seed(normalized_params if isinstance(normalized_params, dict) else {}),
        trade_log=[_to_builtin(item) for item in portfolio.trades],
        risk_events=[_to_builtin(item) for item in (portfolio.risk_events or [])],
        equity_curve=np.asarray(portfolio.equity_curve, dtype=np.float64).tolist(),
        returns=np.asarray(portfolio.returns, dtype=np.float64).tolist(),
        performance={str(k): float(v) for k, v in performance.items()},
        risk={str(k): float(v) for k, v in risk.items()},
        trade_metrics={str(k): _to_builtin(v) for k, v in trade_metrics.items()},
    )
