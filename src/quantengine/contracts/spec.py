from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")

DEFAULT_CONTRACT_SPECS: dict[str, dict[str, Any]] = {
    "XAUUSD": {
        "multiplier": 100.0,
        "min_tick": 0.01,
        "trading_sessions": [("00:00", "23:59")],
    }
}


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    multiplier: float
    min_tick: float
    trading_sessions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol 不能为空")
        if self.multiplier <= 0:
            raise ValueError("multiplier 必须大于 0")
        if self.min_tick <= 0:
            raise ValueError("min_tick 必须大于 0")
        if not self.trading_sessions:
            raise ValueError("trading_sessions 不能为空")
        for start, end in self.trading_sessions:
            _validate_hhmm(start)
            _validate_hhmm(end)


def get_contract_spec(symbol: str, overrides: dict[str, dict[str, Any]] | None = None) -> ContractSpec:
    symbol_key = _normalize_symbol(symbol)
    base = dict(DEFAULT_CONTRACT_SPECS.get(symbol_key, _default_fallback_spec()))
    override = _pick_override(symbol_key=symbol_key, overrides=overrides)
    if override:
        base.update(override)
    return contract_spec_from_dict(symbol_key, base)


def contract_spec_from_dict(symbol: str, raw: dict[str, Any]) -> ContractSpec:
    sessions = _normalize_sessions(raw.get("trading_sessions"))
    return ContractSpec(
        symbol=_normalize_symbol(symbol),
        multiplier=float(raw.get("multiplier", 1.0)),
        min_tick=float(raw.get("min_tick", 0.0001)),
        trading_sessions=sessions,
    )


def _normalize_symbol(symbol: str) -> str:
    token = symbol.strip().upper()
    if not token:
        raise ValueError("symbol 不能为空")
    return token


def _default_fallback_spec() -> dict[str, Any]:
    return {
        "multiplier": 1.0,
        "min_tick": 0.0001,
        "trading_sessions": [("00:00", "23:59")],
    }


def _pick_override(symbol_key: str, overrides: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if not overrides:
        return None
    for key, value in overrides.items():
        if _normalize_symbol(key) == symbol_key:
            if not isinstance(value, dict):
                raise ValueError("contracts override 必须是字典")
            return dict(value)
    return None


def _normalize_sessions(raw: Any) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return (("00:00", "23:59"),)
    if not isinstance(raw, (list, tuple)):
        raise ValueError("trading_sessions 必须是列表，例如 [['09:00','23:00']]")

    normalized: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("trading_sessions 每个区间必须是 [start, end]")
        start = str(item[0]).strip()
        end = str(item[1]).strip()
        _validate_hhmm(start)
        _validate_hhmm(end)
        normalized.append((start, end))
    return tuple(normalized)


def _validate_hhmm(value: str) -> None:
    if not _TIME_PATTERN.fullmatch(value):
        raise ValueError(f"时间格式必须为 HH:MM: {value}")
    hh = int(value[:2])
    mm = int(value[3:])
    if hh > 23 or mm > 59:
        raise ValueError(f"时间格式必须为 HH:MM: {value}")
