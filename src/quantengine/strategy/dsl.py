from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .base import BaseStrategy
from .registry import get_strategy

V4_FRAMEWORKS = {"F1", "F2", "F3", "F4", "F5"}
V4_RISK_MODES = {"baseline", "conservative", "standard", "aggressive"}


@dataclass(frozen=True)
class StrategyDSLSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


def load_strategy_dsl(source: str | Path | dict[str, Any]) -> StrategyDSLSpec:
    payload = _load_source(source)
    return parse_strategy_dsl(payload)


def parse_strategy_dsl(payload: dict[str, Any]) -> StrategyDSLSpec:
    root = payload.get("strategy", payload)
    if not isinstance(root, dict):
        raise ValueError("strategy DSL 必须是字典结构")

    name_raw = root.get("name")
    name = str(name_raw).strip().lower() if name_raw is not None else ""
    if not name:
        raise ValueError("strategy.name 不能为空")

    params = _coerce_params(root.get("params"))
    framework = root.get("framework")
    risk_mode = root.get("risk_mode")
    if framework is not None:
        params["framework"] = str(framework).strip().upper()
    if risk_mode is not None:
        params["risk_mode"] = str(risk_mode).strip().lower()

    weight = float(root.get("weight", 1.0))
    if weight <= 0:
        raise ValueError("strategy.weight 必须大于 0")

    _validate_v4_fields(name=name, params=params)
    return StrategyDSLSpec(name=name, params=params, weight=weight)


def build_strategy_from_dsl(spec: StrategyDSLSpec) -> tuple[BaseStrategy, dict[str, Any], float]:
    strategy = get_strategy(spec.name)
    return strategy, dict(spec.params), float(spec.weight)


def load_strategy_from_dsl(source: str | Path | dict[str, Any]) -> tuple[BaseStrategy, dict[str, Any], float]:
    spec = load_strategy_dsl(source)
    return build_strategy_from_dsl(spec)


def _load_source(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return dict(source)
    if isinstance(source, Path):
        return _load_yaml_text(source.read_text(encoding="utf-8"))
    if isinstance(source, str):
        token = source.strip()
        if not token:
            raise ValueError("strategy DSL 不能为空")
        if "\n" not in source:
            maybe_path = Path(token)
            if maybe_path.suffix.lower() in {".yml", ".yaml"} and maybe_path.exists():
                return _load_yaml_text(maybe_path.read_text(encoding="utf-8"))
        return _load_yaml_text(source)
    raise TypeError(f"不支持的 strategy DSL 输入类型: {type(source)}")


def _load_yaml_text(raw: str) -> dict[str, Any]:
    parsed = yaml.safe_load(raw)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("strategy DSL 顶层必须为字典")
    return parsed


def _coerce_params(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("strategy.params 必须是字典")
    return dict(raw)


def _validate_v4_fields(name: str, params: dict[str, Any]) -> None:
    if name != "psar_trade_assist_v4":
        return
    framework = str(params.get("framework", "")).strip().upper()
    if framework not in V4_FRAMEWORKS:
        options = ", ".join(sorted(V4_FRAMEWORKS))
        raise ValueError(f"psar_trade_assist_v4 的 framework 必须在 [{options}] 中")

    risk_mode = str(params.get("risk_mode", "")).strip().lower()
    if risk_mode not in V4_RISK_MODES:
        options = ", ".join(sorted(V4_RISK_MODES))
        raise ValueError(f"psar_trade_assist_v4 的 risk_mode 必须在 [{options}] 中")
