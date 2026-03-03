from __future__ import annotations

from typing import Callable

INDICATOR_REGISTRY: dict[str, Callable[..., object]] = {}


def register_indicator(name: str, fn: Callable[..., object]) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("indicator name cannot be empty")
    INDICATOR_REGISTRY[key] = fn


def get_indicator(name: str) -> Callable[..., object]:
    key = name.strip().lower()
    if key not in INDICATOR_REGISTRY:
        options = ", ".join(sorted(INDICATOR_REGISTRY.keys()))
        raise KeyError(f"未知指标: {name}, 可选: {options}")
    return INDICATOR_REGISTRY[key]


def list_indicators() -> list[str]:
    return sorted(INDICATOR_REGISTRY.keys())
