from __future__ import annotations

from collections.abc import Callable

from .base import BaseStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(name: str) -> Callable[[type[BaseStrategy]], type[BaseStrategy]]:
    def _decorator(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        key = name.strip().lower()
        if not key:
            raise ValueError("strategy name cannot be empty")
        STRATEGY_REGISTRY[key] = cls
        cls.name = key
        return cls

    return _decorator


def get_strategy(name: str) -> BaseStrategy:
    key = name.strip().lower()
    if key not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise KeyError(f"未知策略: {name}, 可选: {available}")
    return STRATEGY_REGISTRY[key]()


def list_strategies() -> list[str]:
    return sorted(STRATEGY_REGISTRY.keys())
