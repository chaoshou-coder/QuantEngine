from __future__ import annotations

import re
from collections.abc import Callable

INDICATOR_REGISTRY: dict[str, dict[str, Callable[..., object]]] = {}
LATEST_INDICATOR_VERSION: dict[str, str] = {}
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def _normalize_name(name: str) -> str:
    key = name.strip().lower()
    if not key:
        raise ValueError("indicator name cannot be empty")
    return key


def _normalize_version(version: str) -> str:
    token = version.strip()
    if not _SEMVER_PATTERN.fullmatch(token):
        raise ValueError(f"indicator version 必须是 semver 格式(x.y.z): {version}")
    return token


def _semver_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def register_indicator(name: str, fn: Callable[..., object], version: str = "1.0.0") -> None:
    key = _normalize_name(name)
    version_key = _normalize_version(version)

    version_map = INDICATOR_REGISTRY.setdefault(key, {})
    version_map[version_key] = fn

    latest = LATEST_INDICATOR_VERSION.get(key)
    if latest is None or _semver_key(version_key) >= _semver_key(latest):
        LATEST_INDICATOR_VERSION[key] = version_key


def get_indicator(name: str, version: str | None = None) -> Callable[..., object]:
    key = _normalize_name(name)
    if key not in INDICATOR_REGISTRY:
        options = ", ".join(list_indicators())
        raise KeyError(f"未知指标: {name}, 可选: {options}")
    if version is None:
        latest = LATEST_INDICATOR_VERSION[key]
        return INDICATOR_REGISTRY[key][latest]

    version_key = _normalize_version(version)
    versions = INDICATOR_REGISTRY[key]
    if version_key not in versions:
        available = ", ".join(list_indicator_versions(key))
        raise KeyError(f"未知指标版本: {name}@{version_key}, 可选: {available}")
    return versions[version_key]


def list_indicators() -> list[str]:
    return sorted(INDICATOR_REGISTRY.keys())


def list_indicator_versions(name: str) -> list[str]:
    key = _normalize_name(name)
    if key not in INDICATOR_REGISTRY:
        return []
    return sorted(INDICATOR_REGISTRY[key].keys(), key=_semver_key)
