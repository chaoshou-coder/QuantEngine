from __future__ import annotations

import pytest

from quantengine.indicators.registry import (
    get_indicator,
    list_indicator_versions,
    list_indicators,
    register_indicator,
)


def test_indicator_registry_supports_versioned_lookup_and_latest_resolution() -> None:
    name = "unit_test_indicator"

    def _v1(*_args, **_kwargs) -> float:
        return 1.0

    def _v2(*_args, **_kwargs) -> float:
        return 2.0

    register_indicator(name, _v1, version="1.0.0")
    register_indicator(name, _v2, version="1.1.0")

    assert get_indicator(name, version="1.0.0") is _v1
    assert get_indicator(name) is _v2
    assert name in list_indicators()
    assert list_indicator_versions(name) == ["1.0.0", "1.1.0"]


def test_indicator_registry_raises_for_unknown_version() -> None:
    with pytest.raises(KeyError, match="9\\.9\\.9"):
        get_indicator("sma", version="9.9.9")
