from __future__ import annotations

import pytest

from quantengine.contracts.spec import ContractSpec, get_contract_spec


def test_get_contract_spec_returns_builtin_xauusd_defaults() -> None:
    spec = get_contract_spec("XAUUSD")
    assert isinstance(spec, ContractSpec)
    assert spec.symbol == "XAUUSD"
    assert spec.multiplier > 0.0
    assert spec.min_tick > 0.0
    assert len(spec.trading_sessions) >= 1


def test_get_contract_spec_allows_config_override() -> None:
    spec = get_contract_spec(
        "xauusd",
        overrides={
            "XAUUSD": {
                "multiplier": 50.0,
                "min_tick": 0.05,
                "trading_sessions": [["01:00", "23:00"]],
            }
        },
    )
    assert spec.symbol == "XAUUSD"
    assert spec.multiplier == pytest.approx(50.0)
    assert spec.min_tick == pytest.approx(0.05)
    assert spec.trading_sessions == (("01:00", "23:00"),)


def test_get_contract_spec_rejects_invalid_session_format() -> None:
    with pytest.raises(ValueError, match="HH:MM"):
        get_contract_spec(
            "XAUUSD",
            overrides={
                "XAUUSD": {
                    "trading_sessions": [["0100", "2300"]],
                }
            },
        )
