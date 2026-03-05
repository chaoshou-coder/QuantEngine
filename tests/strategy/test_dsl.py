from __future__ import annotations

import pytest

from quantengine.strategy.dsl import StrategyDSLSpec, build_strategy_from_dsl, load_strategy_dsl

V4_FRAMEWORKS = ["F1", "F2", "F3", "F4", "F5"]
V4_RISK_MODES = ["baseline", "conservative", "standard", "aggressive"]


@pytest.mark.parametrize("framework", V4_FRAMEWORKS)
@pytest.mark.parametrize("risk_mode", V4_RISK_MODES)
def test_v4_dsl_supports_all_framework_and_risk_mode_combinations(framework: str, risk_mode: str) -> None:
    spec = load_strategy_dsl(
        {
            "strategy": {
                "name": "psar_trade_assist_v4",
                "framework": framework,
                "risk_mode": risk_mode,
                "params": {"atr_period": 14},
            }
        }
    )
    strategy, params, weight = build_strategy_from_dsl(spec)
    assert strategy.name == "psar_trade_assist_v4"
    assert params["framework"] == framework
    assert params["risk_mode"] == risk_mode
    assert params["atr_period"] == 14
    assert weight == pytest.approx(1.0)


def test_v4_dsl_rejects_unknown_framework() -> None:
    with pytest.raises(ValueError, match="framework"):
        load_strategy_dsl(
            {
                "strategy": {
                    "name": "psar_trade_assist_v4",
                    "framework": "F9",
                    "risk_mode": "standard",
                }
            }
        )


def test_dsl_parses_yaml_text_and_builds_generic_strategy() -> None:
    spec = load_strategy_dsl(
        """
strategy:
  name: sma_cross
  weight: 0.6
  params:
    fast: 10
    slow: 50
"""
    )
    assert isinstance(spec, StrategyDSLSpec)

    strategy, params, weight = build_strategy_from_dsl(spec)
    assert strategy.name == "sma_cross"
    assert params == {"fast": 10, "slow": 50}
    assert weight == pytest.approx(0.6)
