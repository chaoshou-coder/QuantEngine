from __future__ import annotations

import numpy as np
import pytest

from quantengine.engine.commission import (
    FixedCommission,
    PercentCommission,
    TieredCommission,
    build_commission,
)


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        (3.0, 6.0),
        (0.0, 0.0),
        (-5.0, 0.0),
    ],
)
def test_fixed_commission_compute(quantity: float, expected: float) -> None:
    model = FixedCommission(value=2.0)
    assert model.compute(notional=1_000.0, quantity=quantity) == pytest.approx(expected)


def test_fixed_commission_compute_vector_caps_negative_quantity() -> None:
    model = FixedCommission(value=1.5)
    quantity = np.array([2.0, 0.0, -1.0], dtype=float)
    notional = np.array([100.0, 200.0, 300.0], dtype=float)
    actual = model.compute_vector(notional=notional, quantity=quantity)
    np.testing.assert_allclose(actual, np.array([3.0, 0.0, 0.0], dtype=float))


@pytest.mark.parametrize(
    ("notional", "expected"),
    [
        (1_000.0, 5.0),
        (0.0, 0.0),
        (-100.0, 0.0),
    ],
)
def test_percent_commission_compute(notional: float, expected: float) -> None:
    model = PercentCommission(rate=0.005)
    assert model.compute(notional=notional, quantity=1.0) == pytest.approx(expected)


def test_percent_commission_compute_vector_caps_negative_notional() -> None:
    model = PercentCommission(rate=0.01)
    notional = np.array([100.0, 0.0, -20.0], dtype=float)
    quantity = np.array([1.0, 1.0, 1.0], dtype=float)
    actual = model.compute_vector(notional=notional, quantity=quantity)
    np.testing.assert_allclose(actual, np.array([1.0, 0.0, 0.0], dtype=float))


def test_tiered_commission_compute_uses_sorted_thresholds() -> None:
    model = TieredCommission(tiers=[(10_000.0, 0.001), (1_000.0, 0.003)])
    assert model.compute(notional=500.0, quantity=1.0) == pytest.approx(1.5)
    assert model.compute(notional=5_000.0, quantity=1.0) == pytest.approx(5.0)
    assert model.compute(notional=20_000.0, quantity=1.0) == pytest.approx(20.0)


def test_tiered_commission_compute_uses_fallback_when_no_tiers() -> None:
    model = TieredCommission(tiers=[], fallback_rate=0.002)
    assert model.compute(notional=2_000.0, quantity=1.0) == pytest.approx(4.0)


def test_tiered_commission_compute_vector_applies_each_tier() -> None:
    model = TieredCommission(tiers=[(100.0, 0.01), (500.0, 0.005), (1_000.0, 0.003)], fallback_rate=0.002)
    notional = np.array([50.0, 200.0, 800.0, 2_000.0], dtype=float)
    quantity = np.ones_like(notional)
    actual = model.compute_vector(notional=notional, quantity=quantity)
    expected = np.array([0.5, 1.0, 2.4, 6.0], dtype=float)
    np.testing.assert_allclose(actual, expected)


def test_tiered_commission_compute_vector_negative_notional_is_zero() -> None:
    model = TieredCommission(tiers=[(100.0, 0.01)], fallback_rate=0.01)
    notional = np.array([-10.0, 0.0, 50.0], dtype=float)
    quantity = np.ones_like(notional)
    actual = model.compute_vector(notional=notional, quantity=quantity)
    np.testing.assert_allclose(actual, np.array([0.0, 0.0, 0.5], dtype=float))


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("fixed", FixedCommission),
        ("percent", PercentCommission),
        ("tiered", TieredCommission),
    ],
)
def test_build_commission_returns_expected_model(name: str, expected_type: type) -> None:
    model = build_commission(model=name, value=0.01, tiers=[(100.0, 0.005)])
    assert isinstance(model, expected_type)


def test_build_commission_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="未知手续费模型"):
        build_commission(model="bad-model", value=0.01)
