from __future__ import annotations

import numpy as np
import pytest

from quantengine.engine.slippage import (
    FixedSlippage,
    PercentSlippage,
    VolumeSlippage,
    build_slippage,
)


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        (1.0, 100.5),
        (-1.0, 99.5),
        (0.0, 100.5),
    ],
)
def test_fixed_slippage_adjust_price(side: float, expected: float) -> None:
    model = FixedSlippage(points=0.5)
    assert model.adjust_price(price=100.0, side=side, quantity=2.0) == expected


def test_fixed_slippage_adjust_price_vector() -> None:
    model = FixedSlippage(points=0.2)
    price = np.array([10.0, 20.0, 30.0], dtype=float)
    side = np.array([1.0, -1.0, 0.0], dtype=float)
    quantity = np.array([1.0, 2.0, 3.0], dtype=float)
    actual = model.adjust_price_vector(price=price, side=side, quantity=quantity)
    expected = np.array([10.2, 19.8, 30.2], dtype=float)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        (1.0, 101.0),
        (-1.0, 99.0),
    ],
)
def test_percent_slippage_adjust_price(side: float, expected: float) -> None:
    model = PercentSlippage(rate=0.01)
    assert model.adjust_price(price=100.0, side=side, quantity=1.0) == pytest.approx(expected)


def test_percent_slippage_adjust_price_vector() -> None:
    model = PercentSlippage(rate=0.02)
    price = np.array([100.0, 100.0], dtype=float)
    side = np.array([1.0, -1.0], dtype=float)
    quantity = np.array([1.0, 1.0], dtype=float)
    actual = model.adjust_price_vector(price=price, side=side, quantity=quantity)
    np.testing.assert_allclose(actual, np.array([102.0, 98.0], dtype=float))


def test_volume_slippage_adjust_price_uses_participation_cap() -> None:
    model = VolumeSlippage(impact=2.0, max_ratio=0.05)
    actual = model.adjust_price(price=100.0, side=1.0, quantity=10.0, bar_volume=100.0)
    # participation=min(10/100, 0.05)=0.05; ratio=impact*participation=0.1
    assert actual == pytest.approx(110.0)


def test_volume_slippage_adjust_price_handles_zero_volume() -> None:
    model = VolumeSlippage(impact=1.0, max_ratio=0.2)
    actual = model.adjust_price(price=50.0, side=-1.0, quantity=2.0, bar_volume=0.0)
    # safe volume=1.0; participation=min(2/1, 0.2)=0.2; ratio=0.2
    assert actual == pytest.approx(40.0)


def test_volume_slippage_adjust_price_vector_with_bar_volume() -> None:
    model = VolumeSlippage(impact=1.5, max_ratio=0.1)
    price = np.array([100.0, 100.0], dtype=float)
    side = np.array([1.0, -1.0], dtype=float)
    quantity = np.array([5.0, 50.0], dtype=float)
    bar_volume = np.array([100.0, 1000.0], dtype=float)
    actual = model.adjust_price_vector(price=price, side=side, quantity=quantity, bar_volume=bar_volume)
    # both participation=0.05, ratio=0.075
    np.testing.assert_allclose(actual, np.array([107.5, 92.5], dtype=float))


def test_volume_slippage_adjust_price_vector_without_bar_volume_defaults_to_ones() -> None:
    model = VolumeSlippage(impact=1.0, max_ratio=0.05)
    price = np.array([10.0, 10.0], dtype=float)
    side = np.array([1.0, -1.0], dtype=float)
    quantity = np.array([10.0, 10.0], dtype=float)
    actual = model.adjust_price_vector(price=price, side=side, quantity=quantity, bar_volume=None)
    # safe volume defaults to 1 -> participation capped to 0.05
    np.testing.assert_allclose(actual, np.array([10.5, 9.5], dtype=float))


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("fixed", FixedSlippage),
        (" percent ", PercentSlippage),
        ("VOLUME", VolumeSlippage),
    ],
)
def test_build_slippage_returns_expected_model(name: str, expected_type: type) -> None:
    model = build_slippage(model=name, value=0.01, impact=1.2)
    assert isinstance(model, expected_type)


def test_build_slippage_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="未知滑点模型"):
        build_slippage(model="unknown", value=0.0)
