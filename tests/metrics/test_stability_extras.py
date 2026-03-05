"""Tests for BH correction and parameter sensitivity analysis in stability.py."""

from __future__ import annotations

import numpy as np

from quantengine.metrics.stability import (
    benjamini_hochberg_correction,
    parameter_sensitivity_analysis,
)


def test_bh_all_significant():
    p_values = [0.001, 0.005, 0.01]
    result = benjamini_hochberg_correction(p_values, alpha=0.05)
    assert result.method.lower() == "benjamini-hochberg"
    assert result.alpha == 0.05
    assert all(result.rejected)
    assert len(result.adjusted_p_values) == 3
    assert result.summary


def test_bh_none_significant():
    p_values = [0.5, 0.8, 0.95]
    result = benjamini_hochberg_correction(p_values, alpha=0.05)
    assert not any(result.rejected)
    assert result.significant_count == 0


def test_bh_mixed():
    p_values = [0.01, 0.04, 0.5, 0.9]
    result = benjamini_hochberg_correction(p_values, alpha=0.05)
    assert len(result.adjusted_p_values) == 4
    assert result.significant_count >= 1


def test_bh_single_pvalue():
    result = benjamini_hochberg_correction([0.03], alpha=0.05)
    assert len(result.adjusted_p_values) == 1
    assert result.rejected[0] is True


def test_bh_numpy_input():
    p_values = np.array([0.01, 0.1, 0.5])
    result = benjamini_hochberg_correction(p_values, alpha=0.05)
    assert len(result.adjusted_p_values) == 3


def test_sensitivity_numeric_params():
    base = {"fast": 20.0, "slow": 50.0}

    def evaluate(params):
        return params["fast"] / params["slow"]

    results = parameter_sensitivity_analysis(base, evaluate, perturbation=0.1)
    assert len(results) >= 2
    param_names = [r.param_name for r in results]
    assert "fast" in param_names
    assert "slow" in param_names
    for r in results:
        assert r.base_value > 0
        assert r.base_score > 0


def test_sensitivity_skips_non_numeric():
    base = {"name": "strategy_a", "period": 14.0}

    def evaluate(params):
        return float(params["period"])

    results = parameter_sensitivity_analysis(base, evaluate, perturbation=0.1)
    param_names = [r.param_name for r in results]
    assert "name" not in param_names
    assert "period" in param_names


def test_sensitivity_zero_base_value():
    base = {"x": 0.0, "y": 10.0}

    def evaluate(params):
        return params["x"] + params["y"]

    results = parameter_sensitivity_analysis(base, evaluate, perturbation=0.1)
    param_names = [r.param_name for r in results]
    assert "y" in param_names
