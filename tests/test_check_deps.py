"""check_deps 模块基础测试。"""

import sys

import pytest

from quantengine.check_deps import (
    check_core_deps,
    check_engine_deps,
    check_python_version,
    run_check,
)


def test_check_python_version() -> None:
    r = check_python_version()
    assert r.name == "Python"
    assert r.required == ">=3.11"
    assert r.installed is not None
    assert r.status in ("OK", "版本不符")
    if sys.version_info >= (3, 11):
        assert r.status == "OK"


def test_check_core_deps_returns_list() -> None:
    results = check_core_deps()
    assert isinstance(results, list)
    assert len(results) >= 8
    for r in results:
        assert hasattr(r, "name")
        assert hasattr(r, "required")
        assert hasattr(r, "installed")
        assert hasattr(r, "status")
        assert r.status in ("OK", "缺失", "版本不符")


def test_check_engine_deps_returns_list() -> None:
    results = check_engine_deps()
    assert isinstance(results, list)
    assert len(results) >= 5
    for r in results:
        assert hasattr(r, "name")
        assert hasattr(r, "status")


def test_run_check_structure() -> None:
    result = run_check(include_engine=False)
    assert "python" in result
    assert "core" in result
    assert "engine" in result
    assert result["engine"] is None
    assert "all_ok" in result
    assert isinstance(result["all_ok"], bool)


def test_run_check_with_engine() -> None:
    result = run_check(include_engine=True)
    assert result["engine"] is not None
    assert isinstance(result["engine"], list)
