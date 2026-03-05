"""依赖检查模块：校验 Python 版本与核心/可选依赖。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from typing import Literal

# 核心依赖：(显示名, 元数据包名, 最小版本约束)
_CORE_SPECS: list[tuple[str, str, str]] = [
    ("numpy", "numpy", ">=1.26.0"),
    ("pandas", "pandas", ">=2.0.0"),
    ("polars", "polars", ">=0.20.0"),
    ("pyarrow", "pyarrow", ">=14.0.0"),
    ("pydantic", "pydantic", ">=2.7.0"),
    ("pyyaml", "PyYAML", ">=6.0"),
    ("rich", "rich", ">=13.7.0"),
    ("click", "click", ">=8.1.0"),
]

# 可选 engine 依赖：(显示名, 元数据包名, 最小版本约束)
_ENGINE_SPECS: list[tuple[str, str, str]] = [
    ("cupy", "cupy-cuda12x", ">=13.0"),
    ("cudf", "cudf-cu12", ">=24.0"),
    ("optuna", "optuna", ">=3.6"),
    ("nevergrad", "nevergrad", ">=1.0"),
    ("deap", "deap", ">=1.4"),
    ("plotly", "plotly", ">=5.18"),
    ("matplotlib", "matplotlib", ">=3.8"),
    ("torch", "torch", ">=2.2"),
    ("numba", "numba", ">=0.59"),
]

# 备选包名（cupy-cuda12x 可能以 cupy 注册）
_ENGINE_ALTERNATIVES: dict[str, list[str]] = {
    "cupy-cuda12x": ["cupy", "cupy_cuda12x"],
    "cudf-cu12": ["cudf", "cudf_cu12"],
}


@dataclass
class CheckResult:
    """单条依赖检查结果。"""

    name: str
    required: str
    installed: str | None
    status: Literal["OK", "缺失", "版本不符"]


def _parse_spec(spec: str) -> tuple[str, str]:
    """解析 '>=1.26.0' 为 (op, version)。"""
    for op in (">=", ">", "==", "!=", "<=", "<"):
        if spec.startswith(op):
            return op, spec[len(op) :].strip()
    return ">=", spec


def _satisfies(installed: str, required: str) -> bool:
    """简单版本满足检查。支持 >= 约束。"""
    try:
        from packaging.version import Version

        op, req_ver = _parse_spec(required)
        if op == ">=":
            return Version(installed) >= Version(req_ver)
        return True
    except Exception:
        return True


def _get_installed_version(meta_name: str, alternatives: list[str] | None = None) -> str | None:
    """获取已安装版本，尝试主包名与备选包名。"""
    names = [meta_name]
    if alternatives:
        names = [meta_name] + alternatives
    for n in names:
        try:
            return get_version(n)
        except PackageNotFoundError:
            continue
    return None


def check_core_deps() -> list[CheckResult]:
    """检查核心依赖。"""
    results: list[CheckResult] = []
    for display_name, meta_name, required in _CORE_SPECS:
        installed = _get_installed_version(meta_name)
        if installed is None:
            results.append(CheckResult(name=display_name, required=required, installed=None, status="缺失"))
        elif _satisfies(installed, required):
            results.append(CheckResult(name=display_name, required=required, installed=installed, status="OK"))
        else:
            results.append(CheckResult(name=display_name, required=required, installed=installed, status="版本不符"))
    return results


def check_engine_deps() -> list[CheckResult]:
    """检查可选 engine 依赖。"""
    results: list[CheckResult] = []
    for display_name, meta_name, required in _ENGINE_SPECS:
        alts = _ENGINE_ALTERNATIVES.get(meta_name)
        installed = _get_installed_version(meta_name, alts)
        if installed is None:
            results.append(CheckResult(name=display_name, required=required, installed=None, status="缺失"))
        elif _satisfies(installed, required):
            results.append(CheckResult(name=display_name, required=required, installed=installed, status="OK"))
        else:
            results.append(CheckResult(name=display_name, required=required, installed=installed, status="版本不符"))
    return results


def check_python_version() -> CheckResult:
    """检查 Python 版本 >= 3.11。"""
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):  # noqa: UP036
        return CheckResult(name="Python", required=">=3.11", installed=ver, status="OK")
    return CheckResult(name="Python", required=">=3.11", installed=ver, status="版本不符")


def run_check(include_engine: bool = False) -> dict:
    """执行完整检查，供 CLI 调用。

    Returns:
        {
            "python": CheckResult,
            "core": list[CheckResult],
            "engine": list[CheckResult] | None,
            "all_ok": bool,
        }
    """
    python_result = check_python_version()
    core_results = check_core_deps()
    engine_results = check_engine_deps() if include_engine else None

    core_ok = all(r.status == "OK" for r in core_results)
    engine_ok = all(r.status == "OK" for r in engine_results) if engine_results else True
    all_ok = python_result.status == "OK" and core_ok and engine_ok

    return {
        "python": python_result,
        "core": core_results,
        "engine": engine_results,
        "all_ok": all_ok,
    }
