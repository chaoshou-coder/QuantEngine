# 导入示例策略以触发注册
from . import examples as _examples  # noqa: F401
from .base import BaseStrategy, ParameterSpace, cartesian_from_spaces
from .dsl import StrategyDSLSpec, build_strategy_from_dsl, load_strategy_dsl, load_strategy_from_dsl
from .registry import get_strategy, list_strategies, register_strategy
from .signal import clip_signal, to_position

__all__ = [
    "BaseStrategy",
    "ParameterSpace",
    "StrategyDSLSpec",
    "build_strategy_from_dsl",
    "cartesian_from_spaces",
    "clip_signal",
    "get_strategy",
    "load_strategy_dsl",
    "load_strategy_from_dsl",
    "list_strategies",
    "register_strategy",
    "to_position",
]
