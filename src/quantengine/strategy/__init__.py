from .base import BaseStrategy, ParameterSpace, cartesian_from_spaces
from .registry import get_strategy, list_strategies, register_strategy
from .signal import clip_signal, to_position

# 导入示例策略以触发注册
from . import examples as _examples  # noqa: F401

__all__ = [
    "BaseStrategy",
    "ParameterSpace",
    "cartesian_from_spaces",
    "clip_signal",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "to_position",
]
