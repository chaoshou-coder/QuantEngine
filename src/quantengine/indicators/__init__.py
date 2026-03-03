from . import technical
from .registry import get_indicator, list_indicators, register_indicator

register_indicator("sma", technical.sma)
register_indicator("ema", technical.ema)
register_indicator("rsi", technical.rsi)
register_indicator("macd", technical.macd)
register_indicator("atr", technical.atr)

__all__ = [
    "get_indicator",
    "list_indicators",
    "register_indicator",
    "technical",
]
