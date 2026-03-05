from . import technical
from .registry import get_indicator, list_indicator_versions, list_indicators, register_indicator

register_indicator("sma", technical.sma, version="1.0.0")
register_indicator("ema", technical.ema, version="1.0.0")
register_indicator("rsi", technical.rsi, version="1.0.0")
register_indicator("macd", technical.macd, version="1.0.0")
register_indicator("atr", technical.atr, version="1.0.0")
register_indicator("parabolic_sar", technical.parabolic_sar, version="1.0.0")
register_indicator("adx", technical.adx, version="1.0.0")

__all__ = [
    "get_indicator",
    "list_indicator_versions",
    "list_indicators",
    "register_indicator",
    "technical",
]
