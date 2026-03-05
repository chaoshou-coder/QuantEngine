from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from quantengine.interface.api import QuantEngineAPI
from quantengine.logging_config import get_logger

__all__ = ["__version__", "QuantEngineAPI", "get_logger"]

try:
    __version__ = version("quantengine")
except PackageNotFoundError:
    __version__ = "0.1.0"
