from __future__ import annotations

from quantengine.interface.api import QuantEngineAPI
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

__all__ = ["__version__"]
__all__.append("QuantEngineAPI")

try:
    __version__ = version("quantengine")
except PackageNotFoundError:
    __version__ = "0.1.0"
