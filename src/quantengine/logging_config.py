"""Centralized logging configuration for QuantEngine.

Usage:
    from quantengine.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Step started", extra={"n_experiments": 40})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
) -> None:
    """Configure root logger for QuantEngine.

    Call once at application entry point. Subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger("quantengine")
    root.setLevel(level)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    root.addHandler(console)

    if log_file is not None:
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the quantengine namespace.

    If configure_logging() has not been called, applies a default config.
    """
    if not _CONFIGURED:
        configure_logging()
    if not name.startswith("quantengine"):
        name = f"quantengine.{name}"
    return logging.getLogger(name)
