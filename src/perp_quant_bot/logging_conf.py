"""Centralized logging via loguru."""
from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def setup_logging(level: str = "INFO"):
    global _CONFIGURED
    if _CONFIGURED:
        return logger
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - <level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )
    _CONFIGURED = True
    return logger


__all__ = ["logger", "setup_logging"]
