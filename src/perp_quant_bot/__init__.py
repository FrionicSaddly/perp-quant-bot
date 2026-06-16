"""perp-quant-bot: research-grade crypto perpetual-futures trading bot."""
from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config, load_secrets  # noqa: E402

__all__ = ["Config", "load_config", "load_secrets", "__version__"]
