"""ccxt exchange factory (Bybit by default, testnet-aware)."""
from __future__ import annotations

import ccxt

from ..config import Config, Secrets, load_secrets
from ..logging_conf import setup_logging

logger = setup_logging()


def make_exchange(cfg: Config, secrets: Secrets | None = None, *, with_keys: bool = False):
    """Create a configured ccxt exchange instance.

    Parameters
    ----------
    with_keys:
        If True, attach API keys from .env (needed for private endpoints /
        testnet trading). Public market data does not need keys.
    """
    secrets = secrets or load_secrets()
    if not hasattr(ccxt, cfg.exchange.id):
        raise ValueError(f"Unknown ccxt exchange id: {cfg.exchange.id}")
    klass = getattr(ccxt, cfg.exchange.id)

    params: dict = {
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {"defaultType": "swap"},  # perpetual swaps
    }
    if with_keys:
        if not (secrets.bybit_api_key and secrets.bybit_api_secret):
            logger.warning("with_keys=True but BYBIT_API_KEY/SECRET are not set in .env")
        params["apiKey"] = secrets.bybit_api_key
        params["secret"] = secrets.bybit_api_secret

    ex = klass(params)
    if cfg.exchange.testnet:
        try:
            ex.set_sandbox_mode(True)
            logger.info("{} sandbox/testnet mode enabled", cfg.exchange.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enable sandbox mode: {}", exc)
    return ex
