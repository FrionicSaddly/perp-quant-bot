"""ccxt exchange factory (Bybit by default, testnet-aware)."""
from __future__ import annotations

import ccxt

from ..config import Config, Secrets, load_secrets
from ..logging_conf import setup_logging

logger = setup_logging()


def make_exchange(
    cfg: Config,
    secrets: Secrets | None = None,
    *,
    with_keys: bool = False,
    sandbox: bool | None = None,
    exchange_id: str | None = None,
):
    """Create a configured ccxt exchange instance.

    Parameters
    ----------
    with_keys:
        If True, attach API keys from .env (needed for private endpoints /
        testnet trading). Public market data does not need keys.
    sandbox:
        Override sandbox/testnet mode. Defaults to ``cfg.exchange.testnet``.
        Historical market data should use mainnet (sandbox=False) for full history.
    """
    secrets = secrets or load_secrets()
    ex_id = exchange_id or cfg.exchange.id
    if not hasattr(ccxt, ex_id):
        raise ValueError(f"Unknown ccxt exchange id: {ex_id}")
    klass = getattr(ccxt, ex_id)

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
    use_sandbox = cfg.exchange.testnet if sandbox is None else sandbox
    if use_sandbox:
        try:
            ex.set_sandbox_mode(True)
            logger.info("{} sandbox/testnet mode enabled", ex_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enable sandbox mode: {}", exc)
    return ex


def make_data_exchange(cfg: Config, secrets: Secrets | None = None):
    """Public market-data client: always mainnet (real history), keyless, read-only.

    Uses ``cfg.data.exchange_id`` if set (decouples the DATA venue from the
    EXECUTION venue, which honors cfg.exchange.id / cfg.exchange.testnet).
    """
    data_id = cfg.data.exchange_id or cfg.exchange.id
    return make_exchange(cfg, secrets, with_keys=False, sandbox=False, exchange_id=data_id)
