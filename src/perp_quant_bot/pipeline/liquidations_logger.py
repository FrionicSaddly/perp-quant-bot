"""WebSocket liquidations collector for Bybit perps (ccxt.pro).

Liquidation cascades are one of the most predictive short-horizon perp signals
(forced flow that overshoots, then snaps back). Bybit only streams them over
WebSocket, so this uses ccxt.pro to subscribe per symbol and append each event.

Bounded by ``duration`` so it fits a scheduled CI job; designed to fail soft
(retry market load, swallow transient socket errors) so a long run never crashes.
Liquidations are sporadic — short bursts capture whatever happens in the window;
over many runs a useful sample accumulates.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ..logging_conf import setup_logging
from .microstructure_logger import MICRO_UNIVERSE

logger = setup_logging()

LIQ_FIELDS = ["ts", "datetime", "venue", "symbol", "side", "price", "amount", "quote_value", "logged_at"]


def normalize_liquidation(liq: dict, venue: str) -> dict:
    """Map a ccxt unified liquidation to our flat schema (defensive .get)."""
    amount = liq.get("contracts")
    if amount is None:
        amount = liq.get("amount")
    return {
        "ts": liq.get("timestamp"),
        "datetime": liq.get("datetime"),
        "venue": venue,
        "symbol": liq.get("symbol"),
        "side": liq.get("side"),
        "price": liq.get("price"),
        "amount": amount,
        "quote_value": liq.get("quoteValue"),
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }


async def _load_markets_retry(ex, retries: int = 6) -> bool:
    for i in range(retries):
        try:
            await ex.load_markets()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("WS load_markets retry {}/{}: {}", i + 1, retries,
                           str(exc).splitlines()[0][:80])
            await asyncio.sleep(2.0 * (i + 1))
    return False


async def collect_liquidations(
    venue: str = "bybit", symbols: list[str] | None = None, duration: float = 600.0
) -> list[dict]:
    """Subscribe to liquidations for ``symbols`` and collect events for ``duration`` s."""
    import ccxt.pro as ccxtpro

    symbols = symbols or MICRO_UNIVERSE
    ex = getattr(ccxtpro, venue)({
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "fetchMarkets": ["linear"]},
    })
    events: list[dict] = []
    if not await _load_markets_retry(ex):
        logger.error("WS load_markets failed after retries; no liquidations collected")
        await ex.close()
        return events

    stop = asyncio.Event()

    async def watch(sym: str) -> None:
        while not stop.is_set():
            try:
                liqs = await ex.watch_liquidations(sym)
                for liq in liqs:
                    events.append(normalize_liquidation(liq, venue))
                    logger.info("LIQ {} {} amt={} @ {}", sym, liq.get("side"),
                                liq.get("contracts") or liq.get("amount"), liq.get("price"))
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.debug("watch_liquidations {} err: {}", sym, exc)
                await asyncio.sleep(1.0)

    logger.info("WS liquidations: venue={} symbols={} duration={}s", venue, len(symbols), duration)
    tasks = [asyncio.create_task(watch(s)) for s in symbols]
    try:
        await asyncio.sleep(duration)
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await ex.close()
        except Exception:  # noqa: BLE001
            pass
    logger.info("collected {} liquidation events", len(events))
    return events


def run_liquidations_logger(
    venue: str = "bybit", symbols: list[str] | None = None, duration: float = 600.0
) -> list[dict]:
    return asyncio.run(collect_liquidations(venue=venue, symbols=symbols, duration=duration))
