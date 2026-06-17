"""WebSocket liquidations collector for Bybit perps (direct V5 public stream).

Liquidation cascades are one of the most predictive short-horizon perp signals
(forced flow that overshoots, then snaps back). Bybit only streams them over
WebSocket. We connect directly to the documented V5 public endpoint via aiohttp
(already a dependency) and subscribe to the ``allLiquidation.{symbol}`` topic.

Why not ccxt.pro: its bybit ``load_markets`` fails on the linear instruments
endpoint in this ccxt build (both locally and in CI), which blocks ``watch_*``.
The raw stream needs no market load and is robust.

Bounded by ``duration`` so it fits a scheduled CI job; fails soft (transient
socket errors are swallowed, empty windows are normal — liquidations are
sporadic). Over many short runs a useful sample accumulates.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

from ..logging_conf import setup_logging
from .microstructure_logger import MICRO_UNIVERSE

logger = setup_logging()

BYBIT_WS_LINEAR = "wss://stream.bybit.com/v5/public/linear"
LIQ_FIELDS = ["ts", "datetime", "venue", "symbol", "side", "price", "amount", "logged_at"]


def to_bybit_symbol(sym: str) -> str:
    """`BTC/USDT:USDT` -> `BTCUSDT` (Bybit WS topic symbol)."""
    return sym.split(":")[0].replace("/", "")


def normalize_bybit_liquidation(item: dict, venue: str = "bybit") -> dict:
    """Map a Bybit V5 `allLiquidation` data item to our flat schema.

    Item fields: T (ms ts), s (symbol), S (liquidated side Buy/Sell), v (size), p (price).
    """
    ts = item.get("T")
    dt = None
    if ts is not None:
        try:
            dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            dt = None
    return {
        "ts": int(ts) if ts is not None else None,
        "datetime": dt,
        "venue": venue,
        "symbol": item.get("s"),
        "side": item.get("S"),
        "price": float(item["p"]) if item.get("p") is not None else None,
        "amount": float(item["v"]) if item.get("v") is not None else None,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }


async def collect_liquidations(
    venue: str = "bybit", symbols: list[str] | None = None, duration: float = 600.0
) -> list[dict]:
    """Subscribe to Bybit liquidations for ``symbols`` and collect for ``duration`` s."""
    import aiohttp

    if venue != "bybit":
        logger.error("direct WS liquidations only implemented for bybit (got {})", venue)
        return []

    symbols = symbols or MICRO_UNIVERSE
    topics = [f"allLiquidation.{to_bybit_symbol(s)}" for s in symbols]
    events: list[dict] = []
    logger.info("WS liquidations: {} topics, duration={}s", len(topics), duration)

    end = time.time() + duration
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(BYBIT_WS_LINEAR, heartbeat=20.0) as ws:
                await ws.send_json({"op": "subscribe", "args": topics})
                logger.info("subscribed; streaming liquidations...")
                while time.time() < end:
                    remaining = end - time.time()
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=min(remaining, 10.0))
                    except asyncio.TimeoutError:
                        continue
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            payload = json.loads(msg.data)
                        except Exception:  # noqa: BLE001
                            continue
                        topic = payload.get("topic", "")
                        if topic.startswith("allLiquidation"):
                            for it in payload.get("data", []) or []:
                                ev = normalize_bybit_liquidation(it, venue)
                                events.append(ev)
                                logger.info("LIQ {} {} amt={} @ {}",
                                            ev["symbol"], ev["side"], ev["amount"], ev["price"])
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.warning("WS closed/error; stopping")
                        break
    except Exception as exc:  # noqa: BLE001
        logger.error("WS liquidations failed: {}", str(exc).splitlines()[0][:120])

    logger.info("collected {} liquidation events", len(events))
    return events


def run_liquidations_logger(
    venue: str = "bybit", symbols: list[str] | None = None, duration: float = 600.0
) -> list[dict]:
    return asyncio.run(collect_liquidations(venue=venue, symbols=symbols, duration=duration))
