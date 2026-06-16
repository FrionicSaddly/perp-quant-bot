"""Historical OHLCV download with pagination + parquet caching."""
from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

from ..config import Config
from ..logging_conf import setup_logging
from .exchange import make_data_exchange

logger = setup_logging()

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _sanitize(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-")


def _cache_path(cfg: Config, symbol: str) -> Path:
    venue = cfg.data.exchange_id or cfg.exchange.id
    name = f"{venue}_{_sanitize(symbol)}_{cfg.universe.timeframe}_ohlcv.parquet"
    return cfg.raw_dir() / name


def download_ohlcv(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Paginate ``fetch_ohlcv`` from *since_ms* to *until_ms* (default: now)."""
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    until_ms = until_ms or exchange.milliseconds()
    cursor = since_ms
    rows: list[list] = []

    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        except ccxt.BaseError as exc:
            logger.warning("fetch_ohlcv error for {}: {} (retrying once)", symbol, exc)
            time.sleep(2)
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)

        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor:
            break  # no forward progress -> stop (avoids infinite loop)
        cursor = next_cursor
        logger.debug("{}: {} bars (cursor={})", symbol, len(rows), cursor)
        time.sleep(max(exchange.rateLimit, 50) / 1000)

    if not rows:
        return pd.DataFrame(columns=_OHLCV_COLS)

    df = pd.DataFrame(rows, columns=["timestamp", *_OHLCV_COLS])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[df.index <= pd.to_datetime(until_ms, unit="ms", utc=True)]
    return df.astype(float)


def load_or_download_ohlcv(cfg: Config, symbol: str, exchange=None, force: bool = False) -> pd.DataFrame:
    """Return cached OHLCV if present, otherwise download and cache it."""
    path = _cache_path(cfg, symbol)
    if path.exists() and not force:
        logger.info("Loading cached OHLCV: {}", path.name)
        return pd.read_parquet(path)

    exchange = exchange or make_data_exchange(cfg)
    since_ms = exchange.parse8601(cfg.universe.since)
    logger.info("Downloading OHLCV {} {} since {}", symbol, cfg.universe.timeframe, cfg.universe.since)
    df = download_ohlcv(exchange, symbol, cfg.universe.timeframe, since_ms)
    if df.empty:
        logger.warning("No OHLCV returned for {}", symbol)
        return df
    df.to_parquet(path)
    logger.info("Saved {} bars -> {}", len(df), path.name)
    return df
