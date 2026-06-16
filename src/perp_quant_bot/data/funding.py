"""Funding-rate and open-interest history (perp-specific signals & costs)."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import Config
from ..logging_conf import setup_logging
from .exchange import make_exchange

logger = setup_logging()


def _sanitize(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-")


def _cache_path(cfg: Config, symbol: str) -> Path:
    name = f"{cfg.exchange.id}_{_sanitize(symbol)}_funding_oi.parquet"
    return cfg.raw_dir() / name


def download_funding(exchange, symbol: str, since_ms: int, until_ms: int | None = None) -> pd.DataFrame:
    """Funding-rate history -> DataFrame[funding_rate] indexed by UTC timestamp."""
    if not exchange.has.get("fetchFundingRateHistory"):
        logger.warning("{} has no fetchFundingRateHistory", exchange.id)
        return pd.DataFrame(columns=["funding_rate"])

    until_ms = until_ms or exchange.milliseconds()
    cursor = since_ms
    rows: list[dict] = []
    while cursor < until_ms:
        try:
            batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=200)
        except Exception as exc:  # noqa: BLE001
            logger.warning("funding history error {}: {}", symbol, exc)
            break
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if last_ts is None or last_ts + 1 <= cursor:
            break
        cursor = last_ts + 1
        if len(batch) < 200:
            break
        time.sleep(max(exchange.rateLimit, 50) / 1000)

    if not rows:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(
        {
            "timestamp": [r["timestamp"] for r in rows],
            "funding_rate": [r.get("fundingRate") for r in rows],
        }
    )
    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").astype(float)


def download_open_interest(
    exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int | None = None
) -> pd.DataFrame:
    """Open-interest history -> DataFrame[open_interest]. Best-effort (venue-dependent)."""
    if not exchange.has.get("fetchOpenInterestHistory"):
        logger.info("{} has no fetchOpenInterestHistory; skipping OI", exchange.id)
        return pd.DataFrame(columns=["open_interest"])

    until_ms = until_ms or exchange.milliseconds()
    cursor = since_ms
    rows: list[dict] = []
    while cursor < until_ms:
        try:
            batch = exchange.fetch_open_interest_history(symbol, timeframe, since=cursor, limit=200)
        except Exception as exc:  # noqa: BLE001
            logger.info("open-interest history unavailable for {}: {}", symbol, exc)
            break
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if last_ts is None or last_ts + 1 <= cursor:
            break
        cursor = last_ts + 1
        if len(batch) < 200:
            break
        time.sleep(max(exchange.rateLimit, 50) / 1000)

    if not rows:
        return pd.DataFrame(columns=["open_interest"])
    df = pd.DataFrame(
        {
            "timestamp": [r["timestamp"] for r in rows],
            "open_interest": [
                r.get("openInterestAmount") or r.get("openInterestValue") for r in rows
            ],
        }
    )
    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").astype(float)


def load_or_download_funding(cfg: Config, symbol: str, exchange=None, force: bool = False) -> pd.DataFrame:
    """Return cached funding(+OI) frame, else download. Tolerant of missing data."""
    path = _cache_path(cfg, symbol)
    if path.exists() and not force:
        logger.info("Loading cached funding/OI: {}", path.name)
        return pd.read_parquet(path)

    exchange = exchange or make_exchange(cfg)
    since_ms = exchange.parse8601(cfg.universe.since)

    funding = download_funding(exchange, symbol, since_ms)
    parts = [funding]
    if cfg.features.include_open_interest:
        oi = download_open_interest(exchange, symbol, cfg.universe.timeframe, since_ms)
        if not oi.empty:
            parts.append(oi)

    if all(p.empty for p in parts):
        logger.warning("No funding/OI data for {}", symbol)
        return pd.DataFrame(columns=["funding_rate", "open_interest"])

    df = pd.concat(parts, axis=1).sort_index()
    df.to_parquet(path)
    logger.info("Saved funding/OI ({} rows) -> {}", len(df), path.name)
    return df
