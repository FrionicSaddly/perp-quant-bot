"""Continuously log OpenNews items to disk to build a backtestable feature stream.

The 6551 API is recent-only, so we poll periodically and append new items (deduped by
id) to a parquet. Over days/weeks this accumulates the timestamped sentiment / market
signal history that an honest backtest needs. Output lives under data/raw (git-ignored).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..data.opennews import OpenNewsClient
from ..logging_conf import setup_logging

logger = setup_logging()


def _normalize(item: dict) -> dict:
    coins = item.get("coins")
    if isinstance(coins, list):
        coins = ",".join(str(c) for c in coins)
    return {
        "id": str(item.get("id")),
        "ts": item.get("ts"),
        "engineType": item.get("engineType"),
        "newsType": item.get("newsType"),
        "source": item.get("source"),
        "score": item.get("score"),
        "signal": item.get("signal") or item.get("tradingSignal") or item.get("direction"),
        "coins": coins,
        "text": (item.get("text") or item.get("description") or "")[:500],
    }


def run_logger(once: bool = False, interval: int = 90, limit: int = 200, out: str | None = None) -> None:
    cfg = load_config()
    out_path = Path(out) if out else cfg.raw_dir() / "opennews_log.parquet"

    seen: set[str] = set()
    if out_path.exists():
        try:
            seen = set(pd.read_parquet(out_path)["id"].astype(str))
        except Exception:  # noqa: BLE001
            seen = set()

    client = OpenNewsClient()
    logger.info("OpenNews logger -> {} (interval {}s, {} known ids)", out_path, interval, len(seen))
    try:
        while True:
            try:
                items = client.latest(limit=limit)
                rows = []
                for it in items:
                    rid = str(it.get("id"))
                    if rid and rid != "None" and rid not in seen:
                        seen.add(rid)
                        rec = _normalize(it)
                        rec["logged_at"] = pd.Timestamp.utcnow().isoformat()
                        rows.append(rec)
                if rows:
                    df = pd.DataFrame(rows)
                    if out_path.exists():
                        df = pd.concat([pd.read_parquet(out_path), df], ignore_index=True)
                    df.to_parquet(out_path)
                    logger.info("logged +{} new (total rows {})", len(rows), len(df))
                else:
                    logger.info("no new items this poll")
            except Exception as exc:  # noqa: BLE001
                logger.error("poll error: {}", exc)
            if once:
                break
            time.sleep(interval)
    finally:
        client.close()
