"""Self-contained single-poll OpenNews logger for CI (GitHub Actions).

No project imports / heavy deps — only httpx + pandas + pyarrow — so CI stays fast.
Reads OPENNEWS_TOKEN from the environment (a GitHub secret), polls once, dedupes by
id against an existing parquet, and appends new rows.

Captures the FULL structure: article-level aiRating (signal/grade/score), the
per-coin list (each {symbol, signal, score, grade}) as JSON, and the raw item (minus
the long text) so nothing of value is lost for later feature engineering.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx
import pandas as pd

API_BASE = "https://ai.6551.io"


def _record(it: dict, now: str) -> dict:
    ai = it.get("aiRating") or {}
    coins = it.get("coins")
    raw = {k: v for k, v in it.items() if k != "text"}
    return {
        "id": str(it.get("id")),
        "ts": it.get("ts"),
        "engineType": it.get("engineType"),
        "newsType": it.get("newsType"),
        "source": it.get("source"),
        "score": it.get("score"),
        "ai_signal": ai.get("signal"),
        "ai_grade": ai.get("grade"),
        "ai_score": ai.get("score"),
        "coins_json": json.dumps(coins, ensure_ascii=False) if coins is not None else None,
        "raw": json.dumps(raw, ensure_ascii=False),
        "text": (it.get("text") or it.get("description") or "")[:300],
        "logged_at": now,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="parquet path to append to")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    token = os.environ.get("OPENNEWS_TOKEN", "")
    if not token:
        print("OPENNEWS_TOKEN not set", file=sys.stderr)
        return 1

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = httpx.post(
        f"{API_BASE}/open/news_search",
        headers=headers,
        json={"limit": args.limit, "page": 1},
        timeout=30.0,
    )
    resp.raise_for_status()
    items = resp.json().get("data") or []

    seen: set[str] = set()
    if os.path.exists(args.out):
        try:
            seen = set(pd.read_parquet(args.out)["id"].astype(str))
        except Exception:  # noqa: BLE001
            seen = set()

    now = pd.Timestamp.now(tz="UTC").isoformat()
    rows = [
        _record(it, now)
        for it in items
        if str(it.get("id")) not in seen and str(it.get("id")) not in ("None", "")
    ]

    if not rows:
        print("no new items")
        return 0

    df = pd.DataFrame(rows)
    if os.path.exists(args.out):
        df = pd.concat([pd.read_parquet(args.out), df], ignore_index=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    print(f"appended {len(rows)} new (total {len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
