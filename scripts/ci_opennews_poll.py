"""Self-contained single-poll OpenNews logger for CI (GitHub Actions).

No project imports / heavy deps — only httpx + pandas + pyarrow — so CI stays fast.
Reads OPENNEWS_TOKEN from the environment (a GitHub secret), polls once, dedupes by
id against an existing parquet, and appends new rows.
"""
from __future__ import annotations

import argparse
import os
import sys

import httpx
import pandas as pd

API_BASE = "https://ai.6551.io"


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

    rows = []
    now = pd.Timestamp.now(tz="UTC").isoformat()
    for it in items:
        rid = str(it.get("id"))
        if not rid or rid == "None" or rid in seen:
            continue
        coins = it.get("coins")
        if isinstance(coins, list):
            coins = ",".join(str(c) for c in coins)
        rows.append(
            {
                "id": rid,
                "ts": it.get("ts"),
                "engineType": it.get("engineType"),
                "newsType": it.get("newsType"),
                "source": it.get("source"),
                "score": it.get("score"),
                "signal": it.get("signal") or it.get("tradingSignal") or it.get("direction"),
                "coins": coins,
                "text": (it.get("text") or it.get("description") or "")[:500],
                "logged_at": now,
            }
        )

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
