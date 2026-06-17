"""CI burst-collector for Bybit perp liquidations over WebSocket (GitHub Actions).

Streams liquidation events for a bounded duration and appends them to a single
parquet (deduped) on the ``liquidations-data`` branch. Liquidations are sporadic,
so an empty window is normal and not an error.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from perp_quant_bot.pipeline.liquidations_logger import run_liquidations_logger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="parquet path to append to")
    ap.add_argument("--duration", type=float, default=600.0, help="seconds to stream")
    ap.add_argument("--venue", default="bybit")
    args = ap.parse_args()

    try:
        events = run_liquidations_logger(venue=args.venue, duration=args.duration)
    except Exception as exc:  # noqa: BLE001
        print(f"liquidation stream failed: {exc}", file=sys.stderr)
        events = []

    if not events:
        print("no liquidations in window (normal for calm markets)")
        return 0

    df = pd.DataFrame(events)
    if os.path.exists(args.out):
        try:
            df = pd.concat([pd.read_parquet(args.out), df], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            print(f"could not read existing parquet ({exc}); starting fresh", file=sys.stderr)
    df = df.drop_duplicates(subset=["ts", "symbol", "side", "price", "amount"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    print(f"collected {len(events)} liquidation events (total {len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
