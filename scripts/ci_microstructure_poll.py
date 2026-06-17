"""CI burst-collector for Bybit perp microstructure (GitHub Actions).

Runs the real ``MicrostructureCollector`` for a bounded duration (so it fits a
scheduled job), accumulating order-book imbalance / CVD / OI / funding rows, then
appends them to a single parquet (deduped by ts+symbol) that lives on the
``microstructure-data`` branch. Reuses the project collector so CI and local runs
never drift.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from perp_quant_bot.pipeline.microstructure_logger import MICRO_UNIVERSE, MicrostructureCollector


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="parquet path to append to")
    ap.add_argument("--duration", type=float, default=280.0, help="seconds to collect")
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between polls")
    ap.add_argument("--venue", default="bybit")
    args = ap.parse_args()

    collector = MicrostructureCollector(venue=args.venue, symbols=MICRO_UNIVERSE)
    rows: list[dict] = []
    start = time.time()
    polls = 0
    while time.time() - start < args.duration:
        t0 = time.time()
        try:
            rows.extend(collector.poll_rows())
            polls += 1
        except Exception as exc:  # noqa: BLE001
            print(f"poll error (continuing): {exc}", file=sys.stderr)
        time.sleep(max(0.0, args.interval - (time.time() - t0)))

    if not rows:
        print("no rows collected")
        return 0

    df = pd.DataFrame(rows)
    if os.path.exists(args.out):
        try:
            df = pd.concat([pd.read_parquet(args.out), df], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            print(f"could not read existing parquet ({exc}); starting fresh", file=sys.stderr)
    df = df.drop_duplicates(subset=["ts", "symbol"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    print(f"collected {len(rows)} rows over {polls} polls (total {len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
