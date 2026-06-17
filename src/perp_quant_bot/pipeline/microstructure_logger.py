"""Continuously collect Bybit perp microstructure -> append-only daily CSV.

This captures the short-horizon signals that bar OHLCV lacks and that actually
carry predictive power on perps:
  * order-book imbalance & microprice (top 1 / 5 / 25 levels)
  * trade-flow CVD (taker buy volume - sell volume) since the last poll
  * open interest, funding rate, mark / last price

Leak-safety: every row is stamped with the wall-clock collection time. Any
feature built later must use only rows at or before the decision bar. The
collector only *records*; it never looks ahead.

Robustness: uses ``fetchMarkets=['linear']`` so Bybit's geo-flaky spot category
is never loaded (that endpoint intermittently 403s from some regions). Each
sub-call is independently guarded so one failing field never drops the row.

Storage: append-only daily CSV (``<venue>_micro_YYYY-MM-DD.csv``) so a cheap
always-on host or a scheduled CI job can accumulate history cheaply.
"""
from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt

from ..logging_conf import setup_logging

logger = setup_logging()

MICRO_UNIVERSE = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

FIELDS = [
    "ts", "venue", "symbol", "last", "mark", "bid", "ask", "spread_bps",
    "imb_1", "imb_5", "imb_25", "bid_vol_5", "ask_vol_5", "microprice",
    "open_interest", "funding_rate", "buy_vol", "sell_vol", "cvd_delta", "trade_count",
]


def imbalance(bids: list, asks: list, n: int) -> tuple[float, float, float]:
    """Top-n order-book volume imbalance in [-1, 1]; +1 = all bid (buy pressure)."""
    bv = float(sum(lvl[1] for lvl in bids[:n]))
    av = float(sum(lvl[1] for lvl in asks[:n]))
    tot = bv + av
    return ((bv - av) / tot if tot > 0 else 0.0), bv, av


def microprice(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
    """Size-weighted fair price; leans toward the thinner side (likely move)."""
    tot = bid_size + ask_size
    if tot <= 0:
        return 0.5 * (bid + ask)
    return (bid * ask_size + ask * bid_size) / tot


class MicrostructureCollector:
    def __init__(
        self,
        venue: str = "bybit",
        symbols: list[str] | None = None,
        out_dir: str | Path | None = None,
        ob_limit: int = 25,
        exchange=None,
    ):
        self.venue = venue
        self.symbols = symbols or MICRO_UNIVERSE
        self.ob_limit = int(ob_limit)
        self.out_dir = Path(out_dir or "data/microstructure")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._last_trade_ts: dict[str, int] = {}
        if exchange is not None:
            self.ex = exchange
        else:
            self.ex = getattr(ccxt, venue)({
                "enableRateLimit": True,
                "timeout": 20000,
                "options": {"defaultType": "swap", "fetchMarkets": ["linear"]},
            })
            self._load_markets_retry()

    def _load_markets_retry(self, retries: int = 6) -> None:
        for i in range(retries):
            try:
                self.ex.load_markets()
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("load_markets retry {}/{}: {}", i + 1, retries,
                               str(exc).splitlines()[0][:80])
                time.sleep(2.0 * (i + 1))
        logger.error("load_markets failed after {} retries; calls may degrade", retries)

    def snapshot(self, symbol: str) -> dict:
        row: dict[str, object] = {k: None for k in FIELDS}
        row["ts"] = datetime.now(timezone.utc).isoformat()
        row["venue"] = self.venue
        row["symbol"] = symbol

        # --- order book ---
        try:
            ob = self.ex.fetch_order_book(symbol, limit=self.ob_limit)
            bids, asks = ob.get("bids") or [], ob.get("asks") or []
            if bids and asks:
                bid, ask = float(bids[0][0]), float(asks[0][0])
                bsz, asz = float(bids[0][1]), float(asks[0][1])
                mid = 0.5 * (bid + ask)
                row["bid"], row["ask"] = bid, ask
                row["spread_bps"] = (ask - bid) / mid * 1e4 if mid > 0 else None
                row["microprice"] = microprice(bid, ask, bsz, asz)
                row["imb_1"] = imbalance(bids, asks, 1)[0]
                imb5, bv5, av5 = imbalance(bids, asks, 5)
                row["imb_5"], row["bid_vol_5"], row["ask_vol_5"] = imb5, bv5, av5
                row["imb_25"] = imbalance(bids, asks, 25)[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug("ob {}: {}", symbol, exc)

        # --- trade-flow CVD since last poll ---
        try:
            since = self._last_trade_ts.get(symbol)
            trades = self.ex.fetch_trades(symbol, since=since, limit=1000)
            buy = sell = 0.0
            cnt = 0
            max_ts = since or 0
            for t in trades:
                ts = int(t.get("timestamp") or 0)
                if since and ts <= since:
                    continue
                amt = float(t.get("amount") or 0.0)
                side = t.get("side")
                if side == "buy":
                    buy += amt
                elif side == "sell":
                    sell += amt
                cnt += 1
                max_ts = max(max_ts, ts)
            if max_ts:
                self._last_trade_ts[symbol] = max_ts
            row["buy_vol"], row["sell_vol"] = buy, sell
            row["cvd_delta"] = buy - sell
            row["trade_count"] = cnt
        except Exception as exc:  # noqa: BLE001
            logger.debug("trades {}: {}", symbol, exc)

        # --- open interest ---
        try:
            oi = self.ex.fetch_open_interest(symbol)
            row["open_interest"] = oi.get("openInterestAmount") or oi.get("openInterestValue")
        except Exception as exc:  # noqa: BLE001
            logger.debug("oi {}: {}", symbol, exc)

        # --- funding + mark + last (one call) ---
        try:
            fr = self.ex.fetch_funding_rate(symbol)
            row["funding_rate"] = fr.get("fundingRate")
            row["mark"] = fr.get("markPrice")
            info = fr.get("info") or {}
            row["last"] = info.get("lastPrice") or row["mark"]
        except Exception as exc:  # noqa: BLE001
            logger.debug("funding {}: {}", symbol, exc)

        return row

    def _path_for(self, ts_iso: str) -> Path:
        return self.out_dir / f"{self.venue}_micro_{ts_iso[:10]}.csv"

    def append(self, rows: list[dict]) -> None:
        for row in rows:
            p = self._path_for(str(row["ts"]))
            is_new = not p.exists()
            with open(p, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if is_new:
                    w.writeheader()
                w.writerow(row)

    def poll_rows(self) -> list[dict]:
        """Snapshot every symbol and return the rows WITHOUT writing to disk.

        Used by the CI accumulator, which persists to a single parquet on a data
        branch. CVD dedup state lives on the instance, so looping over this on one
        collector keeps trade-flow interval-aligned.
        """
        return [self.snapshot(s) for s in self.symbols]

    def poll_once(self) -> list[dict]:
        rows = self.poll_rows()
        self.append(rows)
        return rows

    def run(self, interval: float = 15.0, once: bool = False, duration: float | None = None) -> None:
        """Poll forever (or for ``duration`` seconds, or just ``once``).

        Ctrl-C stops cleanly. Each poll is independently guarded, so a transient
        network error logs and retries on the next tick rather than crashing the
        long-running collector.
        """
        logger.info("microstructure collector: venue={} symbols={} interval={}s -> {}",
                    self.venue, len(self.symbols), interval, self.out_dir)
        start = time.time()
        polls = 0
        try:
            while True:
                t0 = time.time()
                try:
                    rows = self.poll_once()
                    polls += 1
                    got = sum(1 for r in rows if r.get("imb_5") is not None)
                    logger.info("poll {}: logged {} rows ({} with OB) @ {}",
                                polls, len(rows), got, rows[0]["ts"] if rows else "-")
                except Exception as exc:  # noqa: BLE001
                    logger.error("poll failed (will retry next tick): {}", exc)
                if once:
                    break
                if duration is not None and (time.time() - start) >= duration:
                    logger.info("duration {}s reached after {} polls; stopping", duration, polls)
                    break
                time.sleep(max(0.0, interval - (time.time() - t0)))
        except KeyboardInterrupt:
            logger.info("interrupted; collected {} polls into {}", polls, self.out_dir)


def run_microstructure_logger(
    venue: str = "bybit", interval: float = 15.0, once: bool = False,
    symbols: list[str] | None = None, out_dir: str | None = None,
    duration: float | None = None,
) -> None:
    MicrostructureCollector(venue=venue, symbols=symbols, out_dir=out_dir).run(
        interval=interval, once=once, duration=duration
    )
