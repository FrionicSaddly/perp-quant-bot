"""Offline deterministic tests for the microstructure collector (no network)."""
from __future__ import annotations

import csv

from perp_quant_bot.pipeline.microstructure_logger import (
    MicrostructureCollector,
    imbalance,
    microprice,
)


class FakeExchange:
    """Returns fixed book/trades/OI/funding so snapshot math is checkable."""

    def __init__(self):
        self.bids = [[100.0, 2.0], [99.0, 1.0], [98.0, 1.0], [97.0, 1.0], [96.0, 1.0]]
        self.asks = [[101.0, 1.0], [102.0, 1.0], [103.0, 1.0], [104.0, 1.0], [105.0, 1.0]]
        self.trades = [
            {"timestamp": 1000, "side": "buy", "amount": 3.0},
            {"timestamp": 1001, "side": "sell", "amount": 1.0},
            {"timestamp": 1002, "side": "buy", "amount": 2.0},
        ]

    def fetch_order_book(self, symbol, limit=25):
        return {"bids": self.bids, "asks": self.asks}

    def fetch_trades(self, symbol, since=None, limit=1000):
        return list(self.trades)

    def fetch_open_interest(self, symbol):
        return {"openInterestAmount": 12345.0, "openInterestValue": None}

    def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.0001, "markPrice": 100.5, "info": {"lastPrice": 100.4}}


def test_imbalance_and_microprice_math():
    bids = [[100.0, 2.0], [99.0, 1.0]]
    asks = [[101.0, 1.0], [102.0, 1.0]]
    imb1, bv, av = imbalance(bids, asks, 1)
    assert abs(imb1 - (2.0 - 1.0) / 3.0) < 1e-9 and bv == 2.0 and av == 1.0
    # microprice leans toward the thinner side (ask thinner -> price above mid 100.5)
    mp = microprice(100.0, 101.0, 2.0, 1.0)
    assert abs(mp - (100.0 * 1.0 + 101.0 * 2.0) / 3.0) < 1e-9
    assert mp > 100.5


def test_snapshot_fields_and_cvd_dedup(tmp_path):
    fake = FakeExchange()
    c = MicrostructureCollector(venue="bybit", symbols=["BTC/USDT:USDT"],
                                out_dir=tmp_path, exchange=fake)
    row = c.snapshot("BTC/USDT:USDT")

    assert abs(row["imb_5"] - (6.0 - 5.0) / 11.0) < 1e-9
    assert abs(row["imb_1"] - (2.0 - 1.0) / 3.0) < 1e-9
    assert row["cvd_delta"] == 4.0 and row["buy_vol"] == 5.0 and row["sell_vol"] == 1.0
    assert row["trade_count"] == 3
    assert row["open_interest"] == 12345.0
    assert row["funding_rate"] == 0.0001
    assert row["mark"] == 100.5 and row["last"] == 100.4
    assert row["bid"] == 100.0 and row["ask"] == 101.0

    # second poll: all trades are <= last seen ts -> no double counting
    row2 = c.snapshot("BTC/USDT:USDT")
    assert row2["cvd_delta"] == 0.0 and row2["trade_count"] == 0


def test_normalize_liquidation():
    from perp_quant_bot.pipeline.liquidations_logger import normalize_liquidation

    liq = {
        "timestamp": 123, "datetime": "2026-06-17T00:00:00Z", "symbol": "BTC/USDT:USDT",
        "side": "sell", "price": 65000.0, "contracts": 0.5, "quoteValue": 32500.0,
    }
    r = normalize_liquidation(liq, "bybit")
    assert r["ts"] == 123 and r["symbol"] == "BTC/USDT:USDT" and r["side"] == "sell"
    assert r["price"] == 65000.0 and r["amount"] == 0.5 and r["quote_value"] == 32500.0
    assert r["venue"] == "bybit" and r["logged_at"]

    # falls back to "amount" when "contracts" is absent
    liq2 = {"timestamp": 1, "symbol": "ETH/USDT:USDT", "side": "buy", "price": 1.0, "amount": 2.0}
    assert normalize_liquidation(liq2, "bybit")["amount"] == 2.0


def test_append_writes_csv_with_header(tmp_path):
    fake = FakeExchange()
    c = MicrostructureCollector(venue="bybit", symbols=["BTC/USDT:USDT"],
                                out_dir=tmp_path, exchange=fake)
    rows = c.poll_once()
    assert len(rows) == 1
    files = list(tmp_path.glob("bybit_micro_*.csv"))
    assert len(files) == 1
    with open(files[0], newline="", encoding="utf-8") as fh:
        rd = list(csv.DictReader(fh))
    assert len(rd) == 1
    assert rd[0]["symbol"] == "BTC/USDT:USDT"
    assert float(rd[0]["cvd_delta"]) == 4.0
