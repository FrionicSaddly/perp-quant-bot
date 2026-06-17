"""Tests for the OpenNews feature extractor (no network)."""
from __future__ import annotations

import json

import pandas as pd

from perp_quant_bot.features.news_features import (
    base_of,
    bucketed_features,
    coin_signal_panel,
    news_features_for_symbol,
)


def _sample_log() -> pd.DataFrame:
    rows = [
        {
            "ts": pd.Timestamp("2026-06-17T06:00:00Z"),
            "coins_json": json.dumps([{"symbol": "BTC", "signal": "long", "score": 80}]),
            "ai_signal": "long", "ai_score": 80,
        },
        {
            "ts": pd.Timestamp("2026-06-17T06:30:00Z"),
            "coins_json": json.dumps(
                [{"symbol": "BTC", "signal": "short", "score": 60},
                 {"symbol": "ETH", "signal": "long", "score": 70}]
            ),
            "ai_signal": "short", "ai_score": 60,
        },
        {
            "ts": pd.Timestamp("2026-06-17T07:10:00Z"),
            "coins_json": json.dumps([{"symbol": "BTC", "signal": "neutral", "score": 10}]),
            "ai_signal": "neutral", "ai_score": 10,
        },
    ]
    return pd.DataFrame(rows)


def test_base_of():
    assert base_of("BTC/USDT:USDT") == "BTC"
    assert base_of("ETHUSDT") == "ETH"


def test_coin_panel_and_buckets():
    panel = coin_signal_panel(_sample_log())
    assert set(panel["symbol"]) >= {"BTC", "ETH"}
    assert set(panel["signal"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    bk = bucketed_features(panel, freq="1h")
    b6 = bk[(bk["symbol"] == "BTC") & (bk["bucket"] == pd.Timestamp("2026-06-17T06:00:00Z"))]
    assert not b6.empty
    # 06:00 bucket: BTC long(+1) and short(-1) -> net 0, n=2, 1 long, 1 short
    assert float(b6["news_net_signal"].iloc[0]) == 0.0
    assert float(b6["news_n"].iloc[0]) == 2.0
    assert float(b6["news_n_long"].iloc[0]) == 1.0
    assert float(b6["news_n_short"].iloc[0]) == 1.0


def test_align_is_leak_safe():
    bk = bucketed_features(coin_signal_panel(_sample_log()), freq="1h")
    idx = pd.date_range("2026-06-17T05:00:00Z", periods=5, freq="1h", tz="UTC")
    feat = news_features_for_symbol(bk, "BTC/USDT:USDT", idx)
    assert list(feat.index) == list(idx)
    # before any news -> neutral 0
    assert float(feat.loc[idx[0], "news_n"]) == 0.0
    # at 06:00 -> the 06:00 bucket (n=2)
    assert float(feat.loc[pd.Timestamp("2026-06-17T06:00:00Z"), "news_n"]) == 2.0
