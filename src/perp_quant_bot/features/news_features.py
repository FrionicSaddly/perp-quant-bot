"""Turn the OpenNews log into leak-safe per-(time, symbol) features.

The 6551 feed gives an AI signal (long/short/neutral) + impact score per article and
per coin. We map signal -> {-1, 0, +1} and aggregate onto time buckets per coin:
net/mean signal, mean impact score, item and long/short counts. Aggregation uses each
item's own timestamp, and alignment onto bars is as-of/forward-fill, so a feature at
bar t uses only news published up to t (leak-free).

NOTE: these features can only be BACKTESTED once enough history has been logged; the
6551 API is recent-only. This module is the ready-to-use plumbing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

_SIGNAL_MAP = {"long": 1.0, "short": -1.0, "neutral": 0.0}

_AGG_COLS = [
    "news_net_signal", "news_mean_signal", "news_mean_score",
    "news_n", "news_n_long", "news_n_short",
]


def base_of(symbol: str) -> str:
    """Map a market symbol to its base asset, e.g. 'BTC/USDT:USDT' -> 'BTC'."""
    s = str(symbol).upper().split("/")[0].split(":")[0]
    return s.replace("USDT", "").strip() or str(symbol).upper()


def load_news_log(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def coin_signal_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Explode to one tidy row per (ts, base_symbol) with numeric signal + score."""
    recs: list[dict] = []
    cjs = df.get("coins_json")
    ai_sigs = df.get("ai_signal")
    ai_scores = df.get("ai_score")
    for i in range(len(df)):
        ts = df["ts"].iloc[i]
        cj = cjs.iloc[i] if cjs is not None else None
        ai_sig = ai_sigs.iloc[i] if ai_sigs is not None else None
        ai_score = ai_scores.iloc[i] if ai_scores is not None else None
        coins = []
        if isinstance(cj, str) and cj:
            try:
                coins = json.loads(cj)
            except Exception:  # noqa: BLE001
                coins = []
        for c in coins or []:
            if not isinstance(c, dict):
                continue
            sym = str(c.get("symbol") or "").upper().strip()
            if not sym:
                continue
            sig = c.get("signal") or ai_sig
            score = c.get("score")
            if score is None:
                score = ai_score
            recs.append(
                {
                    "ts": ts,
                    "symbol": base_of(sym),
                    "signal": _SIGNAL_MAP.get(sig, np.nan),
                    "score": float(score) if score is not None else np.nan,
                }
            )
    return pd.DataFrame(recs)


def bucketed_features(panel: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Aggregate coin signals into time buckets per symbol (the feature panel)."""
    if panel.empty:
        return pd.DataFrame(columns=["symbol", "bucket", *_AGG_COLS])
    p = panel.dropna(subset=["symbol"]).copy()
    p["bucket"] = p["ts"].dt.floor(freq)
    g = p.groupby(["symbol", "bucket"])
    out = g.agg(
        news_net_signal=("signal", "sum"),
        news_mean_signal=("signal", "mean"),
        news_mean_score=("score", "mean"),
        news_n=("signal", "size"),
        news_n_long=("signal", lambda s: float((s > 0).sum())),
        news_n_short=("signal", lambda s: float((s < 0).sum())),
    ).reset_index()
    return out


def news_features_for_symbol(bucketed: pd.DataFrame, market_symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of align a symbol's bucketed news features onto a bar index (leak-free).

    Bars before any news coverage get a neutral 0 (a constant prior, not the future).
    """
    base = base_of(market_symbol)
    if bucketed.empty:
        return pd.DataFrame(0.0, index=index, columns=_AGG_COLS)
    sub = bucketed[bucketed["symbol"] == base].set_index("bucket").sort_index()
    if sub.empty:
        return pd.DataFrame(0.0, index=index, columns=_AGG_COLS)
    aligned = sub[_AGG_COLS].reindex(index, method="ffill")
    return aligned.fillna(0.0)
