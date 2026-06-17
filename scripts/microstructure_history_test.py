"""Honest historical test: does order-flow + positioning beat technical-only?

We cannot get historical order-BOOK depth (must collect live), but Binance
publishes, for years back:
  * per-bar taker-buy volume in klines  -> aggressive order-flow (CVD-style)
  * 5-min metrics                        -> taker buy/sell ratio, top-trader and
                                            retail long/short ratios, open interest

So the core microstructure hypothesis is testable NOW, on lots of data, with the
project's leak-safe machinery (triple-barrier + purged walk-forward + DSR). We
compare a technical-only model against technical + flow + positioning on the same
OOS folds. If the extra signals don't lift DSR / hit-rate, the honest answer is
"no edge here", and we've saved weeks of waiting.

Usage: python scripts/microstructure_history_test.py BTCUSDT 2025-06-01 2026-05-31 5m
"""
from __future__ import annotations

import sys

import httpx
import numpy as np
import pandas as pd

from perp_quant_bot.backtest import backtest_signal
from perp_quant_bot.backtest.metrics import deflated_sharpe_ratio, infer_bars_per_year
from perp_quant_bot.config import load_config
from perp_quant_bot.data.binance_vision import klines_ohlcv, metrics_history
from perp_quant_bot.features.technical import compute_atr, technical_features
from perp_quant_bot.labeling import triple_barrier_labels
from perp_quant_bot.models import LightGBMModel
from perp_quant_bot.validation import purged_walk_forward_splits


def flow_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Order-flow from per-bar taker-buy volume (known at bar close -> leak-safe)."""
    vol = ohlcv["volume"].replace(0.0, np.nan)
    tbr = (ohlcv["taker_buy_volume"] / vol).clip(0.0, 1.0)
    imb = 2.0 * tbr - 1.0  # +1 all aggressive buys, -1 all aggressive sells
    f = {"taker_buy_ratio": tbr, "taker_imb": imb}
    for w in (6, 12, 48):
        f[f"taker_imb_mean_{w}"] = imb.rolling(w).mean()
    signed = imb * vol
    f["taker_flow_z_48"] = (signed - signed.rolling(48).mean()) / signed.rolling(48).std()
    return pd.DataFrame(f, index=ohlcv.index)


def positioning_features(metrics: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Crowding / smart-money positioning from Binance 5-min metrics (ffill = past-only)."""
    if metrics is None or metrics.empty:
        return pd.DataFrame(index=index)
    m = metrics.reindex(index, method="ffill")
    f: dict[str, pd.Series] = {}
    for col in ("sum_taker_long_short_vol_ratio", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio"):
        if col in m.columns:
            s = m[col]
            f[col] = s
            f[f"{col}_chg_12"] = s - s.shift(12)
            f[f"{col}_z_48"] = (s - s.rolling(48).mean()) / s.rolling(48).std()
    if "sum_open_interest" in m.columns:
        oi = m["sum_open_interest"]
        f["oi_chg_12"] = oi.pct_change(12)
        f["oi_z_48"] = (oi - oi.rolling(48).mean()) / oi.rolling(48).std()
    return pd.DataFrame(f, index=index)


def evaluate(name, X, y, t1, w, ohlcv, atr_pct, cfg) -> dict:
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[X.dropna(how="any").index[0]:].ffill().dropna(how="any")
    common = X.index.intersection(y.index)
    X, yy, tt, ww = X.loc[common], y.loc[common], t1.loc[common], w.loc[common]
    splits = purged_walk_forward_splits(X.index, tt, cfg.validation.n_splits, cfg.validation.embargo_bars)
    oos = pd.Series(0, index=X.index, dtype=int)
    fold_sharpes = []
    for tr, te in splits:
        m = LightGBMModel(params=cfg.model.params, threshold=cfg.model.prob_threshold,
                          n_seeds=cfg.model.n_seeds, calibrate=cfg.model.calibrate)
        m.fit(X.iloc[tr], yy.iloc[tr], sample_weight=ww.iloc[tr].to_numpy())
        sig = m.predict_signal(X.iloc[te])
        oos.iloc[te] = sig
        ti = X.index[te]
        bt = backtest_signal(ohlcv.loc[ti], pd.Series(sig, index=ti), atr_pct.loc[ti], cfg, None)
        fold_sharpes.append(bt["metrics"]["sharpe"])
    first = splits[0][1][0]
    oi = X.index[first:]
    bt = backtest_signal(ohlcv.loc[oi], oos.loc[oi], atr_pct.loc[oi], cfg, None)
    bpy = infer_bars_per_year(oi)
    dsr = deflated_sharpe_ratio(bt["results"]["net_ret"], [s / np.sqrt(bpy) for s in fold_sharpes if np.isfinite(s)])
    return {
        "name": name, "n_features": X.shape[1], "dsr": float(dsr),
        "sharpe": bt["metrics"]["sharpe"], "hit": bt["metrics"].get("hit_rate", float("nan")),
        "ret": bt["metrics"]["total_return"], "traded": float((oos.loc[oi] != 0).mean()),
    }


def run(symbol: str, start: str, end: str, interval: str) -> None:
    import os

    cfg = load_config()
    # Optional cost override to separate "signal" from "cost drag" (e.g. maker fills).
    if os.environ.get("PQB_FEE"):
        cfg.backtest.fee_rate = float(os.environ["PQB_FEE"])
    if os.environ.get("PQB_SLIP") is not None and os.environ.get("PQB_SLIP") != "":
        cfg.backtest.slippage_bps = float(os.environ["PQB_SLIP"])
    print(f"costs: fee={cfg.backtest.fee_rate * 100:.3f}% slippage={cfg.backtest.slippage_bps}bps")
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        print(f"downloading {symbol} {interval} klines + metrics {start}..{end} ...")
        ohlcv = klines_ohlcv("um", symbol, interval, start, end, client)
        metrics = metrics_history(symbol, start, end, client)
    if ohlcv.empty:
        print("no klines"); return
    print(f"klines: {len(ohlcv)} bars | metrics: {len(metrics)} rows "
          f"({list(metrics.columns) if not metrics.empty else 'NONE'})")

    atr = compute_atr(ohlcv, cfg.features.atr_period)
    atr_pct = (atr / ohlcv["close"])
    labels = triple_barrier_labels(ohlcv, atr, cfg)
    y = labels["label"].astype(int)
    t1 = labels["t1"]
    w = labels["w"].astype(float)

    tech = technical_features(ohlcv, cfg)
    flow = flow_features(ohlcv)
    pos = positioning_features(metrics, ohlcv.index)

    results = [
        evaluate("technical-only", tech.copy(), y, t1, w, ohlcv, atr_pct, cfg),
        evaluate("+ order-flow", pd.concat([tech, flow], axis=1), y, t1, w, ohlcv, atr_pct, cfg),
        evaluate("+ flow + positioning", pd.concat([tech, flow, pos], axis=1), y, t1, w, ohlcv, atr_pct, cfg),
    ]

    print(f"\n========= HONEST OOS: {symbol} {interval} ({start}..{end}) =========")
    print(f"{'model':<24}{'feats':>6}{'DSR':>7}{'sharpe':>8}{'hit':>7}{'ret':>9}{'traded':>8}")
    for r in results:
        print(f"{r['name']:<24}{r['n_features']:>6}{r['dsr']:>7.2f}{r['sharpe']:>8.2f}"
              f"{(r['hit'] or 0):>7.1%}{r['ret']:>9.1%}{r['traded']:>8.1%}")


if __name__ == "__main__":
    args = sys.argv[1:]
    symbol = args[0] if len(args) > 0 else "BTCUSDT"
    start = args[1] if len(args) > 1 else "2025-06-01"
    end = args[2] if len(args) > 2 else "2026-05-31"
    interval = args[3] if len(args) > 3 else "5m"
    run(symbol, start, end, interval)
