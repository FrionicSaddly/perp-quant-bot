"""Empirical leakage detector: prove (or disprove) the edge is real, not a bug.

For one symbol it trains three quick models on a 70/30 time split and compares the
out-of-sample Sharpe:

* clean    — the real features.
* shuffled — labels randomly permuted; a healthy pipeline collapses to ~0 edge.
* leaked   — a deliberate future-return feature injected; OOS Sharpe should explode.

If `shuffled` is not near zero, something leaks. If `leaked` is not far above
`clean`, the detector (or the backtest) is broken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest import backtest_signal
from ..config import Config, load_config
from ..data.exchange import make_exchange
from ..logging_conf import setup_logging
from .train import build_model, prepare_dataset

logger = setup_logging()


def leak_check(cfg: Config | None = None, symbol: str | None = None, exchange=None) -> dict:
    cfg = cfg or load_config()
    symbol = symbol or cfg.universe.symbols[0]
    exchange = exchange or make_exchange(cfg)

    ds = prepare_dataset(cfg, symbol, exchange)
    X, y, ohlcv, funding, atr_pct = ds["X"], ds["y"], ds["ohlcv"], ds["funding"], ds["atr_pct"]
    n = len(X)
    if n < 200:
        raise RuntimeError(f"Not enough samples for leak check ({n})")
    split = int(n * 0.7)
    te_idx = X.index[split:]

    def oos_sharpe(X_tr, y_tr, X_te) -> float:
        model = build_model(cfg)
        model.fit(X_tr, y_tr)
        sig = pd.Series(model.predict_signal(X_te), index=X_te.index)
        bt = backtest_signal(ohlcv.loc[te_idx], sig, atr_pct.loc[te_idx], cfg, funding)
        return float(bt["metrics"]["sharpe"])

    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr = y.iloc[:split]

    clean = oos_sharpe(X_tr, y_tr, X_te)

    rng = np.random.default_rng(0)
    y_shuf = pd.Series(rng.permutation(y_tr.to_numpy()), index=y_tr.index)
    shuffled = oos_sharpe(X_tr, y_shuf, X_te)

    future = ohlcv["close"].pct_change().shift(-1)  # next-bar return = blatant leak
    X_tr_leak = X_tr.assign(_future=future.reindex(X_tr.index)).fillna(0.0)
    X_te_leak = X_te.assign(_future=future.reindex(X_te.index)).fillna(0.0)
    leaked = oos_sharpe(X_tr_leak, y_tr, X_te_leak)

    verdict = "OK" if (abs(shuffled) < max(0.5, abs(clean)) and leaked > clean + 1.0) else "SUSPECT"
    result = {"symbol": symbol, "clean": clean, "shuffled": shuffled, "leaked": leaked, "verdict": verdict}
    logger.info(
        "leakcheck {}: clean_sharpe={:.2f} shuffled={:.2f} leaked={:.2f} -> {}",
        symbol, clean, shuffled, leaked, verdict,
    )
    return result


def permutation_importance(cfg: Config | None = None, symbol: str | None = None, exchange=None) -> dict:
    """Out-of-sample permutation importance: how much holdout accuracy drops when
    each feature is shuffled. Low/negative drop = noise feature (prune candidate)."""
    cfg = cfg or load_config()
    symbol = symbol or cfg.universe.symbols[0]
    exchange = exchange or make_exchange(cfg)

    ds = prepare_dataset(cfg, symbol, exchange)
    X, y = ds["X"], ds["y"]
    n = len(X)
    if n < 200:
        raise RuntimeError(f"Not enough samples ({n})")
    split = int(n * 0.7)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    model = build_model(cfg)
    model.fit(X_tr, y_tr)
    classes = np.asarray(model.classes_)
    y_true = y_te.to_numpy()

    base_acc = float((classes[model.predict_proba(X_te).argmax(1)] == y_true).mean())
    rng = np.random.default_rng(0)
    drops: dict[str, float] = {}
    for col in X.columns:
        Xp = X_te.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        acc = float((classes[model.predict_proba(Xp).argmax(1)] == y_true).mean())
        drops[col] = base_acc - acc

    importance = pd.Series(drops).sort_values(ascending=False)
    logger.info("permutation importance {} (base_acc={:.3f}): top={}",
                symbol, base_acc, list(importance.head(5).index))
    return {"symbol": symbol, "base_accuracy": base_acc, "importance": importance}
