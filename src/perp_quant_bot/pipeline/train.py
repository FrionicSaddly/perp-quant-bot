"""Training pipeline: data -> features -> labels -> purged walk-forward -> save."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..backtest import backtest_signal
from ..backtest.metrics import deflated_sharpe_ratio, infer_bars_per_year
from ..config import Config, load_config
from ..data import load_or_download_funding, load_or_download_ohlcv
from ..data.exchange import make_data_exchange
from ..features import build_feature_matrix
from ..labeling import meta_labels, primary_side, triple_barrier_labels
from ..logging_conf import setup_logging
from ..models import LightGBMModel
from ..validation import purged_walk_forward_splits

logger = setup_logging()


def _meta_proba(cfg: Config, X_tr, y_tr, w_tr, X_te):
    """Seed-ensembled P(primary side is correct) on the test fold (binary GBM)."""
    import lightgbm as lgb

    params = dict(cfg.model.params)
    params.pop("objective", None)
    params.pop("num_class", None)
    n_seeds = max(1, int(cfg.model.n_seeds))
    proba = np.zeros(len(X_te), dtype=float)
    for s in range(n_seeds):
        clf = lgb.LGBMClassifier(**params, objective="binary", random_state=s, verbose=-1)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        proba += clf.predict_proba(X_te)[:, 1]
    return proba / n_seeds


def regime_breakdown(ohlcv, signal, atr_pct, cfg, funding=None) -> dict:
    """OOS hit-rate / return split by volatility and trend regime.

    Reveals *where* (if anywhere) the signal has edge, so trading can be
    restricted to favorable regimes. Backward-looking buckets only.
    """
    close = ohlcv["close"]
    fwd = close.pct_change().shift(-1)  # next-bar return the signal is judged on
    sig = signal.reindex(close.index).fillna(0)
    traded = sig != 0
    out: dict[str, dict] = {}

    vol = atr_pct.reindex(close.index)
    hi_vol = vol > vol.median()
    er_up = close.pct_change(24) > 0  # simple trend proxy
    buckets = {
        "low_vol": ~hi_vol, "high_vol": hi_vol,
        "trend_up": er_up, "trend_dn": ~er_up,
    }
    for name, mask in buckets.items():
        m = mask & traded
        n = int(m.sum())
        if n == 0:
            out[name] = {"n": 0, "hit_rate": float("nan"), "mean_ret_bps": float("nan")}
            continue
        # direction-aware: did the signed bet match the next-bar move?
        correct = (np.sign(sig[m]) == np.sign(fwd[m])) & (fwd[m] != 0)
        out[name] = {
            "n": n,
            "hit_rate": round(float(correct.mean()), 4),
            "mean_ret_bps": round(float((np.sign(sig[m]) * fwd[m]).mean() * 1e4), 2),
        }
    return out


def build_model(cfg: Config) -> LightGBMModel:
    return LightGBMModel(
        params=cfg.model.params,
        threshold=cfg.model.prob_threshold,
        n_seeds=cfg.model.n_seeds,
        calibrate=cfg.model.calibrate,
    )


def _sanitize(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-")


def model_path(cfg: Config, symbol: str):
    return cfg.models_dir() / f"{cfg.exchange.id}_{_sanitize(symbol)}_{cfg.universe.timeframe}.pkl"


def prepare_dataset(cfg: Config, symbol: str, exchange=None):
    ohlcv = load_or_download_ohlcv(cfg, symbol, exchange)
    if ohlcv.empty:
        raise RuntimeError(f"No OHLCV data for {symbol}")
    funding = load_or_download_funding(cfg, symbol, exchange)

    anchor = None
    if cfg.features.use_cross_asset and symbol != cfg.features.anchor_symbol:
        anchor = load_or_download_ohlcv(cfg, cfg.features.anchor_symbol, exchange)

    X, atr = build_feature_matrix(ohlcv, funding, cfg, anchor_ohlcv=anchor)
    labels = triple_barrier_labels(ohlcv, atr, cfg)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common, "label"].astype(int)
    t1 = labels.loc[common, "t1"]
    w = labels.loc[common, "w"].astype(float)
    atr_pct = (atr / ohlcv["close"]).reindex(common)
    return {"ohlcv": ohlcv, "funding": funding, "X": X, "y": y, "t1": t1, "w": w, "atr_pct": atr_pct}


def train_symbol(cfg: Config, symbol: str, exchange=None) -> dict:
    ds = prepare_dataset(cfg, symbol, exchange)
    X, y, t1, w, ohlcv, funding, atr_pct = (
        ds["X"], ds["y"], ds["t1"], ds["w"], ds["ohlcv"], ds["funding"], ds["atr_pct"]
    )
    logger.info("{}: {} samples | {} features | class balance {}",
                symbol, len(X), X.shape[1], y.value_counts().to_dict())

    splits = purged_walk_forward_splits(
        X.index, t1, cfg.validation.n_splits, cfg.validation.embargo_bars
    )
    if not splits:
        raise RuntimeError(f"Could not build CV splits for {symbol} (too little data?)")

    meta_on = bool(cfg.model.meta_labeling)
    side = ymeta = None
    if meta_on:
        side = primary_side(ohlcv["close"], cfg.model.primary_window).reindex(X.index).fillna(0.0)
        if cfg.model.primary_kind == "reversion":
            side = -side  # fade the move (crypto mean-reverts intrabar/at short horizons)
        ymeta = meta_labels(side, y)
        logger.info("  meta-labeling ON: primary={}({}), act when P>{:.2f} | "
                    "meta-class balance {}", cfg.model.primary_kind, cfg.model.primary_window,
                    cfg.model.meta_threshold, ymeta.value_counts().to_dict())

    oos_signal = pd.Series(0, index=X.index, dtype=int)
    fold_sharpes: list[float] = []
    for fi, (tr, te) in enumerate(splits):
        te_idx = X.index[te]
        if meta_on:
            proba = _meta_proba(cfg, X.iloc[tr], ymeta.iloc[tr], w.iloc[tr].to_numpy(), X.iloc[te])
            act = proba > float(cfg.model.meta_threshold)
            sig = (side.iloc[te].to_numpy() * act).astype(int)
        else:
            model = build_model(cfg)
            model.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr].to_numpy())
            sig = model.predict_signal(X.iloc[te])
        oos_signal.iloc[te] = sig
        bt = backtest_signal(
            ohlcv.loc[te_idx], pd.Series(sig, index=te_idx), atr_pct.loc[te_idx], cfg, funding
        )
        fold_sharpes.append(bt["metrics"]["sharpe"])
        logger.info("  fold {}: sharpe={:.2f} ret={:.1%}", fi + 1,
                    bt["metrics"]["sharpe"], bt["metrics"]["total_return"])

    # combined out-of-sample backtest over the union of test regions
    first_test = splits[0][1][0]
    oos_idx = X.index[first_test:]
    oos_bt = backtest_signal(
        ohlcv.loc[oos_idx], oos_signal.loc[oos_idx], atr_pct.loc[oos_idx], cfg, funding
    )

    # Honest baselines: the signal must beat buy&hold AND a random signal OOS, after costs.
    oos_close = ohlcv.loc[oos_idx, "close"]
    buyhold_return = float(oos_close.iloc[-1] / oos_close.iloc[0] - 1.0)
    rng = np.random.default_rng(42)
    rand_sharpes = []
    for _ in range(3):
        rand_sig = pd.Series(rng.choice([-1, 0, 1], size=len(oos_idx)), index=oos_idx)
        rb = backtest_signal(ohlcv.loc[oos_idx], rand_sig, atr_pct.loc[oos_idx], cfg, funding)
        rand_sharpes.append(rb["metrics"]["sharpe"])
    random_sharpe = float(np.mean(rand_sharpes))

    # Deflated Sharpe Ratio: credit the OOS Sharpe only after accounting for the
    # number of CV trials (anti multiple-testing / anti-overfitting).
    bpy = infer_bars_per_year(oos_idx)
    per_bar_trials = [s / np.sqrt(bpy) for s in fold_sharpes if np.isfinite(s)]
    dsr = deflated_sharpe_ratio(oos_bt["results"]["net_ret"], per_bar_trials)

    # OOS selectivity & precision (the meta-labeling payoff: trade fewer, better)
    oos_sig = oos_signal.loc[oos_idx]
    pct_traded = float((oos_sig != 0).mean())
    regimes = regime_breakdown(ohlcv.loc[oos_idx], oos_sig, atr_pct.loc[oos_idx], cfg, funding)

    # final model trained on all available data
    path = model_path(cfg, symbol)
    if meta_on:
        import joblib
        import lightgbm as lgb

        params = dict(cfg.model.params)
        params.pop("objective", None)
        params.pop("num_class", None)
        clf = lgb.LGBMClassifier(**params, objective="binary", random_state=0, verbose=-1)
        clf.fit(X, ymeta, sample_weight=w.to_numpy())
        joblib.dump(
            {"type": "meta", "clf": clf, "features": list(X.columns),
             "primary_window": int(cfg.model.primary_window),
             "meta_threshold": float(cfg.model.meta_threshold)},
            path,
        )
    else:
        final = build_model(cfg)
        final.fit(X, y, sample_weight=w.to_numpy())
        final.save(path)

    meta = {
        "symbol": symbol,
        "exchange": cfg.exchange.id,
        "timeframe": cfg.universe.timeframe,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
        "features": list(X.columns),
        "meta_labeling": meta_on,
        "class_balance": {int(k): int(v) for k, v in y.value_counts().items()},
        "fold_sharpes": [round(s, 3) for s in fold_sharpes],
        "oos_metrics": {k: round(float(v), 4) for k, v in oos_bt["metrics"].items()},
        "oos_pct_traded": round(pct_traded, 4),
        "oos_regimes": regimes,
        "baselines": {
            "buyhold_return": round(buyhold_return, 4),
            "random_sharpe": round(random_sharpe, 3),
        },
        "deflated_sharpe": round(float(dsr), 4),
    }
    with open(str(path).replace(".pkl", ".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(
        "{} OOS: sharpe={:.2f} ret={:.1%} maxDD={:.1%} hit={:.1%} -> {}",
        symbol,
        oos_bt["metrics"]["sharpe"],
        oos_bt["metrics"]["total_return"],
        oos_bt["metrics"]["max_drawdown"],
        oos_bt["metrics"].get("hit_rate", float("nan")),
        path.name,
    )
    logger.info(
        "{} edge check: sharpe={:.2f} (random={:.2f}) PSR={:.2f} DSR={:.2f} | ret={:.1%} (buy&hold={:.1%})",
        symbol, oos_bt["metrics"]["sharpe"], random_sharpe,
        oos_bt["metrics"].get("psr", float("nan")), dsr,
        oos_bt["metrics"]["total_return"], buyhold_return,
    )
    logger.info(
        "{} selectivity: traded {:.0%} of OOS bars | hit-rate by regime: "
        "lowvol={:.0%}/{} hivol={:.0%}/{} up={:.0%}/{} dn={:.0%}/{}",
        symbol, pct_traded,
        regimes["low_vol"]["hit_rate"], regimes["low_vol"]["n"],
        regimes["high_vol"]["hit_rate"], regimes["high_vol"]["n"],
        regimes["trend_up"]["hit_rate"], regimes["trend_up"]["n"],
        regimes["trend_dn"]["hit_rate"], regimes["trend_dn"]["n"],
    )
    return {"meta": meta, "oos": oos_bt}


def train_all(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    exchange = make_data_exchange(cfg)
    out = {}
    for symbol in cfg.universe.symbols:
        try:
            out[symbol] = train_symbol(cfg, symbol, exchange)
        except Exception as exc:  # noqa: BLE001
            logger.error("Training failed for {}: {}", symbol, exc)
    return out
