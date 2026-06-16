"""Training pipeline: data -> features -> labels -> purged walk-forward -> save."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..backtest import backtest_signal
from ..config import Config, load_config
from ..data import load_or_download_funding, load_or_download_ohlcv
from ..data.exchange import make_exchange
from ..features import build_feature_matrix
from ..labeling import triple_barrier_labels
from ..logging_conf import setup_logging
from ..models import LightGBMModel
from ..validation import purged_walk_forward_splits

logger = setup_logging()


def build_model(cfg: Config) -> LightGBMModel:
    return LightGBMModel(params=cfg.model.params, threshold=cfg.model.prob_threshold)


def _sanitize(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-")


def model_path(cfg: Config, symbol: str):
    return cfg.models_dir() / f"{cfg.exchange.id}_{_sanitize(symbol)}_{cfg.universe.timeframe}.pkl"


def prepare_dataset(cfg: Config, symbol: str, exchange=None):
    ohlcv = load_or_download_ohlcv(cfg, symbol, exchange)
    if ohlcv.empty:
        raise RuntimeError(f"No OHLCV data for {symbol}")
    funding = load_or_download_funding(cfg, symbol, exchange)
    X, atr = build_feature_matrix(ohlcv, funding, cfg)
    labels = triple_barrier_labels(ohlcv, atr, cfg)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common, "label"].astype(int)
    t1 = labels.loc[common, "t1"]
    atr_pct = (atr / ohlcv["close"]).reindex(common)
    return {"ohlcv": ohlcv, "funding": funding, "X": X, "y": y, "t1": t1, "atr_pct": atr_pct}


def train_symbol(cfg: Config, symbol: str, exchange=None) -> dict:
    ds = prepare_dataset(cfg, symbol, exchange)
    X, y, t1, ohlcv, funding, atr_pct = (
        ds["X"], ds["y"], ds["t1"], ds["ohlcv"], ds["funding"], ds["atr_pct"]
    )
    logger.info("{}: {} samples | class balance {}", symbol, len(X), y.value_counts().to_dict())

    splits = purged_walk_forward_splits(
        X.index, t1, cfg.validation.n_splits, cfg.validation.embargo_bars
    )
    if not splits:
        raise RuntimeError(f"Could not build CV splits for {symbol} (too little data?)")

    oos_signal = pd.Series(0, index=X.index, dtype=int)
    fold_sharpes: list[float] = []
    for fi, (tr, te) in enumerate(splits):
        model = build_model(cfg)
        model.fit(X.iloc[tr], y.iloc[tr])
        sig = model.predict_signal(X.iloc[te])
        oos_signal.iloc[te] = sig
        te_idx = X.index[te]
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

    # final model trained on all available data
    final = build_model(cfg)
    final.fit(X, y)
    path = model_path(cfg, symbol)
    final.save(path)

    meta = {
        "symbol": symbol,
        "exchange": cfg.exchange.id,
        "timeframe": cfg.universe.timeframe,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
        "features": list(X.columns),
        "class_balance": {int(k): int(v) for k, v in y.value_counts().items()},
        "fold_sharpes": [round(s, 3) for s in fold_sharpes],
        "oos_metrics": {k: round(float(v), 4) for k, v in oos_bt["metrics"].items()},
        "baselines": {
            "buyhold_return": round(buyhold_return, 4),
            "random_sharpe": round(random_sharpe, 3),
        },
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
        "{} edge check: strat_sharpe={:.2f} vs random={:.2f} | strat_ret={:.1%} vs buy&hold={:.1%}",
        symbol, oos_bt["metrics"]["sharpe"], random_sharpe,
        oos_bt["metrics"]["total_return"], buyhold_return,
    )
    return {"meta": meta, "oos": oos_bt}


def train_all(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    exchange = make_exchange(cfg)
    out = {}
    for symbol in cfg.universe.symbols:
        try:
            out[symbol] = train_symbol(cfg, symbol, exchange)
        except Exception as exc:  # noqa: BLE001
            logger.error("Training failed for {}: {}", symbol, exc)
    return out
