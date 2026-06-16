"""Offline end-to-end smoke test on synthetic OHLCV (no network, no keys)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from perp_quant_bot.backtest import backtest_signal
from perp_quant_bot.config import load_config
from perp_quant_bot.execution import PaperBroker
from perp_quant_bot.execution.broker import Order
from perp_quant_bot.features import build_feature_matrix
from perp_quant_bot.labeling import triple_barrier_labels
from perp_quant_bot.models import LightGBMModel
from perp_quant_bot.validation import purged_walk_forward_splits


def make_synthetic_ohlcv(n: int = 2500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    # random walk with mild autocorrelation so labels aren't pure noise
    shocks = rng.normal(0, 0.005, n)
    drift = pd.Series(shocks).rolling(5).mean().fillna(0).to_numpy() * 0.3
    ret = shocks + drift
    close = 100.0 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.0025, n)) * close
    high = close + spread
    low = close - spread
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def _dataset():
    cfg = load_config()
    ohlcv = make_synthetic_ohlcv()
    X, atr = build_feature_matrix(ohlcv, None, cfg)
    labels = triple_barrier_labels(ohlcv, atr, cfg)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common, "label"].astype(int)
    t1 = labels.loc[common, "t1"]
    atr_pct = (atr / ohlcv["close"]).reindex(common)
    return cfg, ohlcv, X, y, t1, atr_pct


def test_features_and_labels_align():
    _cfg, _ohlcv, X, y, t1, _atr = _dataset()
    assert len(X) > 500
    assert len(X) == len(y) == len(t1)
    assert set(np.unique(y)).issubset({-1, 0, 1})
    assert not X.isna().any().any()


def test_walk_forward_train_predict_backtest():
    cfg, ohlcv, X, y, t1, atr_pct = _dataset()
    splits = purged_walk_forward_splits(X.index, t1, n_splits=3, embargo_bars=cfg.labeling.horizon_bars)
    assert len(splits) >= 1

    tr, te = splits[0]
    # no overlap between train and test positions
    assert len(set(tr).intersection(set(te))) == 0

    model = LightGBMModel(params=cfg.model.params, threshold=cfg.model.prob_threshold)
    model.fit(X.iloc[tr], y.iloc[tr])
    sig = model.predict_signal(X.iloc[te])
    assert set(np.unique(sig)).issubset({-1, 0, 1})

    te_idx = X.index[te]
    bt = backtest_signal(ohlcv.loc[te_idx], pd.Series(sig, index=te_idx), atr_pct.loc[te_idx], cfg)
    assert len(bt["equity"]) == len(te_idx)
    for key in ("sharpe", "psr", "max_drawdown", "total_return"):
        assert key in bt["metrics"]


def test_paper_broker_fills():
    b = PaperBroker(initial_cash=10_000.0, fee_rate=0.0)
    b.update_price("BTC/USDT:USDT", 100.0)
    b.create_order(Order(symbol="BTC/USDT:USDT", side="buy", amount=1.0))
    assert b.get_position("BTC/USDT:USDT") == 1.0
    # equity ~ unchanged at same price (no fees here)
    assert abs(b.get_equity() - 10_000.0) < 1e-6
    b.update_price("BTC/USDT:USDT", 110.0)
    assert abs(b.get_equity() - 10_010.0) < 1e-6
