"""Deterministic leakage / lookahead tests — the real guarantee of honest accuracy.

These do not need any API key. They assert the invariants that, if broken, make a
backtest lie:
  * features are CAUSAL (a feature at bar t never changes when future bars change),
  * the purged walk-forward never lets a training label reach into the test window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from perp_quant_bot.backtest.metrics import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from perp_quant_bot.config import load_config
from perp_quant_bot.features import build_feature_matrix
from perp_quant_bot.features.cross_asset import anchor_features
from perp_quant_bot.features.technical import technical_features
from perp_quant_bot.labeling import triple_barrier_labels
from perp_quant_bot.models import LightGBMModel
from perp_quant_bot.validation import purged_walk_forward_splits


def make_synthetic_ohlcv(n: int = 2000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    ret = rng.normal(0, 0.005, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.0025, n)) * close
    high = close + spread
    low = close - spread
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def test_features_are_causal():
    """A feature at bar t must be identical whether computed on the full series or
    on the series truncated at t (i.e. it cannot depend on t+1..)."""
    cfg = load_config()
    ohlcv = make_synthetic_ohlcv()
    full = technical_features(ohlcv, cfg)

    for t in (300, 700, 1300):
        truncated = technical_features(ohlcv.iloc[: t + 1], cfg)
        a = full.iloc[t].to_numpy(dtype=float)
        b = truncated.iloc[t].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=1e-8, equal_nan=True), (
            f"feature row {t} changed when future bars were added -> lookahead leak"
        )


def test_purge_invariant():
    """In every walk-forward split: train is strictly before test, and no training
    label's end-time t1 reaches into the test window."""
    cfg = load_config()
    ohlcv = make_synthetic_ohlcv()
    X, atr = build_feature_matrix(ohlcv, None, cfg)
    labels = triple_barrier_labels(ohlcv, atr, cfg)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    t1 = labels.loc[common, "t1"]

    splits = purged_walk_forward_splits(
        X.index, t1, n_splits=4, embargo_bars=cfg.labeling.horizon_bars
    )
    assert len(splits) >= 1

    times_i8 = X.index.as_unit("ns").asi8
    t1_i8 = pd.DatetimeIndex(pd.to_datetime(t1.to_numpy(), utc=True)).as_unit("ns").asi8

    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()  # train strictly precedes test
        test_start_ns = times_i8[test_idx[0]]
        # every training label must END before the test window starts
        assert bool((t1_i8[train_idx] < test_start_ns).all()), "purge failed: label leaks into test"


def test_cross_asset_features_are_causal():
    """Anchor (BTC) features at bar t must not depend on the anchor's future bars."""
    cfg = load_config()
    ohlcv = make_synthetic_ohlcv()
    anchor = make_synthetic_ohlcv(seed=99)  # a different series acting as the anchor
    full = anchor_features(anchor, ohlcv.index, cfg)
    for t in (300, 900):
        trunc = anchor_features(anchor.iloc[: t + 1], ohlcv.index[: t + 1], cfg)
        a = full.iloc[t].to_numpy(dtype=float)
        b = trunc.iloc[t].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=1e-8, equal_nan=True), "anchor feature leaked the future"


def test_label_weights_present_and_normalized():
    cfg = load_config()
    ohlcv = make_synthetic_ohlcv()
    _, atr = build_feature_matrix(ohlcv, None, cfg)
    labels = triple_barrier_labels(ohlcv, atr, cfg)
    assert "w" in labels.columns
    w = labels["w"].to_numpy(dtype=float)
    assert np.all(w > 0), "sample weights must be positive"
    assert abs(float(np.mean(w)) - 1.0) < 1e-6, "sample weights should average to 1"


def test_model_ensemble_and_calibration():
    """Seed-ensemble + calibrated probabilities are well-formed (sum to 1, valid signals)."""
    cfg = load_config()
    ohlcv = make_synthetic_ohlcv(n=2200)
    X, atr = build_feature_matrix(ohlcv, None, cfg)
    labels = triple_barrier_labels(ohlcv, atr, cfg)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common, "label"].astype(int)

    model = LightGBMModel(params=cfg.model.params, threshold=0.4, n_seeds=3, calibrate=True)
    cut = int(len(X) * 0.8)
    model.fit(X.iloc[:cut], y.iloc[:cut])
    proba = model.predict_proba(X.iloc[cut:])
    assert proba.shape[1] == len(model.classes_)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(model.predict_signal(X.iloc[cut:]))).issubset({-1, 0, 1})


def test_psr_and_dsr_in_unit_interval():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 1500))
    psr = probabilistic_sharpe_ratio(r)
    dsr = deflated_sharpe_ratio(r, [0.5, 0.7, 0.6, 0.4, 0.8])
    assert 0.0 <= psr <= 1.0
    assert 0.0 <= dsr <= 1.0
    # DSR is stricter than PSR (benchmark raised for multiple testing)
    assert dsr <= psr + 1e-9


def test_labels_have_no_intrabar_plus_one_bias():
    """Labels stay in {-1,0,1}; ambiguous intrabar double-touches are neutral (0),
    so symmetric barriers should not be dominated by +1."""
    cfg = load_config()
    ohlcv = make_synthetic_ohlcv()
    _, atr = build_feature_matrix(ohlcv, None, cfg)
    labels = triple_barrier_labels(ohlcv, atr, cfg)
    vals = labels["label"].to_numpy()
    assert set(np.unique(vals)).issubset({-1, 0, 1})
    # with symmetric pt==sl, +1 should not massively outnumber -1 (sanity, generous)
    n_pos = int((vals == 1).sum())
    n_neg = int((vals == -1).sum())
    if n_pos + n_neg > 50:
        ratio = n_pos / max(n_neg, 1)
        assert 0.5 < ratio < 2.0, f"directional label imbalance suspicious: +1/-1={ratio:.2f}"
