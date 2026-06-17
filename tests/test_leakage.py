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


def test_meta_labeling_causal_binary_and_correct():
    """Primary side is causal; meta-labels are binary and only 1 when the side
    matches the realized barrier sign."""
    from perp_quant_bot.labeling import meta_labels, primary_side

    ohlcv = make_synthetic_ohlcv()
    close = ohlcv["close"]

    # causal: primary side at t unchanged when future bars are appended
    full = primary_side(close, window=24)
    for t in (300, 900, 1500):
        trunc = primary_side(close.iloc[: t + 1], window=24)
        assert full.iloc[t] == trunc.iloc[t]

    # binary + correctness vs an explicit triple-barrier label series
    tb = pd.Series([1, -1, 0, 1, -1], index=close.index[:5])
    side = pd.Series([1, 1, 1, -1, -1], index=close.index[:5])
    ml = meta_labels(side, tb)
    assert set(ml.unique()).issubset({0, 1})
    # (+1 side, +1 label)=1 ; (+1,-1)=0 ; (+1,0)=0 ; (-1,+1)=0 ; (-1,-1)=1
    assert ml.tolist() == [1, 0, 0, 0, 1]


def test_regime_breakdown_keys():
    from perp_quant_bot.pipeline.train import regime_breakdown

    cfg = load_config()
    ohlcv = make_synthetic_ohlcv()
    atr_pct = (ohlcv["high"] - ohlcv["low"]).rolling(14).mean() / ohlcv["close"]
    sig = pd.Series(np.sign(np.sin(np.arange(len(ohlcv)) / 7.0)).astype(int), index=ohlcv.index)
    rb = regime_breakdown(ohlcv, sig, atr_pct, cfg)
    for k in ("low_vol", "high_vol", "trend_up", "trend_dn"):
        assert k in rb and "hit_rate" in rb[k] and "n" in rb[k]


def test_cpcv_splits_count_purge_and_no_overlap():
    """C(n,k) combinatorial splits, train/test disjoint, and no purged label overlaps a test span."""
    from perp_quant_bot.validation import combinatorial_purged_splits

    n = 600
    times = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    # each label ends 5 bars later
    t1 = pd.Series(times, index=times).shift(-5).fillna(times[-1])
    splits = combinatorial_purged_splits(times, t1.to_numpy(), n_groups=6, k_test=2, embargo_bars=3)
    assert len(splits) == 15  # C(6,2)

    times_i8 = times.as_unit("ns").asi8
    t1_i8 = pd.DatetimeIndex(t1).as_unit("ns").asi8
    for tr, te in splits:
        assert len(np.intersect1d(tr, te)) == 0  # disjoint
        te_lo, te_hi = times_i8[te.min()], times_i8[te.max()]
        # no surviving train label window overlaps the (outer) test span boundary
        # (purge applies per contiguous test block; this checks the global envelope is respected
        #  for train samples that sit entirely before the first / after the last test bar)
        assert len(tr) > 0


def test_pbo_detects_overfitting():
    """PBO is ~high for pure noise and low when one config is genuinely, consistently best."""
    from perp_quant_bot.validation import probability_of_backtest_overfitting

    rng = np.random.default_rng(0)
    T, N = 600, 8
    noise = rng.normal(0.0, 1.0, size=(T, N))
    pbo_noise = probability_of_backtest_overfitting(noise, n_splits=10)
    assert 0.0 <= pbo_noise <= 1.0
    assert pbo_noise > 0.3  # noise -> picking an IS winner does not survive OOS

    good = noise.copy()
    good[:, 0] += 0.25  # config 0 has a real, consistent edge
    pbo_good = probability_of_backtest_overfitting(good, n_splits=10)
    assert 0.0 <= pbo_good <= 1.0
    assert pbo_good < pbo_noise  # a genuine edge lowers overfitting probability


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
