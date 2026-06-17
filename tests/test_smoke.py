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


def test_cross_sectional_backtest_is_market_neutral():
    from perp_quant_bot.strategies.cross_sectional import cross_sectional_backtest

    cfg = load_config()
    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-01", periods=600, freq="1D", tz="UTC")
    cols = {
        f"SYM{i}/USDT:USDT": 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.03, len(idx))))
        for i in range(10)
    }
    close = pd.DataFrame(cols, index=idx)
    res = cross_sectional_backtest(close, cfg, lookback=30, top_frac=0.3, min_names=4)
    m = res["metrics"]
    for key in ("sharpe", "psr", "deflated_sharpe", "max_drawdown", "n_symbols"):
        assert key in m
    assert m["n_symbols"] == 10
    assert len(res["equity"]) > 0
    # market-neutral: net weight each bar must be ~0
    assert float(res["weights"].sum(axis=1).abs().max()) < 1e-9


def test_funding_carry_backtest_is_market_neutral():
    from perp_quant_bot.strategies.funding_carry import funding_carry_backtest

    cfg = load_config()
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=400, freq="8h", tz="UTC")
    cols = [f"S{i}/USDT:USDT" for i in range(8)]
    close = pd.DataFrame(
        {c: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx)))) for c in cols}, index=idx
    )
    funding = pd.DataFrame(
        {c: rng.normal(0.0001, 0.0005, len(idx)) for c in cols}, index=idx
    )
    res = funding_carry_backtest(close, funding, cfg, top_frac=0.3, min_names=4)
    m = res["metrics"]
    for k in ("sharpe", "psr", "deflated_sharpe", "n_symbols"):
        assert k in m
    assert m["n_symbols"] == 8
    assert float(res["weights"].sum(axis=1).abs().max()) < 1e-9  # market-neutral


def test_basis_carry_backtest_runs():
    from perp_quant_bot.strategies.basis_carry import basis_carry_backtest

    cfg = load_config()
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=400, freq="8h", tz="UTC")
    cols = [f"S{i}/USDT:USDT" for i in range(6)]
    base = {c: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx)))) for c in cols}
    perp = pd.DataFrame(base, index=idx)
    # spot tracks perp closely (small basis noise)
    spot = pd.DataFrame(
        {c: base[c] * (1 + rng.normal(0, 0.0005, len(idx))) for c in cols}, index=idx
    )
    # funding mostly positive
    funding = pd.DataFrame(
        {c: np.abs(rng.normal(0.0002, 0.0003, len(idx))) for c in cols}, index=idx
    )
    res = basis_carry_backtest(perp, spot, funding, cfg)
    m = res["metrics"]
    for k in ("sharpe", "psr", "deflated_sharpe", "n_symbols", "pct_engaged"):
        assert k in m
    assert m["n_symbols"] == 6
    assert m["pct_engaged"] > 0.5  # funding mostly positive -> engaged most bars


def test_pairs_stat_arb_backtest():
    """A cointegrated pair (mean-reverting spread) is captured gross by the fader."""
    from perp_quant_bot.strategies.pairs import pairs_stat_arb_backtest

    cfg = load_config()
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-01", periods=600, freq="1D", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.01, len(idx)))  # shared trend
    p1 = pd.Series(100.0 * np.exp(common + rng.normal(0, 0.005, len(idx))), index=idx)
    # p2 tracks p1 with a STATIONARY (mean-reverting) spread -> cointegrated
    ms = np.zeros(len(idx))
    for t in range(1, len(idx)):
        ms[t] = 0.9 * ms[t - 1] + rng.normal(0, 0.01)  # AR(1) mean-reverting
    p2 = pd.Series(100.0 * np.exp(common + ms), index=idx)
    prices = pd.DataFrame({"P1USDT": p1, "P2USDT": p2})
    r = pairs_stat_arb_backtest(prices, cfg, lookback=40, fee_rate=0.0, slippage_bps=0.0)
    m = r["metrics"]
    assert m["n_pairs"] == 1
    assert m["gross_total_return"] > 0  # mean-reverting spread, zero cost -> captured


def test_cross_exchange_funding_backtest():
    """A persistent positive A-minus-B spread is collected (gross positive, delta-neutral)."""
    from perp_quant_bot.strategies.cross_exchange import cross_exchange_funding_backtest

    cfg = load_config()
    rng = np.random.default_rng(9)
    idx = pd.date_range("2024-01-01", periods=300, freq="8h", tz="UTC")
    cols = [f"S{i}/USDT:USDT" for i in range(4)]
    f_b = pd.DataFrame({c: rng.normal(0, 0.0001, len(idx)) for c in cols}, index=idx)
    f_a = f_b + 0.0002 + pd.DataFrame(rng.normal(0, 2e-5, (len(idx), len(cols))), index=idx, columns=cols)
    r = cross_exchange_funding_backtest(f_a, f_b, cfg, fee_rate=0.0, slippage_bps=0.0)
    m = r["metrics"]
    assert m["n_symbols"] == 4
    assert m["pct_engaged"] > 0.8  # spread clears the threshold almost always
    assert m["gross_total_return"] > 0  # persistent spread, zero cost -> collected


def test_leverage_report_scales_and_flags_liquidation():
    from perp_quant_bot.strategies.basis_carry import leverage_report

    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=400, freq="1D", tz="UTC")
    net = pd.Series(0.0005 + rng.normal(0, 0.0002, 400), index=idx)  # small positive + noise
    by = {r["leverage"]: r for r in leverage_report(net, levels=(1, 2, 5))}
    assert abs(by[1]["sharpe"] - by[5]["sharpe"]) < 1e-6  # Sharpe is scale-invariant
    assert by[5]["ann_return"] > by[2]["ann_return"] > by[1]["ann_return"]  # return scales up
    assert not by[5]["liquidation_risk"]

    blown = net.copy()
    blown.iloc[10] = -0.5  # a -50% bar at 3x -> -150% -> account wiped
    assert leverage_report(blown, levels=(3,))[0]["liquidation_risk"]


def test_basis_carry_top_k_concentrates():
    """top_k holds only the K richest-funding names each bar (leak-safe ranking)."""
    from perp_quant_bot.strategies.basis_carry import basis_carry_backtest

    cfg = load_config()
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=200, freq="1D", tz="UTC")
    cols = [f"S{i}/USDT:USDT" for i in range(6)]
    base = {c: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for c in cols}
    perp = pd.DataFrame(base, index=idx)
    spot = pd.DataFrame({c: base[c] * (1 + rng.normal(0, 0.0003, len(idx))) for c in cols}, index=idx)
    # distinct, constant positive funding: S0 richest ... S5 thinnest
    funding = pd.DataFrame(
        {c: np.full(len(idx), 0.0010 - 0.0001 * i) for i, c in enumerate(cols)}, index=idx
    )
    res = basis_carry_backtest(perp, spot, funding, cfg, top_k=2)
    w = res["weights"]
    assert int((w > 0).sum(axis=1).max()) <= 2  # never more than K names
    held = w.iloc[-1] > 0
    assert held["S0/USDT:USDT"] and held["S1/USDT:USDT"]  # the two richest


def test_carry_reconcile_idempotent_and_exits():
    """Reconcile = minimal orders to target; no-op at target; closes dropped symbols."""
    from perp_quant_bot.execution.carry_executor import CarryLeg, CarryPlan, reconcile_orders

    leg = CarryLeg("BTC/USDT:USDT", "BTC/USDT", 0.0001, 100.0, 100.0, 100.0, 1.0, 1.0, 50.0, 0.03)
    plan = CarryPlan("bybit", 150.0, 2.0, 0.0002, [leg])

    # flat -> enter both legs (buy spot, short perp)
    o = reconcile_orders(plan, {}, {})
    spot = [x for x in o if x.market == "spot"][0]
    perp = [x for x in o if x.market == "perp"][0]
    assert spot.side == "buy" and abs(spot.amount - 1.0) < 1e-9
    assert perp.side == "sell" and abs(perp.amount - 1.0) < 1e-9

    # already at target -> no orders
    assert reconcile_orders(plan, {"BTC": 1.0}, {"BTC/USDT:USDT": -1.0}) == []

    # a dropped symbol (funding flipped) -> close both legs
    o2 = reconcile_orders(plan, {"BTC": 1.0, "ETH": 2.0}, {"BTC/USDT:USDT": -1.0, "ETH/USDT:USDT": -2.0})
    exits = [x for x in o2 if x.reason == "exit"]
    assert any(x.market == "perp" and x.symbol == "ETH/USDT:USDT" and x.side == "buy" for x in exits)
    assert any(x.market == "spot" and x.symbol == "ETH/USDT" and x.side == "sell" for x in exits)


def test_carry_plan_math_and_live_guard():
    """CarryPlan accounting is correct; execute_live refuses without confirm."""
    import pytest

    from perp_quant_bot.execution.carry_executor import CarryLeg, CarryPlan, execute_live, render_plan

    # one leg: $100 notional, funding 0.0001/8h -> 0.03/day; leverage 2 -> margin $50
    leg = CarryLeg(
        perp_symbol="BTC/USDT:USDT", spot_symbol="BTC/USDT", funding_rate=0.0001,
        spot_price=100.0, perp_price=100.0, leg_notional=100.0,
        spot_amount=1.0, perp_amount=1.0, perp_margin=50.0,
        funding_per_day=100.0 * 0.0001 * 3.0,
    )
    plan = CarryPlan(venue="bybit", capital=150.0, leverage=2.0, fee_rate=0.0002, legs=[leg])
    assert abs(plan.deployed - 150.0) < 1e-9  # notional 100 + margin 50
    assert abs(plan.funding_per_day - 0.03) < 1e-9
    assert plan.funding_apr > 0
    # round trip: 2 legs * notional * fee * 2 = 2*100*0.0002*2 = 0.08
    assert abs(plan.entry_fees - 0.08) < 1e-9
    assert plan.payback_days > 0
    assert "LONG" in render_plan(plan) and "SHORT" in render_plan(plan)

    # empty plan -> sit out message, no crash
    empty = CarryPlan(venue="bybit", capital=100.0, leverage=2.0, fee_rate=0.0002, legs=[])
    assert "sit out" in render_plan(empty)

    # the only money-spending path must refuse without explicit confirmation
    with pytest.raises(RuntimeError):
        execute_live(plan, confirm=False)


def test_paper_broker_fills():
    b = PaperBroker(initial_cash=10_000.0, fee_rate=0.0)
    b.update_price("BTC/USDT:USDT", 100.0)
    b.create_order(Order(symbol="BTC/USDT:USDT", side="buy", amount=1.0))
    assert b.get_position("BTC/USDT:USDT") == 1.0
    # equity ~ unchanged at same price (no fees here)
    assert abs(b.get_equity() - 10_000.0) < 1e-6
    b.update_price("BTC/USDT:USDT", 110.0)
    assert abs(b.get_equity() - 10_010.0) < 1e-6
