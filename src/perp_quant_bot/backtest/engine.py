"""Vectorized, cost-aware backtester for a discrete {-1,0,1} signal.

This is a returns-level simulation (not an order-book matching engine):

* the signal at ``close[t]`` is acted on for the NEXT bar (no lookahead);
* position size comes from the RiskManager (vol targeting);
* costs = exchange fee + slippage applied to turnover;
* funding is charged/credited on the held position at funding timestamps.

It answers the only question that matters early: does the edge survive costs?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_conf import setup_logging
from ..risk import RiskManager
from .metrics import performance_summary

logger = setup_logging()


def _extract_trade_returns(pos_used: np.ndarray, net_ret: np.ndarray) -> list[float]:
    """Compound net returns within each maximal run of constant nonzero sign."""
    trades: list[float] = []
    sign = np.sign(pos_used)
    i, n = 0, len(sign)
    while i < n:
        if sign[i] == 0:
            i += 1
            continue
        j = i
        cur = sign[i]
        comp = 1.0
        while j < n and sign[j] == cur:
            comp *= 1.0 + net_ret[j]
            j += 1
        trades.append(comp - 1.0)
        i = j
    return trades


def backtest_signal(
    ohlcv: pd.DataFrame,
    signal: pd.Series,
    atr_pct: pd.Series,
    cfg: Config,
    funding: pd.DataFrame | None = None,
) -> dict:
    idx = ohlcv.index
    close = ohlcv["close"]
    signal = signal.reindex(idx).fillna(0.0)
    atr_pct = atr_pct.reindex(idx)

    rm = RiskManager(cfg.risk)
    size_frac = rm.position_fraction(atr_pct).reindex(idx).fillna(0.0)

    # decide at close[t]; the position becomes effective on the NEXT bar.
    pos_target = (signal * size_frac).astype(float)
    pos_used = pos_target.shift(1).fillna(0.0)

    if getattr(cfg.backtest, "fill", "next_open") == "next_open":
        # realistic: filled at the next bar's OPEN, earn that bar's open->open return.
        # Decision at close[t-1] -> fill at open[t] -> earn open[t]->open[t+1].
        open_ = ohlcv["open"]
        bar_ret = (open_.shift(-1) / open_ - 1.0).fillna(0.0)
    else:
        # close-to-close approximation (decision and fill at the same close[t-1]).
        bar_ret = close.pct_change().fillna(0.0)
    gross = pos_used * bar_ret

    turnover = (pos_used - pos_used.shift(1).fillna(0.0)).abs()
    cost_rate = cfg.backtest.fee_rate + cfg.backtest.slippage_bps / 1e4
    costs = turnover * cost_rate

    funding_term = pd.Series(0.0, index=idx)
    if cfg.backtest.apply_funding and funding is not None and "funding_rate" in getattr(funding, "columns", []):
        fr = funding["funding_rate"].reindex(idx).fillna(0.0)
        # longs pay positive funding -> reduces return of a long position
        funding_term = pos_used * fr

    net = gross - costs - funding_term
    equity = cfg.backtest.initial_capital * (1.0 + net).cumprod()

    trade_returns = _extract_trade_returns(pos_used.to_numpy(), net.to_numpy())
    summary = performance_summary(net, equity, trade_returns)

    results = pd.DataFrame(
        {
            "close": close,
            "signal": signal,
            "position": pos_used,
            "bar_ret": bar_ret,
            "cost": costs,
            "funding": funding_term,
            "net_ret": net,
            "equity": equity,
        }
    )
    logger.info(
        "Backtest: total={:.1%} sharpe={:.2f} maxDD={:.1%} trades={}",
        summary["total_return"], summary["sharpe"], summary["max_drawdown"],
        summary.get("n_trades", 0),
    )
    return {"results": results, "metrics": summary, "equity": equity}
