"""Cross-sectional funding-carry, market-neutral. Backtestable on real funding history.

Perp funding is a real recurring cashflow: when funding > 0, longs pay shorts. So we
go SHORT the highest-funding perps (collect funding) and LONG the most-negative-funding
perps (collect funding), with net-zero gross-1 weights. Market-neutral construction
hedges price beta; the harvested edge is the cross-sectional funding spread, minus the
price drift of the legs and trading costs.

Decisions at bar t use funding known at t; PnL (price move + funding settled) is
realized over t -> t+1. Leak-free.
"""
from __future__ import annotations

import time

import ccxt
import numpy as np
import pandas as pd

from ..backtest.metrics import deflated_sharpe_ratio, performance_summary
from ..config import Config, load_config
from ..data.exchange import make_exchange
from ..data.ohlcv import _sanitize, download_ohlcv
from ..logging_conf import setup_logging
from .cross_sectional import DEFAULT_UNIVERSE

logger = setup_logging()


def current_funding(venue: str = "bybit", symbols: list[str] | None = None) -> pd.DataFrame:
    """Live funding snapshot: positive funding => carry = SHORT perp + LONG spot.

    Read-only, no keys. Shows where the carry is right now on a given venue.
    """
    symbols = symbols or DEFAULT_UNIVERSE
    ex = getattr(ccxt, venue)(
        {"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}}
    )
    rows = []
    for s in symbols:
        try:
            fr = ex.fetch_funding_rate(s)
        except Exception:  # noqa: BLE001
            continue
        rate = fr.get("fundingRate")
        if rate is None:
            continue
        rate = float(rate)
        rows.append({
            "symbol": s,
            "funding_rate": rate,
            "annualized_pct": rate * 3 * 365 * 100.0,  # 8h funding -> rough APR
        })
    if not rows:
        return pd.DataFrame(columns=["symbol", "funding_rate", "annualized_pct"])
    return pd.DataFrame(rows).sort_values("funding_rate", ascending=False).reset_index(drop=True)


def funding_carry_backtest(
    close: pd.DataFrame,
    funding: pd.DataFrame,
    cfg: Config,
    top_frac: float = 0.30,
    min_names: int = 5,
) -> dict:
    """Pure backtest: aligned [time x symbol] close + funding -> results."""
    common = close.index.intersection(funding.index)
    close = close.loc[common]
    funding = funding.reindex(columns=close.columns).loc[common]
    rets = close.pct_change()

    n_avail = funding.notna().sum(axis=1)
    ranks = funding.rank(axis=1, pct=True)
    # LONG the lowest funding (collect), SHORT the highest funding (collect)
    long = (ranks <= top_frac).astype(float)
    short = (ranks >= 1.0 - top_frac).astype(float)
    long_w = long.div(long.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * 0.5
    short_w = short.div(short.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * 0.5
    w = (long_w - short_w).where(n_avail >= min_names, 0.0)

    w_used = w.shift(1).fillna(0.0)  # decide at t, hold over t -> t+1
    price_pnl = (w_used * rets).sum(axis=1)
    # perp holder pays funding when long & funding>0  => funding PnL = -w * funding
    funding_pnl = (-w_used * funding.fillna(0.0)).sum(axis=1)

    turnover = (w_used - w_used.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_rate = cfg.backtest.fee_rate + cfg.backtest.slippage_bps / 1e4
    net = price_pnl + funding_pnl - turnover * cost_rate

    active = w_used.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[active.idxmax():]
    equity = cfg.backtest.initial_capital * (1.0 + net).cumprod()

    metrics = performance_summary(net, equity)
    chunks = np.array_split(net.dropna().to_numpy(), 5)
    trials = [float(c.mean() / c.std()) for c in chunks if len(c) > 5 and c.std() > 0]
    metrics["deflated_sharpe"] = deflated_sharpe_ratio(net, trials)
    metrics["n_symbols"] = int(close.shape[1])
    metrics["funding_pnl_share"] = (
        float(funding_pnl.loc[net.index].sum() / net.sum()) if net.sum() != 0 else float("nan")
    )
    metrics["avg_turnover"] = float(turnover.loc[net.index].mean()) if len(net) else 0.0
    return {"metrics": metrics, "equity": equity, "weights": w, "net": net}


def _funding_series(ex, symbol: str, since_ms: int) -> pd.Series:
    """One paginated pull of funding-rate history -> Series indexed by 8h timestamp."""
    rows: list[dict] = []
    cursor = since_ms
    until = ex.milliseconds()
    for _ in range(12):  # safety cap
        try:
            batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        except Exception:  # noqa: BLE001
            break
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["timestamp"]
        if last is None or last + 1 <= cursor:
            break
        cursor = last + 1
        time.sleep(max(ex.rateLimit, 50) / 1000)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.DataFrame(
        {"ts": [r["timestamp"] for r in rows], "f": [r.get("fundingRate") for r in rows]}
    ).dropna()
    s = s.drop_duplicates("ts").sort_values("ts")
    idx = pd.to_datetime(s["ts"], unit="ms", utc=True).dt.floor("8h")
    return pd.Series(s["f"].to_numpy(dtype=float), index=idx)


def run_funding_carry(
    cfg: Config | None = None,
    symbols: list[str] | None = None,
    funding_venue: str = "mexc",
    top_frac: float = 0.30,
) -> dict:
    cfg = cfg or load_config()
    symbols = symbols or DEFAULT_UNIVERSE
    ex = make_exchange(cfg, sandbox=False, exchange_id=funding_venue)
    since = ex.parse8601(cfg.universe.since)

    closes: dict[str, pd.Series] = {}
    fundings: dict[str, pd.Series] = {}
    for s in symbols:
        try:
            df = download_ohlcv(ex, s, "8h", since)
            f = _funding_series(ex, s, since)
        except Exception as exc:  # noqa: BLE001
            logger.warning("funding-carry: skipping {} ({})", s, str(exc).splitlines()[0][:70])
            continue
        if not df.empty and len(df) > 200 and not f.empty and len(f) > 200:
            closes[s] = df["close"]
            fundings[s] = f
        else:
            logger.warning("funding-carry: skipping {} (insufficient close/funding)", s)

    if len(closes) < 5:
        raise RuntimeError(f"Only {len(closes)} symbols had funding+price; need >= 5")

    close = pd.DataFrame(closes).sort_index()
    # align close to 8h grid (open time) so it lines up with funding timestamps
    close.index = close.index.floor("8h")
    close = close[~close.index.duplicated(keep="last")]
    funding = pd.DataFrame(fundings).sort_index()

    res = funding_carry_backtest(close, funding, cfg, top_frac=top_frac)
    m = res["metrics"]
    logger.info(
        "funding-carry ({} names, {} bars @ {}): sharpe={:.2f} PSR={:.2f} DSR={:.2f} "
        "ret={:.1%} maxDD={:.1%} | funding share of pnl={:.0%}",
        m["n_symbols"], len(res["net"]), funding_venue, m["sharpe"], m["psr"],
        m["deflated_sharpe"], m["total_return"], m["max_drawdown"], m.get("funding_pnl_share", float("nan")),
    )
    return res
