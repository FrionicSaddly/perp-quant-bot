"""Statistical-arbitrage pairs (market-neutral mean reversion).

For correlated perps P1, P2 we form a beta-hedged spread and fade its deviations:
when the spread's trailing z-score is high we SHORT it (short P1 + beta*long P2),
when low we LONG it, exiting near the mean. Beta and z-score use only TRAILING data
(leak-free). A diversified book equal-weights many pairs.

Honest caveat: crypto cointegration is unstable (pairs de-cohere in trends), so this
is judged strictly on OOS / DSR. NOT modeled: borrow, exact per-leg margin.
"""
from __future__ import annotations

from itertools import combinations

import httpx
import numpy as np
import pandas as pd

from ..backtest.metrics import deflated_sharpe_ratio, infer_bars_per_year, performance_summary, sharpe
from ..config import Config, load_config
from ..logging_conf import setup_logging

logger = setup_logging()

PAIRS_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
    "XRPUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
]


def _pair_pnl(
    p1: pd.Series, p2: pd.Series, lookback: int, entry_z: float, exit_z: float
) -> tuple[pd.Series, pd.Series]:
    """Net-of-nothing per-bar return + turnover for one beta-hedged mean-reversion pair."""
    r1, r2 = p1.pct_change(), p2.pct_change()
    lp1, lp2 = np.log(p1), np.log(p2)
    # trailing hedge ratio beta = cov(r1,r2)/var(r2)
    cov = r1.rolling(lookback).cov(r2)
    var = r2.rolling(lookback).var()
    beta = (cov / var.replace(0.0, np.nan)).clip(-5, 5)
    spread = lp1 - beta * lp2
    z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std().replace(0.0, np.nan)

    z_prev = z.shift(1)  # decide on prior bar only
    pos = pd.Series(0.0, index=p1.index)
    cur = 0.0
    for t in range(len(pos)):
        zt = z_prev.iloc[t]
        if not np.isfinite(zt):
            pos.iloc[t] = 0.0
            cur = 0.0
            continue
        if cur == 0.0:
            if zt > entry_z:
                cur = -1.0  # spread rich -> short it
            elif zt < -entry_z:
                cur = 1.0
        elif abs(zt) < exit_z:
            cur = 0.0
        pos.iloc[t] = cur

    beta_prev = beta.shift(1).fillna(0.0)
    spread_ret = r1 - beta_prev * r2  # long-spread return
    pnl = (pos * spread_ret).fillna(0.0)
    gross_lev = (1.0 + beta_prev.abs()).replace(0.0, 1.0)  # notional per unit (P1 + |beta|*P2)
    turnover = (pos - pos.shift(1).fillna(0.0)).abs() * gross_lev
    return pnl, turnover


def pairs_stat_arb_backtest(
    prices: pd.DataFrame,
    cfg: Config,
    pairs: list[tuple[str, str]] | None = None,
    lookback: int = 90,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    fee_rate: float | None = None,
    slippage_bps: float | None = None,
) -> dict:
    cols = list(prices.columns)
    pairs = pairs or list(combinations(cols, 2))
    fr = cfg.backtest.fee_rate if fee_rate is None else fee_rate
    sl = cfg.backtest.slippage_bps if slippage_bps is None else slippage_bps
    cost_rate = fr + sl / 1e4

    pnls, turns = [], []
    for a, b in pairs:
        if a not in prices or b not in prices:
            continue
        pnl, turn = _pair_pnl(prices[a].dropna(), prices[b].dropna(), lookback, entry_z, exit_z)
        pnls.append(pnl)
        turns.append(turn)
    if not pnls:
        raise RuntimeError("no valid pairs")

    npairs = len(pnls)
    gross = pd.concat(pnls, axis=1).fillna(0.0).mean(axis=1)  # equal-weight pairs
    turnover = pd.concat(turns, axis=1).fillna(0.0).mean(axis=1)
    net = gross - turnover * cost_rate
    net = net.loc[(net != 0).idxmax():] if (net != 0).any() else net
    equity = cfg.backtest.initial_capital * (1.0 + net).cumprod()

    metrics = performance_summary(net, equity)
    chunks = np.array_split(net.dropna().to_numpy(), 5)
    trials = [float(c.mean() / c.std()) for c in chunks if len(c) > 5 and c.std() > 0]
    metrics["deflated_sharpe"] = deflated_sharpe_ratio(net, trials)
    metrics["n_pairs"] = npairs
    metrics["avg_turnover"] = float(turnover.loc[net.index].mean()) if len(net) else 0.0
    bpy = infer_bars_per_year(net.index)
    metrics["gross_sharpe"] = sharpe(gross.loc[net.index], bpy)
    metrics["gross_total_return"] = float((1.0 + gross.loc[net.index]).prod() - 1.0)
    return {"metrics": metrics, "equity": equity, "net": net}


def _load_prices(cfg: Config, symbols: list[str], interval: str = "1d") -> pd.DataFrame:
    """Perp closes [time x symbol] from the cached binance_vision parquet (or download)."""
    from ..data import binance_vision as bv

    cache = cfg.raw_dir() / "binvision"
    cache.mkdir(parents=True, exist_ok=True)
    since = pd.Timestamp(cfg.universe.since, tz="UTC")
    until = pd.Timestamp.now(tz="UTC")
    closes: dict[str, pd.Series] = {}
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for sym in symbols:
            px_path = cache / f"{sym}_px.parquet"
            try:
                if px_path.exists():
                    closes[sym] = pd.read_parquet(px_path)["perp"]
                else:
                    s = bv.klines_close("um", sym, interval, since, until, client)
                    if not s.empty:
                        closes[sym] = s
            except Exception as exc:  # noqa: BLE001
                logger.warning("pairs: skip {} ({})", sym, str(exc)[:60])
    if not closes:
        raise RuntimeError("no price data for pairs")
    px = pd.DataFrame(closes).sort_index()
    px.index = px.index.floor("1D")
    return px[~px.index.duplicated(keep="last")]


def run_pairs(cfg: Config | None = None, symbols: list[str] | None = None) -> dict:
    cfg = cfg or load_config()
    prices = _load_prices(cfg, symbols or PAIRS_UNIVERSE)
    logger.info("pairs stat-arb: {} symbols, {} bars", prices.shape[1], len(prices))

    fee_levels_bps = [0.0, 1.0, 2.0, 5.5]
    table, primary = [], None
    for bps in fee_levels_bps:
        r = pairs_stat_arb_backtest(prices, cfg, fee_rate=bps / 1e4, slippage_bps=0.0)
        mm = r["metrics"]
        table.append({"fee_bps": bps, "net_sharpe": mm["sharpe"], "net_return": mm["total_return"],
                      "psr": mm["psr"], "dsr": mm["deflated_sharpe"]})
        if bps == 2.0:
            primary = r
    primary = primary or r
    m = primary["metrics"]
    logger.info("pairs stat-arb ({} pairs, {} bars): GROSS sharpe={:.2f} ret={:.1%} | turnover/bar={:.2f}",
                m["n_pairs"], len(primary["net"]), m["gross_sharpe"], m["gross_total_return"], m["avg_turnover"])
    for row in table:
        logger.info("  fee={:>4.1f}bp/side -> NET sharpe={:6.2f} ret={:7.1%} PSR={:.2f} DSR={:.2f}",
                    row["fee_bps"], row["net_sharpe"], row["net_return"], row["psr"], row["dsr"])
    primary["fee_table"] = table
    return primary
