"""Cross-exchange funding-spread carry (perp-perp, delta-neutral).

The SAME perp funds at different rates on different venues. SHORT the higher-funding
venue and LONG the lower-funding venue (same asset): the two price legs cancel
(delta-neutral, no spot, no borrow) and you collect the funding DIFFERENTIAL each
period. The edge is the persistent inter-venue funding spread.

Decision uses only PRIOR funding (leak-free). We engage a symbol only when the
prior spread magnitude clears a threshold (so the captured differential beats the
two-perp round-trip cost), hold with fixed weight to keep turnover low, and pay
costs on both perp legs when the position changes.

NOT modeled: the small inter-venue price basis, per-venue margin/liquidation, and
spread flips (the main risk). Validated honestly with PSR/DSR + an OOS split.
"""
from __future__ import annotations

import time

import ccxt
import numpy as np
import pandas as pd

from ..backtest.metrics import deflated_sharpe_ratio, infer_bars_per_year, performance_summary, sharpe
from ..config import Config, load_config
from ..logging_conf import setup_logging

logger = setup_logging()

XF_UNIVERSE = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "BNB/USDT:USDT", "ADA/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT", "LTC/USDT:USDT",
]

_VENUE_OPTS = {
    "bybit": {"options": {"defaultType": "swap", "fetchMarkets": ["linear"]}},
    "okx": {"options": {"defaultType": "swap"}},
    "binance": {"options": {"defaultType": "swap"}},
    "gate": {"options": {"defaultType": "swap"}},
}


def _paginate_funding(ex, symbol: str, since_ms: int, max_calls: int = 40) -> pd.Series:
    """Fetch funding-rate history back to ``since_ms`` (8h cadence) via ccxt pagination."""
    out: dict[int, float] = {}
    cursor = since_ms
    for _ in range(max_calls):
        try:
            batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=200)
        except Exception as exc:  # noqa: BLE001
            logger.warning("funding hist {} {}: {}", ex.id, symbol, str(exc).splitlines()[0][:70])
            break
        if not batch:
            break
        for x in batch:
            ts = int(x.get("timestamp") or 0)
            fr = x.get("fundingRate")
            if ts and fr is not None:
                out[ts] = float(fr)
        last = max(int(x.get("timestamp") or 0) for x in batch)
        if last <= cursor or len(batch) < 200:
            break
        cursor = last + 1
        time.sleep(getattr(ex, "rateLimit", 200) / 1000.0)
    if not out:
        return pd.Series(dtype=float)
    s = pd.Series(out)
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    return s.sort_index()


def funding_matrix(venue: str, symbols: list[str], since_ms: int) -> pd.DataFrame:
    """[time x symbol] 8h funding for a venue."""
    ex = getattr(ccxt, venue)({"enableRateLimit": True, "timeout": 20000, **_VENUE_OPTS.get(venue, {})})
    cols: dict[str, pd.Series] = {}
    for s in symbols:
        ser = _paginate_funding(ex, s, since_ms)
        if not ser.empty:
            cols[s] = ser
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def cross_exchange_funding_backtest(
    f_a: pd.DataFrame,
    f_b: pd.DataFrame,
    cfg: Config,
    threshold: float = 0.00005,  # min |spread|/8h to engage (~5.5%/yr)
    smooth: int = 3,
    fee_rate: float | None = None,
    slippage_bps: float | None = None,
) -> dict:
    """Aligned [time x symbol] funding for venue A and B -> spread carry net returns."""
    common_idx = f_a.index.intersection(f_b.index)
    common_cols = f_a.columns.intersection(f_b.columns)
    a = f_a.loc[common_idx, common_cols]
    b = f_b.loc[common_idx, common_cols]
    spread = a - b  # short A + long B collects this per 8h when > 0

    prior = spread.rolling(smooth, min_periods=1).mean().shift(1)
    pos = np.sign(prior).where(prior.abs() > threshold, 0.0).fillna(0.0)
    n = max(len(common_cols), 1)
    w = pos / n  # fixed fractional notional per symbol per leg

    gross = (w * spread).sum(axis=1)
    # turnover: a position change trades BOTH perp legs (A and B)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) * 2.0
    fr = cfg.backtest.fee_rate if fee_rate is None else fee_rate
    sl = cfg.backtest.slippage_bps if slippage_bps is None else slippage_bps
    cost_rate = fr + sl / 1e4
    net = gross - turnover * cost_rate

    active = w.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[active.idxmax():]
    equity = cfg.backtest.initial_capital * (1.0 + net).cumprod()

    metrics = performance_summary(net, equity)
    chunks = np.array_split(net.dropna().to_numpy(), 5)
    trials = [float(c.mean() / c.std()) for c in chunks if len(c) > 5 and c.std() > 0]
    metrics["deflated_sharpe"] = deflated_sharpe_ratio(net, trials)
    metrics["n_symbols"] = int(len(common_cols))
    metrics["pct_engaged"] = float((w.loc[net.index].abs().sum(axis=1) > 0).mean()) if len(net) else 0.0
    metrics["avg_turnover"] = float(turnover.loc[net.index].mean()) if len(net) else 0.0
    g = gross.loc[net.index]
    bpy = infer_bars_per_year(net.index)
    metrics["gross_sharpe"] = sharpe(g, bpy)
    metrics["gross_total_return"] = float((1.0 + g).prod() - 1.0)
    return {"metrics": metrics, "equity": equity, "net": net, "spread": spread, "weights": w}


def run_cross_exchange(
    cfg: Config | None = None,
    venue_a: str = "bybit",
    venue_b: str = "okx",
    symbols: list[str] | None = None,
    lookback_days: int = 365,
) -> dict:
    cfg = cfg or load_config()
    symbols = symbols or XF_UNIVERSE
    since_ms = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)).timestamp() * 1000)

    logger.info("cross-exchange funding: {} vs {} | {} symbols, {}d", venue_a, venue_b, len(symbols), lookback_days)
    f_a = funding_matrix(venue_a, symbols, since_ms)
    f_b = funding_matrix(venue_b, symbols, since_ms)
    if f_a.empty or f_b.empty:
        raise RuntimeError("could not load funding for one or both venues")

    fee_levels_bps = [0.0, 1.0, 2.0, 5.5]
    table: list[dict] = []
    primary = None
    for bps in fee_levels_bps:
        r = cross_exchange_funding_backtest(f_a, f_b, cfg, fee_rate=bps / 1e4, slippage_bps=0.0)
        mm = r["metrics"]
        table.append({"fee_bps": bps, "net_sharpe": mm["sharpe"], "net_return": mm["total_return"],
                      "psr": mm["psr"], "dsr": mm["deflated_sharpe"]})
        if bps == 2.0:
            primary = r
    primary = primary or r
    m = primary["metrics"]
    logger.info(
        "cross-exchange ({} vs {}, {} names, {} bars): GROSS sharpe={:.2f} ret={:.1%} | "
        "engaged={:.0%} turnover/bar={:.2f}",
        venue_a, venue_b, m["n_symbols"], len(primary["net"]), m["gross_sharpe"],
        m["gross_total_return"], m["pct_engaged"], m["avg_turnover"],
    )
    for row in table:
        logger.info("  fee={:>4.1f}bp/side -> NET sharpe={:6.2f} ret={:7.1%} PSR={:.2f} DSR={:.2f}",
                    row["fee_bps"], row["net_sharpe"], row["net_return"], row["psr"], row["dsr"])

    # OOS split (maker 1bp)
    net = primary["net"]
    half = len(net) // 2
    oos = {}
    if half > 10:
        idx = net.index
        common_idx = f_a.index.intersection(f_b.index)
        for lbl, sl in [("h1_in", idx[:half]), ("h2_oos", idx[half:])]:
            rr = cross_exchange_funding_backtest(
                f_a.loc[common_idx], f_b.loc[common_idx], cfg, fee_rate=0.0001, slippage_bps=0.0
            )
            sub = rr["net"].reindex(sl).dropna()
            oos[lbl] = {"sharpe": sharpe(sub, infer_bars_per_year(sub.index)) if len(sub) > 5 else float("nan"),
                        "return": float((1 + sub).prod() - 1) if len(sub) else float("nan")}
        logger.info("  OOS @maker1bp: H1 sharpe={:.2f} ret={:.1%} | H2 sharpe={:.2f} ret={:.1%}",
                    oos["h1_in"]["sharpe"], oos["h1_in"]["return"], oos["h2_oos"]["sharpe"], oos["h2_oos"]["return"])
    primary["fee_table"] = table
    primary["oos"] = oos
    return primary
