"""Delta-neutral funding (basis) carry: LONG spot + SHORT perp to harvest funding
with price risk hedged. This is the genuine, structural perp edge (what funding-arb
desks run), and it is backtestable from spot OHLCV + perp funding.

When funding > 0, perp longs pay shorts: holding short-perp + long-spot collects the
funding while the two price legs cancel. We engage the hedge per symbol only when the
prior funding was positive (decision uses only past data), equal-weight across engaged
symbols, and pay two-leg costs on entries/exits.

Real-world frictions NOT modeled: spot borrow, two-venue execution, margin. So treat
the result as an upper-ish bound on a low-risk carry, validated honestly with PSR/DSR.
"""
from __future__ import annotations

import time

import ccxt
import numpy as np
import pandas as pd

from ..backtest.metrics import (
    deflated_sharpe_ratio,
    infer_bars_per_year,
    performance_summary,
    sharpe,
)
from ..config import Config, load_config
from ..data.ohlcv import download_ohlcv
from ..logging_conf import setup_logging
from .funding_carry import _funding_series

logger = setup_logging()

BASKET = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "ADA/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
]


def _spot_symbol(perp: str) -> str:
    return perp.split(":")[0]  # "BTC/USDT:USDT" -> "BTC/USDT"


def basis_carry_backtest(
    perp_close: pd.DataFrame,
    spot_close: pd.DataFrame,
    funding: pd.DataFrame,
    cfg: Config,
    only_positive: bool = True,
    funding_smooth: int = 7,
    fee_rate: float | None = None,
    slippage_bps: float | None = None,
) -> dict:
    """Aligned [time x symbol] perp_close + spot_close + funding -> delta-neutral carry."""
    common = perp_close.index.intersection(spot_close.index).intersection(funding.index)
    cols = perp_close.columns
    pc = perp_close.loc[common, cols]
    sc = spot_close.reindex(columns=cols).loc[common]
    f = funding.reindex(columns=cols).loc[common]

    perp_ret = pc.pct_change()
    spot_ret = sc.pct_change()

    # Engage the hedge for bar t if SMOOTHED prior funding is positive (leak-free).
    # Smoothing + FIXED per-symbol weight (no daily rescaling) keeps turnover low — the
    # decisive factor, since the funding edge is thin.
    f_signal = f.rolling(funding_smooth, min_periods=1).mean()
    if only_positive:
        engage = (f_signal.shift(1) > 0).astype(float)
    else:
        engage = pd.DataFrame(1.0, index=common, columns=cols)
    w = engage * (1.0 / max(len(cols), 1))  # fixed notional per symbol

    # per-symbol delta-neutral return over bar t: funding received (short perp) + basis drift
    per_sym = f.fillna(0.0) + (spot_ret - perp_ret).fillna(0.0)
    gross = (w * per_sym).sum(axis=1)

    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) * 2.0  # two legs (spot + perp)
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
    metrics["n_symbols"] = int(len(cols))
    metrics["avg_turnover"] = float(turnover.loc[net.index].mean()) if len(net) else 0.0
    metrics["pct_engaged"] = float((w.loc[net.index].abs().sum(axis=1) > 0).mean())

    # gross (pre-cost) decomposition: is the funding edge real before fees/turnover?
    g = gross.loc[net.index]
    bpy = infer_bars_per_year(net.index)
    metrics["gross_total_return"] = float((1.0 + g).prod() - 1.0)
    metrics["gross_sharpe"] = sharpe(g, bpy)
    f_only = (w * funding.fillna(0.0)).sum(axis=1).loc[net.index]
    metrics["funding_total_return"] = float((1.0 + f_only).prod() - 1.0)
    return {"metrics": metrics, "equity": equity, "weights": w, "net": net}


def run_basis_carry(
    cfg: Config | None = None,
    symbols: list[str] | None = None,
    venue: str = "mexc",
) -> dict:
    cfg = cfg or load_config()
    symbols = symbols or BASKET

    perp_ex = getattr(ccxt, venue)(
        {"enableRateLimit": True, "timeout": 30000, "options": {"defaultType": "swap"}}
    )
    spot_ex = getattr(ccxt, venue)(
        {"enableRateLimit": True, "timeout": 30000, "options": {"defaultType": "spot"}}
    )
    since = perp_ex.parse8601(cfg.universe.since)

    perp_closes: dict[str, pd.Series] = {}
    spot_closes: dict[str, pd.Series] = {}
    fundings: dict[str, pd.Series] = {}
    for s in symbols:
        try:
            pdf = download_ohlcv(perp_ex, s, "1d", since)
            sdf = download_ohlcv(spot_ex, _spot_symbol(s), "1d", since)
            # funding posts every 8h; sum to a daily total (what a short perp collects/day)
            fser = _funding_series(perp_ex, s, since).resample("1D").sum()
        except Exception as exc:  # noqa: BLE001
            logger.warning("basis-carry: skipping {} ({})", s, str(exc).splitlines()[0][:70])
            continue
        if pdf.empty or sdf.empty or fser.empty or min(len(pdf), len(sdf), len(fser)) < 150:
            logger.warning("basis-carry: skipping {} (insufficient data)", s)
            continue
        perp_closes[s] = pdf["close"]
        spot_closes[s] = sdf["close"]
        fundings[s] = fser
        time.sleep(0.2)

    if len(perp_closes) < 4:
        raise RuntimeError(f"Only {len(perp_closes)} symbols had spot+perp+funding; need >= 4")

    def _mat(d: dict) -> pd.DataFrame:
        m = pd.DataFrame(d)
        m.index = m.index.floor("1D")
        return m[~m.index.duplicated(keep="last")].sort_index()

    perp_close = _mat(perp_closes)
    spot_close = _mat(spot_closes)
    funding = _mat(fundings)

    # Fee-sensitivity: 0 = gross, 1-2bp ~ maker (how basis-arb is actually executed),
    # 5.5bp = taker. Shows exactly where the (thin) edge survives.
    fee_levels_bps = [0.0, 1.0, 2.0, 5.5]
    table: list[dict] = []
    primary = None
    for bps in fee_levels_bps:
        r = basis_carry_backtest(
            perp_close, spot_close, funding, cfg, fee_rate=bps / 1e4, slippage_bps=0.0
        )
        mm = r["metrics"]
        table.append({
            "fee_bps": bps, "net_sharpe": mm["sharpe"], "net_return": mm["total_return"],
            "psr": mm["psr"], "dsr": mm["deflated_sharpe"],
        })
        if bps == 2.0:
            primary = r
    primary = primary or r
    m = primary["metrics"]
    logger.info(
        "basis-carry ({} names, {} daily bars @ {}): GROSS sharpe={:.2f} ret={:.1%} "
        "(funding {:.1%}) | turnover/bar={:.2f} engaged={:.0%}",
        m["n_symbols"], len(primary["net"]), venue, m["gross_sharpe"], m["gross_total_return"],
        m["funding_total_return"], m["avg_turnover"], m["pct_engaged"],
    )
    for row in table:
        logger.info(
            "  fee={:>4.1f}bp/side -> NET sharpe={:6.2f} ret={:7.1%} PSR={:.2f} DSR={:.2f}",
            row["fee_bps"], row["net_sharpe"], row["net_return"], row["psr"], row["dsr"],
        )
    primary["fee_table"] = table

    # Out-of-sample stability: split the funding-covered period in half (maker 1bp).
    common = perp_close.index.intersection(spot_close.index).intersection(funding.index)
    pc2, sc2, f2 = perp_close.loc[common], spot_close.loc[common], funding.loc[common]
    half = len(common) // 2
    oos: dict[str, dict] = {}
    for lbl, sl in [("h1_in", slice(0, half)), ("h2_oos", slice(half, len(common)))]:
        rr = basis_carry_backtest(
            pc2.iloc[sl], sc2.iloc[sl], f2.iloc[sl], cfg, fee_rate=0.0001, slippage_bps=0.0
        )
        oos[lbl] = {"sharpe": rr["metrics"]["sharpe"], "return": rr["metrics"]["total_return"]}
    primary["oos"] = oos
    logger.info(
        "  OOS @maker1bp: H1(in) sharpe={:.2f} ret={:.1%} | H2(oos) sharpe={:.2f} ret={:.1%}",
        oos["h1_in"]["sharpe"], oos["h1_in"]["return"], oos["h2_oos"]["sharpe"], oos["h2_oos"]["return"],
    )

    primary["data"] = {"perp_close": perp_close, "spot_close": spot_close, "funding": funding}
    return primary
