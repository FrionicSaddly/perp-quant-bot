"""Cross-sectional momentum, market-neutral. Fully backtestable on real history.

Each bar we rank a universe of perps by past-return momentum, then go LONG the top
fraction and SHORT the bottom fraction with net-zero, gross-1 weights. This removes
market beta (the dominant noise that sank single-name directional models) and
isolates *relative strength* — the classic cross-sectional factor.

Decisions use only past data (rank at close[t], hold over t -> t+1). Costs are charged
on turnover. Parameters are standard defaults; they are intentionally NOT tuned to the
result (that would be overfitting — see DSR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.metrics import deflated_sharpe_ratio, performance_summary
from ..config import Config, load_config
from ..data.exchange import make_data_exchange
from ..data.ohlcv import _sanitize, download_ohlcv
from ..logging_conf import setup_logging

logger = setup_logging()

DEFAULT_UNIVERSE = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT",
    "DOGE/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
    "DOT/USDT:USDT", "TRX/USDT:USDT", "ATOM/USDT:USDT", "ETC/USDT:USDT", "FIL/USDT:USDT",
    "XLM/USDT:USDT", "EOS/USDT:USDT", "UNI/USDT:USDT", "AAVE/USDT:USDT", "NEAR/USDT:USDT",
    "ALGO/USDT:USDT", "ICP/USDT:USDT",
]


def cross_sectional_backtest(
    close: pd.DataFrame,
    cfg: Config,
    lookback: int = 30,
    skip: int = 1,
    top_frac: float = 0.30,
    min_names: int = 6,
) -> dict:
    """Pure backtest math (no network): close matrix [time x symbols] -> results."""
    rets = close.pct_change()

    # momentum over [t-skip-lookback, t-skip]; skip the most recent bar(s) to avoid
    # short-term reversal contaminating the signal.
    mom = close.shift(skip) / close.shift(skip + lookback) - 1.0
    n_avail = mom.notna().sum(axis=1)
    ranks = mom.rank(axis=1, pct=True)

    long = (ranks >= 1.0 - top_frac).astype(float)
    short = (ranks <= top_frac).astype(float)
    long_w = long.div(long.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * 0.5
    short_w = short.div(short.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * 0.5
    w = (long_w - short_w).where(n_avail >= min_names, 0.0)  # gross ~1, net ~0

    w_used = w.shift(1).fillna(0.0)  # decide at close[t], hold over t -> t+1
    port_ret = (w_used * rets).sum(axis=1)

    turnover = (w_used - w_used.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_rate = cfg.backtest.fee_rate + cfg.backtest.slippage_bps / 1e4
    net = port_ret - turnover * cost_rate

    active = w_used.abs().sum(axis=1) > 0
    if active.any():
        net = net.loc[active.idxmax():]
    equity = cfg.backtest.initial_capital * (1.0 + net).cumprod()

    metrics = performance_summary(net, equity)

    k = 5
    chunks = np.array_split(net.dropna().to_numpy(), k)
    trial_sharpes = [float(c.mean() / c.std()) for c in chunks if len(c) > 5 and c.std() > 0]
    metrics["deflated_sharpe"] = deflated_sharpe_ratio(net, trial_sharpes)

    eqw = rets.mean(axis=1).loc[net.index]
    bh_equity = (1.0 + eqw.fillna(0.0)).cumprod()
    metrics["benchmark_eqw_return"] = float(bh_equity.iloc[-1] - 1.0) if len(bh_equity) else 0.0
    metrics["n_symbols"] = int(close.shape[1])
    metrics["avg_turnover"] = float(turnover.loc[net.index].mean()) if len(net) else 0.0

    return {"metrics": metrics, "equity": equity, "weights": w, "net": net}


def _load_close_matrix(cfg: Config, symbols: list[str], timeframe: str, exchange) -> pd.DataFrame:
    venue = cfg.data.exchange_id or cfg.exchange.id
    since = exchange.parse8601(cfg.universe.since)
    closes: dict[str, pd.Series] = {}
    for s in symbols:
        path = cfg.raw_dir() / f"{venue}_{_sanitize(s)}_{timeframe}_ohlcv.parquet"
        try:
            if path.exists():
                df = pd.read_parquet(path)
            else:
                df = download_ohlcv(exchange, s, timeframe, since)
                if not df.empty:
                    df.to_parquet(path)
        except Exception as exc:  # noqa: BLE001 (e.g. symbol not listed on this venue)
            logger.warning("cross-sectional: skipping {} ({})", s, str(exc).splitlines()[0][:80])
            continue
        if not df.empty and len(df) > 200:
            closes[s] = df["close"]
        else:
            logger.warning("cross-sectional: skipping {} (insufficient data)", s)
    if not closes:
        raise RuntimeError("No symbols with sufficient data for cross-sectional run")
    return pd.DataFrame(closes).sort_index()


def run_cross_sectional(
    cfg: Config | None = None,
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    lookback: int = 30,
    skip: int = 1,
    top_frac: float = 0.30,
    min_names: int = 6,
) -> dict:
    cfg = cfg or load_config()
    symbols = symbols or DEFAULT_UNIVERSE
    exchange = make_data_exchange(cfg)
    close = _load_close_matrix(cfg, symbols, timeframe, exchange)
    res = cross_sectional_backtest(close, cfg, lookback, skip, top_frac, min_names)
    m = res["metrics"]
    logger.info(
        "cross-sectional ({} names, {} {}): sharpe={:.2f} PSR={:.2f} DSR={:.2f} ret={:.1%} "
        "maxDD={:.1%} | eqw-bench ret={:.1%}",
        m["n_symbols"], len(res["net"]), timeframe, m["sharpe"], m["psr"],
        m["deflated_sharpe"], m["total_return"], m["max_drawdown"], m["benchmark_eqw_return"],
    )
    return res
