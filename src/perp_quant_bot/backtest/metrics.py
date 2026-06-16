"""Performance metrics for strategy returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 365.25 * 24  # assume hourly
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    median_s = float(np.median(deltas.to_numpy())) if len(deltas) else 0.0
    if median_s <= 0:
        return 365.25 * 24
    return _SECONDS_PER_YEAR / median_s


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def sharpe(returns: pd.Series, bars_per_year: float) -> float:
    r = returns.dropna()
    if r.std(ddof=0) == 0 or len(r) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(bars_per_year))


def sortino(returns: pd.Series, bars_per_year: float) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or len(r) == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(bars_per_year))


def cagr(equity: pd.Series, bars_per_year: float) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / bars_per_year
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def performance_summary(
    returns: pd.Series,
    equity: pd.Series,
    trade_returns: list[float] | None = None,
    bars_per_year: float | None = None,
) -> dict:
    bpy = bars_per_year or infer_bars_per_year(equity.index)
    mdd = max_drawdown(equity)
    cg = cagr(equity, bpy)
    summary = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0,
        "cagr": cg,
        "sharpe": sharpe(returns, bpy),
        "sortino": sortino(returns, bpy),
        "max_drawdown": mdd,
        "calmar": float(cg / abs(mdd)) if mdd != 0 else 0.0,
        "volatility_ann": float(returns.std(ddof=0) * np.sqrt(bpy)),
        "n_bars": int(len(returns)),
    }
    if trade_returns is not None and len(trade_returns) > 0:
        tr = np.asarray(trade_returns, dtype=float)
        wins = tr[tr > 0]
        losses = tr[tr < 0]
        summary.update(
            {
                "n_trades": int(len(tr)),
                "hit_rate": float((tr > 0).mean()),
                "avg_win": float(wins.mean()) if len(wins) else 0.0,
                "avg_loss": float(losses.mean()) if len(losses) else 0.0,
                "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
            }
        )
    return summary
