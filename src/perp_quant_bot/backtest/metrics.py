"""Performance metrics for strategy returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Probability that the true (per-bar) Sharpe ratio exceeds *benchmark_sr*.

    Bailey & Lopez de Prado: corrects the observed Sharpe for sample length, skew,
    and fat tails. A high backtest Sharpe with low PSR is likely luck/overfit.
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 10:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr = r.mean() / sd
    s = pd.Series(r)
    skew = float(s.skew())
    kurt = float(s.kurt()) + 3.0  # pandas returns excess kurtosis -> convert to regular
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes) -> float:
    """Deflated Sharpe Ratio: PSR against the Sharpe you'd expect from the BEST of
    N random trials. Accounts for selection bias / multiple testing — a high DSR is
    far more credible than a high raw Sharpe.

    *trial_sharpes* must be in the SAME (per-bar) units as the returns series.
    """
    trials = np.asarray([t for t in trial_sharpes if np.isfinite(t)], dtype=float)
    n_trials = len(trials)
    if n_trials < 2:
        return probabilistic_sharpe_ratio(returns, 0.0)
    var_sr = float(trials.var(ddof=1))
    if var_sr <= 0:
        return probabilistic_sharpe_ratio(returns, 0.0)
    gamma = 0.5772156649015329  # Euler-Mascheroni
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(var_sr) * ((1.0 - gamma) * z1 + gamma * z2)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr0)


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
        "psr": probabilistic_sharpe_ratio(returns),
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
