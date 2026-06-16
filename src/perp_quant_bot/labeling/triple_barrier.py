"""Triple-barrier labeling (Lopez de Prado).

For each bar ``t`` we look forward up to ``horizon_bars`` and place:
  * an upper barrier at ``close[t] + pt_atr_mult * ATR[t]``
  * a lower barrier at ``close[t] - sl_atr_mult * ATR[t]``
  * a vertical (time) barrier at ``t + horizon_bars``.

The first barrier the price path touches sets the label:
  * upper first  -> +1 (up move materialized first)
  * lower first  -> -1 (down move first)
  * neither      -> sign of the return at the vertical barrier, or 0 if |ret| < min_ret.

With ``pt_atr_mult == sl_atr_mult`` the label is a symmetric directional target,
which is the recommended default for a long/short classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def triple_barrier_labels(ohlcv: pd.DataFrame, atr: pd.Series, cfg: Config) -> pd.DataFrame:
    close = ohlcv["close"].to_numpy(dtype=float)
    high = ohlcv["high"].to_numpy(dtype=float)
    low = ohlcv["low"].to_numpy(dtype=float)
    atr_arr = atr.reindex(ohlcv.index).to_numpy(dtype=float)
    times = ohlcv.index

    n = len(close)
    H = int(cfg.labeling.horizon_bars)
    pt = float(cfg.labeling.pt_atr_mult)
    sl = float(cfg.labeling.sl_atr_mult)
    min_ret = float(cfg.labeling.min_ret)

    labels = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    exit_pos = np.full(n, -1, dtype=int)

    for i in range(n):
        a = atr_arr[i]
        # need a valid ATR and a full forward window to avoid truncated labels
        if not np.isfinite(a) or a <= 0 or (i + H) > (n - 1):
            continue
        entry = close[i]
        up = entry + pt * a
        dn = entry - sl * a
        end = i + H
        lab = 0
        ex = end
        ret = close[end] / entry - 1.0
        touched = False
        for j in range(i + 1, end + 1):
            hit_up = high[j] >= up
            hit_dn = low[j] <= dn
            if hit_up and hit_dn:
                # both barriers touched in the same bar: intrabar order is unknown
                # from OHLC, so label it neutral instead of guessing (no +1 bias).
                lab, ex, ret, touched = 0, j, close[j] / entry - 1.0, True
                break
            if hit_up:
                lab, ex, ret, touched = 1, j, up / entry - 1.0, True
                break
            if hit_dn:
                lab, ex, ret, touched = -1, j, dn / entry - 1.0, True
                break
        if not touched:
            ret = close[end] / entry - 1.0
            lab = 0 if abs(ret) < min_ret else (1 if ret > 0 else -1)
        labels[i] = lab
        rets[i] = ret
        exit_pos[i] = ex

    # keep t1 in the same dtype/resolution as the input index (avoids unit drift)
    t1 = pd.Series(pd.NaT, index=times, dtype=times.dtype)
    valid = exit_pos >= 0
    t1.iloc[np.where(valid)[0]] = times[exit_pos[valid]]

    out = pd.DataFrame(
        {"label": labels, "ret": rets, "t1": t1.to_numpy()},
        index=times,
    )
    return out.dropna(subset=["label"])
