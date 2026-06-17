"""Microstructure / perp-specific features: funding rate and open interest.

Lower-frequency series (funding posts every ~8h) are aligned to the bar index by
forward-fill (as-of), so a feature at bar t only uses the last *published* value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def microstructure_features(
    funding_oi: pd.DataFrame,
    index: pd.DatetimeIndex,
    cfg: Config,
    close: pd.Series | None = None,
) -> pd.DataFrame:
    if funding_oi is None or funding_oi.empty:
        return pd.DataFrame(index=index)

    aligned = funding_oi.sort_index().reindex(index, method="ffill")
    feats: dict[str, pd.Series] = {}

    # Price (close) is taken from the OHLCV frame, aligned to the same index, so the
    # OI/price divergence is leak-safe (uses only past closes).
    if close is not None:
        close = close.reindex(index)

    if "funding_rate" in aligned.columns and cfg.features.include_funding:
        fr = aligned["funding_rate"]
        feats["funding_rate"] = fr
        feats["funding_mean_24"] = fr.rolling(24).mean()
        feats["funding_mean_72"] = fr.rolling(72).mean()
        feats["funding_chg_24"] = fr - fr.shift(24)
        # cumulative carry over last day (cost/credit of holding)
        feats["funding_cum_24"] = fr.rolling(24).sum()
        # Funding z-score: how stretched is funding vs its own recent history?
        # Extreme positive funding = crowded longs -> mean-reversion / squeeze risk.
        for w in (72, 168):
            mu = fr.rolling(w).mean()
            sd = fr.rolling(w).std()
            feats[f"funding_z_{w}"] = (fr - mu) / sd.replace(0.0, np.nan)
        # Persistence of the funding sign over the last day (crowding regime).
        feats["funding_sign_persist_24"] = np.sign(fr).rolling(24).mean()

    if "open_interest" in aligned.columns and cfg.features.include_open_interest:
        oi = aligned["open_interest"]
        for w in (24, 72):
            feats[f"oi_chg_{w}"] = oi.pct_change(w)
            mean_w = oi.rolling(w).mean()
            std_w = oi.rolling(w).std()
            feats[f"oi_z_{w}"] = (oi - mean_w) / std_w.replace(0.0, np.nan)
        # OI vs price divergence (interpretation, backward-looking):
        #   price up + OI up   -> new money / continuation
        #   price up + OI down -> short-covering / weak rally
        # Encode as the product of recent price change and recent OI change.
        if close is not None:
            for w in (24, 72):
                price_chg = close.pct_change(w)
                oi_chg = oi.pct_change(w)
                feats[f"oi_price_div_{w}"] = np.sign(price_chg) * oi_chg

    # Bars before funding/OI coverage (or rolling warmup) get a NEUTRAL 0 rather than
    # NaN, so sparse perp data does not truncate the whole OHLCV history. 0 is a
    # constant prior (no future info), so this stays leak-free.
    return pd.DataFrame(feats, index=index).fillna(0.0)
