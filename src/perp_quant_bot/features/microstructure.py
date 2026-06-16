"""Microstructure / perp-specific features: funding rate and open interest.

Lower-frequency series (funding posts every ~8h) are aligned to the bar index by
forward-fill (as-of), so a feature at bar t only uses the last *published* value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def microstructure_features(
    funding_oi: pd.DataFrame, index: pd.DatetimeIndex, cfg: Config
) -> pd.DataFrame:
    if funding_oi is None or funding_oi.empty:
        return pd.DataFrame(index=index)

    aligned = funding_oi.sort_index().reindex(index, method="ffill")
    feats: dict[str, pd.Series] = {}

    if "funding_rate" in aligned.columns and cfg.features.include_funding:
        fr = aligned["funding_rate"]
        feats["funding_rate"] = fr
        feats["funding_mean_24"] = fr.rolling(24).mean()
        feats["funding_mean_72"] = fr.rolling(72).mean()
        feats["funding_chg_24"] = fr - fr.shift(24)
        # cumulative carry over last day (cost/credit of holding)
        feats["funding_cum_24"] = fr.rolling(24).sum()

    if "open_interest" in aligned.columns and cfg.features.include_open_interest:
        oi = aligned["open_interest"]
        for w in (24, 72):
            feats[f"oi_chg_{w}"] = oi.pct_change(w)
            mean_w = oi.rolling(w).mean()
            std_w = oi.rolling(w).std()
            feats[f"oi_z_{w}"] = (oi - mean_w) / std_w.replace(0.0, np.nan)

    return pd.DataFrame(feats, index=index)
