"""Cross-asset (anchor) features: the market driver (e.g. BTC) as exogenous inputs.

In crypto almost everything co-moves with BTC. Feeding the anchor's return/vol/zscore
as features gives alt-coin models a leak-free read on the broader regime. Alignment
is by as-of / forward-fill, so a feature at bar t uses only the anchor's data up to t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def anchor_features(
    anchor_ohlcv: pd.DataFrame | None,
    index: pd.DatetimeIndex,
    cfg: Config,
    prefix: str = "btc",
) -> pd.DataFrame:
    if anchor_ohlcv is None or anchor_ohlcv.empty:
        return pd.DataFrame(index=index)

    ac = anchor_ohlcv["close"].sort_index().reindex(index, method="ffill")
    log_ret = np.log(ac / ac.shift(1))

    feats: dict[str, pd.Series] = {f"{prefix}_ret_1": ac.pct_change()}
    for w in cfg.features.regime_windows:
        feats[f"{prefix}_ret_{w}"] = ac.pct_change(w)
        feats[f"{prefix}_vol_{w}"] = log_ret.rolling(w).std()
        mean_w = ac.rolling(w).mean()
        std_w = ac.rolling(w).std()
        feats[f"{prefix}_zscore_{w}"] = (ac - mean_w) / std_w.replace(0.0, np.nan)

    return pd.DataFrame(feats, index=index)
