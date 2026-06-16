"""Assemble the leak-free feature matrix from OHLCV + funding/OI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_conf import setup_logging
from .microstructure import microstructure_features
from .technical import compute_atr, technical_features

logger = setup_logging()


def build_feature_matrix(
    ohlcv: pd.DataFrame, funding_oi: pd.DataFrame | None, cfg: Config
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, atr)`` aligned on a common, NaN-free index.

    *X* is the feature matrix; *atr* (price units) is returned for the labeler
    and the risk manager so the same volatility estimate is used everywhere.
    """
    tech = technical_features(ohlcv, cfg)
    micro = microstructure_features(funding_oi, ohlcv.index, cfg)
    atr = compute_atr(ohlcv, cfg.features.atr_period)

    X = pd.concat([tech, micro], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely empty (e.g. OI unavailable on the venue)
    empty_cols = [c for c in X.columns if X[c].isna().all()]
    if empty_cols:
        logger.info("Dropping {} empty feature columns: {}", len(empty_cols), empty_cols)
        X = X.drop(columns=empty_cols)

    # Drop warmup rows where any feature is still NaN
    before = len(X)
    X = X.dropna(how="any")
    logger.info("Feature matrix: {} rows x {} cols (dropped {} warmup rows)",
                len(X), X.shape[1], before - len(X))

    atr = atr.reindex(X.index)
    return X, atr


def feature_columns(X: pd.DataFrame) -> list[str]:
    return list(X.columns)
