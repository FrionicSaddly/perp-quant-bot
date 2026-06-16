"""Assemble the leak-free feature matrix from OHLCV + funding/OI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_conf import setup_logging
from .cross_asset import anchor_features
from .microstructure import microstructure_features
from .technical import compute_atr, technical_features

logger = setup_logging()


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    funding_oi: pd.DataFrame | None,
    cfg: Config,
    anchor_ohlcv: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, atr)`` aligned on a common, NaN-free index.

    *X* is the feature matrix; *atr* (price units) is returned for the labeler
    and the risk manager so the same volatility estimate is used everywhere.
    *anchor_ohlcv* (optional) adds leak-free cross-asset (e.g. BTC) features.
    """
    tech = technical_features(ohlcv, cfg)
    micro = microstructure_features(funding_oi, ohlcv.index, cfg)
    atr = compute_atr(ohlcv, cfg.features.atr_period)

    parts = [tech, micro]
    if cfg.features.use_cross_asset and anchor_ohlcv is not None and not anchor_ohlcv.empty:
        parts.append(anchor_features(anchor_ohlcv, ohlcv.index, cfg))

    X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely empty (e.g. OI unavailable on the venue)
    empty_cols = [c for c in X.columns if X[c].isna().all()]
    if empty_cols:
        logger.info("Dropping {} empty feature columns: {}", len(empty_cols), empty_cols)
        X = X.drop(columns=empty_cols)

    # Cut the warmup prefix up to the first fully-valid row.
    valid = X.dropna(how="any")
    if valid.empty:
        logger.warning("Feature matrix is empty after warmup")
        return valid, atr.reindex(valid.index)
    X = X.loc[valid.index[0]:]

    # Leak-safe imputation of INTERIOR gaps: forward-fill only (uses past),
    # never back-fill. Keeps a contiguous bar index so backtest returns don't
    # silently span dropped rows.
    interior = int(X.isna().to_numpy().sum())
    if interior:
        logger.info("Forward-filling {} interior NaNs (leak-safe, past-only)", interior)
        X = X.ffill().dropna(how="any")

    logger.info("Feature matrix: {} rows x {} cols", len(X), X.shape[1])
    atr = atr.reindex(X.index)
    return X, atr


def feature_columns(X: pd.DataFrame) -> list[str]:
    return list(X.columns)
