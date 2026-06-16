"""Purged, embargoed walk-forward cross-validation for time series.

Train only on the past, test on the future. Samples whose label horizon (``t1``)
reaches into the test window are purged from training, and an additional embargo
gap is removed just before each test fold. This kills the leakage that makes
naive backtests look brilliant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def purged_walk_forward_splits(
    sample_times: pd.DatetimeIndex,
    t1: pd.Series | np.ndarray,
    n_splits: int = 5,
    embargo_bars: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` integer-position pairs.

    Parameters
    ----------
    sample_times: index (timestamps) of the samples, sorted ascending.
    t1: per-sample label end-time (same length / order as sample_times).
    """
    times = pd.DatetimeIndex(sample_times)
    n = len(times)
    # normalize to nanoseconds: pandas >=3 indexes can be us/ms/ns, and a unit
    # mismatch between `times` and `t1` would silently break the purge.
    times_i8 = times.as_unit("ns").asi8

    t1_dt = pd.to_datetime(pd.Series(t1).to_numpy(), utc=True)
    # NaT end-times -> treat as far future so they are conservatively purged
    far_future = times.max() + pd.Timedelta(days=3650)
    t1_i8 = pd.DatetimeIndex(pd.Series(t1_dt).fillna(far_future)).as_unit("ns").asi8

    fold = n // (n_splits + 1)
    if fold == 0:
        raise ValueError(f"Not enough samples ({n}) for n_splits={n_splits}")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_splits + 1):
        test_start = k * fold
        test_end = (k + 1) * fold if k < n_splits else n
        test_idx = np.arange(test_start, test_end)
        test_start_ns = times_i8[test_start]

        train_idx = np.arange(0, test_start)
        if embargo_bars > 0:
            train_idx = train_idx[train_idx < (test_start - embargo_bars)]
        # purge: training label must end strictly before the test window starts
        if len(train_idx) > 0:
            keep = t1_i8[train_idx] < test_start_ns
            train_idx = train_idx[keep]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits
