"""Combinatorial Purged CV (CPCV) + Probability of Backtest Overfitting (PBO).

Lopez de Prado, *Advances in Financial ML*, ch. 7 & 11. A single walk-forward gives
ONE out-of-sample path -> one Sharpe -> easy to fool yourself. CPCV partitions time
into ``n_groups`` blocks and tests on every combination of ``k_test`` blocks (training
on the rest, with purge + embargo), yielding C(n_groups, k_test) OOS paths and a whole
DISTRIBUTION of OOS performance. PBO (via CSCV) estimates the probability that the
configuration you'd pick as best in-sample lands below the median out-of-sample — a
direct, quantitative overfitting gauge.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def combinatorial_purged_splits(
    sample_times: pd.DatetimeIndex,
    t1: pd.Series | np.ndarray,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_bars: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` integer-position pairs for all C(n_groups, k_test)
    test-block combinations, purging train labels (``t1``) that overlap any test block
    and applying an embargo after each test block."""
    times = pd.DatetimeIndex(sample_times)
    n = len(times)
    if n < n_groups:
        return []
    times_i8 = times.as_unit("ns").asi8
    t1_dt = pd.to_datetime(pd.Series(t1).to_numpy(), utc=True)
    far_future = times.max() + pd.Timedelta(days=3650)
    t1_i8 = pd.DatetimeIndex(pd.Series(t1_dt).fillna(far_future)).as_unit("ns").asi8

    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for combo in combinations(range(n_groups), k_test):
        if any(len(groups[g]) == 0 for g in combo):
            continue
        test_idx = np.sort(np.concatenate([groups[g] for g in combo]))
        spans = [(times_i8[groups[g][0]], times_i8[groups[g][-1]]) for g in combo]
        train_groups = [g for g in range(n_groups) if g not in combo]
        train_idx = np.sort(np.concatenate([groups[g] for g in train_groups]))

        keep = np.ones(len(train_idx), dtype=bool)
        ti = times_i8[train_idx]
        t1i = t1_i8[train_idx]
        for s, e in spans:  # purge any train label window overlapping a test span
            keep &= ~((t1i >= s) & (ti <= e))
        if embargo_bars > 0:  # embargo positions right after each test block
            for g in combo:
                end_pos = groups[g][-1]
                keep &= ~((train_idx > end_pos) & (train_idx <= end_pos + embargo_bars))
        train_idx = train_idx[keep]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def probability_of_backtest_overfitting(perf_matrix: np.ndarray, n_splits: int = 10) -> float:
    """CSCV PBO in [0, 1]. ``perf_matrix`` is [T_obs x n_configs] of per-bar performance
    (e.g. strategy returns per config). Lower is better (less overfit)."""
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return float("nan")
    T, _ = M.shape
    n_splits = n_splits if n_splits % 2 == 0 else n_splits - 1
    s = T // n_splits
    if s == 0:
        return float("nan")
    blocks = [M[i * s:(i + 1) * s] for i in range(n_splits)]
    half = n_splits // 2

    def sr(a: np.ndarray) -> np.ndarray:
        mu = a.mean(axis=0)
        sd = a.std(axis=0) + 1e-12
        return mu / sd

    logits: list[float] = []
    for combo in combinations(range(n_splits), half):
        is_block = np.concatenate([blocks[i] for i in combo], axis=0)
        oos_block = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo], axis=0)
        is_sr, oos_sr = sr(is_block), sr(oos_block)
        best = int(np.argmax(is_sr))
        rank = float((oos_sr <= oos_sr[best]).sum()) / (len(oos_sr) + 1)  # relative OOS rank
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1.0 - rank)))
    if not logits:
        return float("nan")
    return float((np.asarray(logits) <= 0).mean())
