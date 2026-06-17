"""Meta-labeling (Lopez de Prado, Advances in Financial ML, Ch. 3).

Idea: predicting raw next-move direction on bar data is near-random. Instead:
  1. A simple, parameter-free **primary** rule picks the side (here: momentum —
     the sign of the return over the last ``window`` bars).
  2. A **secondary** model learns *whether to act* on the primary's call, i.e.
     P(primary side is the one that materializes). We trade only when that
     probability is high.

This improves **precision on the trades actually taken** (trade fewer, better),
which is the honest way to raise "accuracy" when direction itself is hard.

Both functions are strictly backward-looking (leak-safe): the primary side at
bar ``t`` uses only ``close[t-window..t]``; the meta-label uses the realized
triple-barrier outcome (whose end ``t1`` is purged from training by the
walk-forward splitter, exactly as for the directional labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def primary_side(close: pd.Series, window: int) -> pd.Series:
    """Momentum primary: +1 if price rose over the window, -1 if it fell, 0 if flat."""
    chg = close - close.shift(int(window))
    return pd.Series(np.sign(chg.to_numpy()), index=close.index).fillna(0.0)


def meta_labels(side: pd.Series, tb_label: pd.Series) -> pd.Series:
    """Binary meta-label: 1 if the primary *side* equals the realized barrier sign.

    ``tb_label`` is the triple-barrier label (-1/0/+1). A meta-label of 1 means
    "acting on the primary side at this bar would have been correct"; 0 means
    "stand down" (wrong side, or no decisive move).
    """
    side = side.reindex(tb_label.index).fillna(0.0)
    correct = (side != 0) & (tb_label != 0) & (np.sign(side) == np.sign(tb_label))
    return correct.astype(int)
