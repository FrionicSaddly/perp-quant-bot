"""Risk management: volatility-targeted sizing and hard risk limits.

Position sizing matters more for survival than the model. We size so that hitting
the ATR-based stop costs approximately ``risk_per_trade`` of equity, and we cap
gross leverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RiskCfg


class RiskManager:
    def __init__(self, cfg: RiskCfg):
        self.cfg = cfg

    def position_fraction(self, atr_pct):
        """Fraction of equity to deploy as notional, from vol targeting.

        ``stop_pct = atr_stop_mult * atr_pct``; risking ``risk_per_trade`` to that
        stop implies notional fraction ``risk_per_trade / stop_pct``, capped at
        ``max_leverage``. Accepts a scalar or a pandas Series.
        """
        stop_pct = self.cfg.atr_stop_mult * atr_pct
        if isinstance(stop_pct, pd.Series):
            frac = self.cfg.risk_per_trade / stop_pct.replace(0.0, np.nan)
            return frac.clip(lower=0.0, upper=self.cfg.max_leverage).fillna(0.0)
        if stop_pct is None or not np.isfinite(stop_pct) or stop_pct <= 0:
            return 0.0
        return float(min(self.cfg.risk_per_trade / stop_pct, self.cfg.max_leverage))

    def size_position(self, equity: float, price: float, atr_pct: float) -> float:
        """Return position size in base units (coins/contracts)."""
        if price <= 0:
            return 0.0
        frac = self.position_fraction(atr_pct)
        notional = equity * float(frac)
        return notional / price

    def allowed_to_trade(self, daily_pnl_pct: float) -> bool:
        """Block new entries once the daily loss limit is breached."""
        return daily_pnl_pct > -abs(self.cfg.daily_max_loss)
