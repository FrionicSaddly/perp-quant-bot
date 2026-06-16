"""Backtesting: vectorized engine + performance metrics."""
from .engine import backtest_signal
from .metrics import performance_summary

__all__ = ["backtest_signal", "performance_summary"]
