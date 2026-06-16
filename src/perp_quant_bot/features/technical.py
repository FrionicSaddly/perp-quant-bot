"""Technical (price/volume) features. All leak-free: row t uses data up to close[t]."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def compute_atr(ohlcv: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range (Wilder's smoothing). Returned in price units."""
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().rename("atr")


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal


def technical_features(ohlcv: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    close = ohlcv["close"]
    high, low, open_ = ohlcv["high"], ohlcv["low"], ohlcv["open"]
    volume = ohlcv["volume"]
    feats: dict[str, pd.Series] = {}

    log_ret = np.log(close / close.shift(1))
    feats["log_ret_1"] = log_ret

    for w in cfg.features.windows:
        feats[f"ret_{w}"] = close.pct_change(w)
        feats[f"vol_{w}"] = log_ret.rolling(w).std()
        mean_w = close.rolling(w).mean()
        std_w = close.rolling(w).std()
        feats[f"zscore_{w}"] = (close - mean_w) / std_w.replace(0.0, np.nan)
        feats[f"vol_ratio_{w}"] = volume / volume.rolling(w).mean()

    # RSI
    feats["rsi"] = _rsi(close, cfg.features.rsi_period)

    # MACD
    macd, macd_sig, macd_hist = _macd(close)
    feats["macd"] = macd / close
    feats["macd_signal"] = macd_sig / close
    feats["macd_hist"] = macd_hist / close

    # ATR as a normalized feature (% of price)
    atr = compute_atr(ohlcv, cfg.features.atr_period)
    feats["atr_pct"] = atr / close

    # Bollinger %B and bandwidth
    bb_mid = close.rolling(cfg.features.bb_period).mean()
    bb_std = close.rolling(cfg.features.bb_period).std()
    upper = bb_mid + 2 * bb_std
    lower = bb_mid - 2 * bb_std
    feats["bb_pctb"] = (close - lower) / (upper - lower).replace(0.0, np.nan)
    feats["bb_bw"] = (upper - lower) / bb_mid.replace(0.0, np.nan)

    # Candle geometry
    rng = (high - low).replace(0.0, np.nan)
    feats["body_frac"] = (close - open_) / rng
    feats["upper_wick"] = (high - close.combine(open_, max)) / rng
    feats["lower_wick"] = (close.combine(open_, min) - low) / rng

    # Regime / statistical features: trend strength, range position, drawdown, shape.
    for w in cfg.features.regime_windows:
        change = (close - close.shift(w)).abs()
        path = close.diff().abs().rolling(w).sum()
        feats[f"er_{w}"] = change / path.replace(0.0, np.nan)  # Kaufman efficiency ratio
        hi = high.rolling(w).max()
        lo = low.rolling(w).min()
        feats[f"donchian_pos_{w}"] = (close - lo) / (hi - lo).replace(0.0, np.nan)
        feats[f"dd_from_high_{w}"] = close / hi.replace(0.0, np.nan) - 1.0
        feats[f"ret_skew_{w}"] = log_ret.rolling(w).skew()
        feats[f"roll_sharpe_{w}"] = (
            log_ret.rolling(w).mean() / log_ret.rolling(w).std().replace(0.0, np.nan)
        )

    # Seasonality (cyclical encodings)
    idx = ohlcv.index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    feats["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=idx)
    feats["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=idx)
    feats["dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 7), index=idx)
    feats["dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 7), index=idx)

    return pd.DataFrame(feats, index=ohlcv.index)
