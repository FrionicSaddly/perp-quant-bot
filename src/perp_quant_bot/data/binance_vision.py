"""Loader for Binance public data dumps (data.binance.vision).

This static CDN serves multi-year history of funding rates and spot/perp klines as
monthly CSV zips. It is reachable even where the Binance trading API is geo-blocked,
so it is our source of DEEP funding history for a robust basis-carry backtest.

Monthly files are fetched concurrently (the CDN is the latency bottleneck).
Symbols are raw Binance tickers, e.g. "BTCUSDT".
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from ..logging_conf import setup_logging

logger = setup_logging()

BASE = "https://data.binance.vision/data"


def _months(since: pd.Timestamp, until: pd.Timestamp):
    cur = (since if since.tzinfo else since.tz_localize("UTC")).to_period("M")
    end = (until if until.tzinfo else until.tz_localize("UTC")).to_period("M")
    while cur <= end:
        yield f"{cur.year:04d}-{cur.month:02d}"
        cur += 1


def _days(since: pd.Timestamp, until: pd.Timestamp):
    cur = (since if since.tzinfo else since.tz_localize("UTC")).normalize()
    end = (until if until.tzinfo else until.tz_localize("UTC")).normalize()
    while cur <= end:
        yield cur.strftime("%Y-%m-%d")
        cur += pd.Timedelta(days=1)


def _get_zip_csv(url: str, client: httpx.Client) -> str | None:
    try:
        r = client.get(url)
    except Exception:  # noqa: BLE001
        return None
    if r.status_code != 200:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        return z.read(z.namelist()[0]).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None


def _fetch_all(urls: list[str], client: httpx.Client, workers: int = 12) -> list[str | None]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda u: _get_zip_csv(u, client), urls))


def funding_history(symbol: str, since, until, client: httpx.Client) -> pd.Series:
    """8h funding-rate series for a UM perp symbol (e.g. 'BTCUSDT')."""
    months = list(_months(pd.Timestamp(since), pd.Timestamp(until)))
    urls = [f"{BASE}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip" for ym in months]
    frames = []
    for txt in _fetch_all(urls, client):
        if not txt:
            continue
        df = pd.read_csv(io.StringIO(txt))
        if "calc_time" in df.columns and "last_funding_rate" in df.columns:
            frames.append(
                df[["calc_time", "last_funding_rate"]].rename(
                    columns={"calc_time": "t", "last_funding_rate": "f"}
                )
            )
    if not frames:
        return pd.Series(dtype=float)
    d = pd.concat(frames, ignore_index=True).dropna()
    idx = pd.to_datetime(d["t"].astype("int64"), unit="ms", utc=True)
    return pd.Series(d["f"].astype(float).to_numpy(), index=idx).sort_index()


def klines_close(market: str, symbol: str, interval: str, since, until, client: httpx.Client) -> pd.Series:
    """Close-price series. market='um' (perp) or 'spot'."""
    seg = "futures/um" if market == "um" else "spot"
    months = list(_months(pd.Timestamp(since), pd.Timestamp(until)))
    urls = [f"{BASE}/{seg}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip" for ym in months]
    frames = []
    for txt in _fetch_all(urls, client):
        if not txt:
            continue
        has_header = txt.splitlines()[0].lower().startswith("open_time")
        df = pd.read_csv(io.StringIO(txt), header=0 if has_header else None)
        if has_header:
            ot, cl = df["open_time"], df["close"]
        else:
            ot, cl = df.iloc[:, 0], df.iloc[:, 4]
        frames.append(pd.DataFrame({"t": ot, "c": cl}))
    if not frames:
        return pd.Series(dtype=float)
    d = pd.concat(frames, ignore_index=True).dropna()
    idx = pd.to_datetime(d["t"].astype("int64"), unit="ms", utc=True)
    return pd.Series(d["c"].astype(float).to_numpy(), index=idx).sort_index()


def klines_ohlcv(market: str, symbol: str, interval: str, since, until, client: httpx.Client) -> pd.DataFrame:
    """Full OHLCV (+ taker_buy_volume) for a Binance symbol. market='um'|'spot'.

    Binance klines carry the per-bar taker-buy base volume (aggressive buy flow),
    so a CVD-style order-flow feature is available historically at any interval.
    """
    seg = "futures/um" if market == "um" else "spot"
    months = list(_months(pd.Timestamp(since), pd.Timestamp(until)))
    urls = [f"{BASE}/{seg}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip" for ym in months]
    frames = []
    for txt in _fetch_all(urls, client):
        if not txt:
            continue
        has_header = txt.splitlines()[0].lower().startswith("open_time")
        df = pd.read_csv(io.StringIO(txt), header=0 if has_header else None)
        # positional: 0 open_time,1 open,2 high,3 low,4 close,5 volume,9 taker_buy_base_volume
        sub = df.iloc[:, [0, 1, 2, 3, 4, 5, 9]].copy()
        sub.columns = ["t", "open", "high", "low", "close", "volume", "taker_buy_volume"]
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).dropna()
    d = d[pd.to_numeric(d["t"], errors="coerce").notna()]
    idx = pd.to_datetime(d["t"].astype("int64"), unit="ms", utc=True)
    out = pd.DataFrame(
        {
            "open": d["open"].astype(float).to_numpy(),
            "high": d["high"].astype(float).to_numpy(),
            "low": d["low"].astype(float).to_numpy(),
            "close": d["close"].astype(float).to_numpy(),
            "volume": d["volume"].astype(float).to_numpy(),
            "taker_buy_volume": d["taker_buy_volume"].astype(float).to_numpy(),
        },
        index=idx,
    )
    return out[~out.index.duplicated(keep="first")].sort_index()


def metrics_history(symbol: str, since, until, client: httpx.Client) -> pd.DataFrame:
    """Binance futures 5-min metrics: OI, taker buy/sell ratio, top-trader & retail
    long/short ratios. Daily dumps; indexed by create_time (UTC)."""
    days = list(_days(pd.Timestamp(since), pd.Timestamp(until)))
    urls = [f"{BASE}/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{d}.zip" for d in days]
    frames = []
    for txt in _fetch_all(urls, client):
        if not txt:
            continue
        try:
            df = pd.read_csv(io.StringIO(txt))
        except Exception:  # noqa: BLE001
            continue
        if "create_time" in df.columns:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    idx = pd.to_datetime(d["create_time"], utc=True)
    d = d.drop(columns=[c for c in ("create_time", "symbol") if c in d.columns])
    d.index = idx
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[~d.index.duplicated(keep="first")].sort_index()
