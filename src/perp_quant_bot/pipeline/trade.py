"""(Paper / testnet) trading loop: fetch -> features -> predict -> size -> order.

Acts only on the most recent CLOSED bar. Live trading is disabled by design.
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import Config, load_config, load_secrets
from ..data.exchange import make_exchange
from ..data.funding import download_funding
from ..data.ohlcv import download_ohlcv
from ..execution import CcxtBroker, PaperBroker
from ..features import build_feature_matrix
from ..logging_conf import setup_logging
from ..models import LightGBMModel
from ..risk import RiskManager
from .train import model_path

logger = setup_logging()


def load_models(cfg: Config) -> dict[str, LightGBMModel]:
    models: dict[str, LightGBMModel] = {}
    for symbol in cfg.universe.symbols:
        p = model_path(cfg, symbol)
        if p.exists():
            models[symbol] = LightGBMModel.load(p)
        else:
            logger.warning("No model for {} ({}); run `pqb train`.", symbol, p.name)
    return models


def make_broker(cfg: Config, secrets):
    mode = cfg.execution.mode
    if mode == "paper":
        return PaperBroker(cfg.backtest.initial_capital, cfg.backtest.fee_rate)
    if mode == "testnet":
        return CcxtBroker(cfg, secrets)
    raise RuntimeError(f"Execution mode '{mode}' not allowed (live disabled).")


def run_once(cfg: Config, broker, models, exchange, rm: RiskManager) -> dict:
    tf = cfg.universe.timeframe
    tf_ms = exchange.parse_timeframe(tf) * 1000
    lookback = max(cfg.features.windows) + 250
    since_ms = exchange.milliseconds() - lookback * tf_ms
    decisions: dict[str, dict] = {}

    for symbol, model in models.items():
        ohlcv = download_ohlcv(exchange, symbol, tf, since_ms)
        if len(ohlcv) < 50:
            logger.warning("Not enough bars for {}", symbol)
            continue
        funding = download_funding(exchange, symbol, since_ms) if cfg.features.include_funding else None
        X, atr = build_feature_matrix(ohlcv, funding, cfg)
        if X.empty:
            logger.warning("No features for {}", symbol)
            continue

        signal = int(model.predict_signal(X.iloc[[-1]])[0])
        price = float(ohlcv["close"].iloc[-1])
        atr_pct_last = float((atr / ohlcv["close"]).reindex(X.index).iloc[-1])

        if isinstance(broker, PaperBroker):
            broker.update_price(symbol, price)
        equity = broker.get_equity()
        frac = float(rm.position_fraction(atr_pct_last))
        target_units = signal * (equity * frac) / price if price > 0 else 0.0
        fill = broker.set_target_position(symbol, target_units, price)

        decisions[symbol] = {
            "signal": signal,
            "price": price,
            "target_units": target_units,
            "equity": equity,
            "traded": fill is not None,
        }
        logger.info(
            "{} signal={} price={:.2f} target_units={:.6f} equity={:.2f}",
            symbol, signal, price, target_units, equity,
        )
    return decisions


def run_paper_loop(once: bool = False, cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    if cfg.execution.mode == "live":
        raise RuntimeError("Live trading is disabled in this codebase.")
    secrets = load_secrets()
    exchange = make_exchange(cfg)  # public market data
    models = load_models(cfg)
    if not models:
        raise RuntimeError("No trained models found. Run `pqb train` first.")
    broker = make_broker(cfg, secrets)
    rm = RiskManager(cfg.risk)

    logger.info(
        "Starting '{}' loop: {} symbols, poll {}s",
        cfg.execution.mode, len(models), cfg.execution.poll_seconds,
    )
    while True:
        try:
            run_once(cfg, broker, models, exchange, rm)
        except Exception as exc:  # noqa: BLE001
            logger.error("Iteration error: {}", exc)
        if once:
            break
        time.sleep(cfg.execution.poll_seconds)
