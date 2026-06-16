"""Paper broker (in-memory) and a thin ccxt testnet broker."""
from __future__ import annotations

from ..config import Config, Secrets, load_secrets
from ..logging_conf import setup_logging
from .broker import Broker, Order

logger = setup_logging()


class PaperBroker(Broker):
    """Zero-risk in-memory broker. Fills market orders at the last seen price."""

    def __init__(self, initial_cash: float = 10_000.0, fee_rate: float = 0.00055):
        self.cash = float(initial_cash)
        self.fee_rate = fee_rate
        self.positions: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.fills: list[dict] = []

    def update_price(self, symbol: str, price: float) -> None:
        self.last_price[symbol] = float(price)

    def get_equity(self) -> float:
        pnl = sum(self.positions.get(s, 0.0) * self.last_price.get(s, 0.0) for s in self.positions)
        return self.cash + pnl

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def create_order(self, order: Order) -> dict:
        price = order.price or self.last_price.get(order.symbol)
        if price is None:
            raise ValueError(f"No price for {order.symbol}; call update_price first")
        signed = order.amount if order.side == "buy" else -order.amount
        notional = abs(order.amount) * price
        fee = notional * self.fee_rate
        self.cash -= signed * price  # buying spends cash, shorting credits it
        self.cash -= fee
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + signed
        fill = {
            "symbol": order.symbol,
            "side": order.side,
            "amount": order.amount,
            "price": price,
            "fee": fee,
            "position": self.positions[order.symbol],
            "equity": self.get_equity(),
        }
        self.fills.append(fill)
        logger.info("PAPER {} {:.6f} {} @ {:.2f} (pos={:.6f})",
                    order.side, order.amount, order.symbol, price, self.positions[order.symbol])
        return fill


class CcxtBroker(Broker):
    """Routes orders to a ccxt exchange (Bybit testnet). Requires API keys."""

    def __init__(self, cfg: Config, secrets: Secrets | None = None):
        from ..data.exchange import make_exchange  # local import avoids cycle

        if cfg.execution.mode == "live" and not cfg.exchange.testnet:
            raise RuntimeError("Live trading is disabled in this codebase.")
        self.cfg = cfg
        self.exchange = make_exchange(cfg, secrets or load_secrets(), with_keys=True)
        self.quote = cfg.exchange.quote

    def get_equity(self) -> float:
        try:
            bal = self.exchange.fetch_balance()
            total = bal.get("total", {})
            return float(total.get(self.quote, 0.0))
        except Exception as exc:  # noqa: BLE001
            logger.error("fetch_balance failed: {}", exc)
            return 0.0

    def get_position(self, symbol: str) -> float:
        try:
            positions = self.exchange.fetch_positions([symbol])
        except Exception as exc:  # noqa: BLE001
            logger.error("fetch_positions failed: {}", exc)
            return 0.0
        for p in positions:
            if p.get("symbol") == symbol and p.get("contracts"):
                contracts = float(p["contracts"])
                return contracts if p.get("side") == "long" else -contracts
        return 0.0

    def set_leverage(self, symbol: str, leverage: float) -> None:
        try:
            self.exchange.set_leverage(leverage, symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("set_leverage failed for {}: {}", symbol, exc)

    def create_order(self, order: Order) -> dict:
        logger.info("TESTNET {} {:.6f} {}", order.side, order.amount, order.symbol)
        return self.exchange.create_order(order.symbol, order.type, order.side, order.amount)
