"""Broker interface + a simple Order type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    amount: float  # base units (always positive)
    type: str = "market"
    price: float | None = None
    info: dict = field(default_factory=dict)


class Broker(ABC):
    """Minimal broker abstraction shared by paper and testnet implementations."""

    min_trade_units: float = 0.0

    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Signed position in base units (positive long, negative short)."""

    @abstractmethod
    def create_order(self, order: Order) -> dict: ...

    def set_target_position(self, symbol: str, target_units: float, price: float) -> dict | None:
        """Trade the delta between the current and target signed position."""
        current = self.get_position(symbol)
        delta = target_units - current
        if abs(delta) <= self.min_trade_units:
            return None
        side = "buy" if delta > 0 else "sell"
        return self.create_order(Order(symbol=symbol, side=side, amount=abs(delta), price=price))
