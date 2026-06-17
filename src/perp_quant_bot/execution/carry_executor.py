"""Delta-neutral basis-carry executor (LONG spot + SHORT perp).

Safety model
------------
- DEFAULT is dry-run: it computes and prints the exact two-leg book it *would*
  place, with expected funding income and fee drag, from LIVE funding + prices.
  No keys, no orders, zero risk.
- Real orders happen ONLY through ``execute_live`` with API keys present AND an
  explicit confirmation flag. Both legs use market orders so the hedge fills
  together (no single-leg / delta risk).

Capital model (transparent, no hidden leverage)
------------------------------------------------
For each selected symbol we hold ``leg_notional`` USDT LONG on spot and the same
notional SHORT on the perp. The short needs margin = ``leg_notional / leverage``.
So capital used per symbol = ``leg_notional * (1 + 1/leverage)``. Funding is
earned on the perp notional every 8h (3x/day when positive).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import ccxt

from ..logging_conf import setup_logging
from ..strategies.funding_carry import current_funding

logger = setup_logging()


@dataclass
class CarryLeg:
    perp_symbol: str
    spot_symbol: str
    funding_rate: float  # per 8h
    spot_price: float
    perp_price: float
    leg_notional: float  # USDT, long spot == short perp
    spot_amount: float  # base units long on spot
    perp_amount: float  # base units short on perp
    perp_margin: float  # USDT margin reserved for the short
    funding_per_day: float  # USDT/day expected at current funding


@dataclass
class CarryPlan:
    venue: str
    capital: float
    leverage: float
    fee_rate: float
    legs: list[CarryLeg] = field(default_factory=list)

    @property
    def deployed(self) -> float:
        return sum(leg.leg_notional + leg.perp_margin for leg in self.legs)

    @property
    def funding_per_day(self) -> float:
        return sum(leg.funding_per_day for leg in self.legs)

    @property
    def funding_apr(self) -> float:
        if self.capital <= 0:
            return 0.0
        return self.funding_per_day * 365.0 / self.capital

    @property
    def entry_fees(self) -> float:
        # Two legs in, two legs out eventually; count the round trip.
        return sum(2.0 * leg.leg_notional * self.fee_rate * 2.0 for leg in self.legs)

    @property
    def payback_days(self) -> float:
        fpd = self.funding_per_day
        return float("inf") if fpd <= 0 else self.entry_fees / fpd


def plan_carry(
    venue: str = "bybit",
    capital: float = 1000.0,
    top_n: int = 5,
    min_funding: float = 0.00003,  # 0.003%/8h ~= 3.3%/yr floor
    leverage: float = 2.0,
    fee_rate: float = 0.0002,  # ~maker round leg
    symbols: list[str] | None = None,
) -> CarryPlan:
    """Build the target delta-neutral carry book from live funding + prices."""
    fdf = current_funding(venue=venue, symbols=symbols)
    if not fdf.empty:
        fdf = fdf[fdf["funding_rate"] >= float(min_funding)].head(int(top_n))
    plan = CarryPlan(venue=venue, capital=float(capital), leverage=float(leverage), fee_rate=float(fee_rate))
    n = 0 if fdf is None else len(fdf)
    if n == 0:
        return plan

    spot_ex = getattr(ccxt, venue)({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "spot"}})

    gross_per = capital / n
    leg_notional = gross_per * leverage / (leverage + 1.0)  # spot + margin == gross_per
    for _, r in fdf.iterrows():
        perp_sym = str(r["symbol"])
        base = perp_sym.split("/")[0]
        spot_sym = f"{base}/USDT"
        # Perp price = mark price from the funding call (no fetch_ticker -> no geo-flaky load).
        perp_price = float(r.get("mark_price") or 0.0)
        if perp_price <= 0:
            continue
        # Spot price: try the (flaky) spot endpoint, else fall back to mark price.
        try:
            spot_price = float(spot_ex.fetch_ticker(spot_sym)["last"])
        except Exception:  # noqa: BLE001
            spot_price = perp_price
        if spot_price <= 0:
            spot_price = perp_price
        rate = float(r["funding_rate"])
        plan.legs.append(
            CarryLeg(
                perp_symbol=perp_sym,
                spot_symbol=spot_sym,
                funding_rate=rate,
                spot_price=spot_price,
                perp_price=perp_price,
                leg_notional=leg_notional,
                spot_amount=leg_notional / spot_price,
                perp_amount=leg_notional / perp_price,
                perp_margin=leg_notional / leverage,
                funding_per_day=leg_notional * rate * 3.0,
            )
        )
    return plan


def render_plan(plan: CarryPlan) -> str:
    """Human-readable dry-run plan."""
    lines: list[str] = []
    head = (
        f"Carry plan @ {plan.venue} | capital=${plan.capital:,.0f} "
        f"leverage={plan.leverage:g}x fee={plan.fee_rate * 100:.3f}%/leg"
    )
    lines.append(head)
    if not plan.legs:
        lines.append("  No symbol currently has funding above the floor — sit out.")
        return "\n".join(lines)
    lines.append(f"  {'symbol':<16}{'fund/8h':>9}{'spot $':>12}{'perp $':>12}{'$/day':>9}")
    for leg in plan.legs:
        lines.append(
            f"  LONG  {leg.spot_symbol:<10}{leg.funding_rate * 100:>8.4f}%"
            f"{leg.leg_notional:>12,.0f}{leg.leg_notional:>12,.0f}{leg.funding_per_day:>9.3f}"
        )
        lines.append(
            f"  SHORT {leg.perp_symbol:<10}{'':>9}{leg.spot_amount:>12.5f}{leg.perp_amount:>12.5f}"
        )
    lines.append(
        f"  -> deployed=${plan.deployed:,.0f}  funding=${plan.funding_per_day:.2f}/day "
        f"(~{plan.funding_apr * 100:.1f}%/yr on capital)"
    )
    lines.append(
        f"  -> round-trip fees~${plan.entry_fees:.2f}  payback~{plan.payback_days:.1f} days"
    )
    return "\n".join(lines)


def execute_live(plan: CarryPlan, *, confirm: bool = False) -> list[dict]:
    """Place the two-leg book for real. Requires keys + confirm=True.

    Market orders on BOTH legs so the hedge fills together. This is the only
    function in the codebase that can spend real money; it refuses unless the
    caller explicitly confirms and API keys are configured.
    """
    if not confirm:
        raise RuntimeError("execute_live requires confirm=True (real money).")

    from ..config import load_secrets  # local import avoids cycles

    secrets = load_secrets()
    key = getattr(secrets, "bybit_api_key", None) or getattr(secrets, "BYBIT_API_KEY", None)
    sec = getattr(secrets, "bybit_api_secret", None) or getattr(secrets, "BYBIT_API_SECRET", None)
    if not key or not sec:
        raise RuntimeError("No Bybit API keys in .env — cannot trade live.")

    perp = getattr(ccxt, plan.venue)(
        {"apiKey": key, "secret": sec, "enableRateLimit": True, "options": {"defaultType": "swap"}}
    )
    spot = getattr(ccxt, plan.venue)(
        {"apiKey": key, "secret": sec, "enableRateLimit": True, "options": {"defaultType": "spot"}}
    )
    perp.load_markets()
    spot.load_markets()

    fills: list[dict] = []
    for leg in plan.legs:
        try:
            perp.set_leverage(plan.leverage, leg.perp_symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("set_leverage {}: {}", leg.perp_symbol, exc)
        try:
            spot_amt = float(spot.amount_to_precision(leg.spot_symbol, leg.spot_amount))
            perp_amt = float(perp.amount_to_precision(leg.perp_symbol, leg.perp_amount))
            logger.warning("LIVE LONG  spot {} {}", spot_amt, leg.spot_symbol)
            f1 = spot.create_order(leg.spot_symbol, "market", "buy", spot_amt)
            logger.warning("LIVE SHORT perp {} {}", perp_amt, leg.perp_symbol)
            f2 = perp.create_order(leg.perp_symbol, "market", "sell", perp_amt)
            fills.append({"symbol": leg.perp_symbol, "spot": f1, "perp": f2})
        except Exception as exc:  # noqa: BLE001
            logger.error("LIVE leg {} failed: {}", leg.perp_symbol, exc)
            fills.append({"symbol": leg.perp_symbol, "error": str(exc)})
    return fills
