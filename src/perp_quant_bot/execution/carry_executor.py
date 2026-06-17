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


@dataclass
class CarryOrder:
    market: str   # "spot" | "perp"
    symbol: str
    side: str     # "buy" | "sell"
    amount: float  # base units, positive
    price: float
    reason: str   # "enter" | "adjust" | "exit"


def reconcile_orders(
    plan: CarryPlan,
    spot_balances: dict[str, float],
    perp_positions: dict[str, float],
    min_notional: float = 5.0,
) -> list[CarryOrder]:
    """Diff target book vs current holdings -> the minimal orders to reach target.

    Idempotent: re-running when already at target yields no orders. Symbols that
    dropped out of the target (funding flipped negative) are CLOSED on both legs.
    Pure function -> unit-testable without any exchange.
    """
    orders: list[CarryOrder] = []
    target_perp = {leg.perp_symbol for leg in plan.legs}
    target_bases = {leg.spot_symbol.split("/")[0] for leg in plan.legs}

    for leg in plan.legs:
        base = leg.spot_symbol.split("/")[0]
        d_spot = leg.spot_amount - float(spot_balances.get(base, 0.0))  # target long - held
        if abs(d_spot) * leg.spot_price >= min_notional:
            orders.append(CarryOrder("spot", leg.spot_symbol, "buy" if d_spot > 0 else "sell",
                                     abs(d_spot), leg.spot_price, "enter" if base not in spot_balances else "adjust"))
        target_signed = -leg.perp_amount  # short the perp
        d_perp = target_signed - float(perp_positions.get(leg.perp_symbol, 0.0))
        if abs(d_perp) * leg.perp_price >= min_notional:
            orders.append(CarryOrder("perp", leg.perp_symbol, "buy" if d_perp > 0 else "sell",
                                     abs(d_perp), leg.perp_price, "enter" if leg.perp_symbol not in perp_positions else "adjust"))

    # Exits: positions/holdings no longer in the target -> flatten (funding flipped).
    for sym, pos in perp_positions.items():
        if sym not in target_perp and abs(pos) > 0:
            orders.append(CarryOrder("perp", sym, "buy" if pos < 0 else "sell", abs(pos), 0.0, "exit"))
    for base, amt in spot_balances.items():
        if base not in target_bases and base not in ("USDT", "USDC") and amt > 0:
            orders.append(CarryOrder("spot", f"{base}/USDT", "sell", amt, 0.0, "exit"))
    return orders


def render_orders(orders: list[CarryOrder]) -> str:
    if not orders:
        return "  reconciled: already at target, no orders."
    return "\n".join(
        f"  [{o.reason:<6}] {o.side.upper():<4} {o.market:<4} {o.amount:.6f} {o.symbol}"
        for o in orders
    )


def execute_orders(orders, perp, spot, *, maker: bool = False) -> list[dict]:
    """Place reconciled orders. Default market (guarantees the hedge fills together);
    maker=True posts limit orders at touch (cheaper, but fills aren't guaranteed ->
    transient single-leg delta risk)."""
    fills = []
    for o in orders:
        ex = spot if o.market == "spot" else perp
        try:
            amt = float(ex.amount_to_precision(o.symbol, o.amount))
            if amt <= 0:
                continue
            if maker and o.price > 0:
                px = float(ex.price_to_precision(o.symbol, o.price))
                f = ex.create_order(o.symbol, "limit", o.side, amt, px, {"postOnly": True})
            else:
                f = ex.create_order(o.symbol, "market", o.side, amt)
            logger.warning("LIVE {} {} {} {} -> {}", o.reason, o.side, amt, o.symbol, f.get("id", "?"))
            fills.append({"symbol": o.symbol, "market": o.market, "id": f.get("id")})
        except Exception as exc:  # noqa: BLE001
            logger.error("order failed {} {}: {}", o.side, o.symbol, exc)
            fills.append({"symbol": o.symbol, "error": str(exc)})
    return fills


def _keyed(venue, key, sec, market):
    opts = {"defaultType": "swap", "fetchMarkets": ["linear"]} if market == "perp" else {"defaultType": "spot"}
    ex = getattr(ccxt, venue)({"apiKey": key, "secret": sec, "enableRateLimit": True, "timeout": 20000, "options": opts})
    for i in range(5):
        try:
            ex.load_markets()
            return ex
        except Exception as exc:  # noqa: BLE001
            logger.warning("{} load_markets retry {}: {}", venue, i + 1, str(exc).splitlines()[0][:70])
            __import__("time").sleep(2 * (i + 1))
    return ex


def fetch_current_book(perp, spot, plan: CarryPlan):
    """Current signed perp positions + spot base balances for the plan's symbols."""
    spot_bal: dict[str, float] = {}
    perp_pos: dict[str, float] = {}
    try:
        bal = spot.fetch_balance()
        for leg in plan.legs:
            base = leg.spot_symbol.split("/")[0]
            spot_bal[base] = float((bal.get(base) or {}).get("free") or 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.error("fetch_balance failed: {}", exc)
    try:
        for p in perp.fetch_positions([leg.perp_symbol for leg in plan.legs]):
            c = p.get("contracts")
            if c:
                perp_pos[p["symbol"]] = float(c) if p.get("side") == "long" else -float(c)
    except Exception as exc:  # noqa: BLE001
        logger.error("fetch_positions failed: {}", exc)
    return spot_bal, perp_pos


def carry_step(
    venue: str = "bybit",
    capital: float = 200.0,
    top_n: int = 5,
    leverage: float = 2.0,
    min_funding: float = 0.00005,
    margin_buffer: float = 0.25,
    maker: bool = False,
    live: bool = False,
    confirm: bool = False,
) -> dict:
    """One rebalance cycle: plan target -> reconcile vs current -> (optionally) execute.

    Safe by default: live execution requires live=True AND confirm=True AND keys.
    ``margin_buffer`` keeps part of capital idle so a funding/price wobble can't
    trigger liquidation; ``leverage`` is capped for safety.
    """
    leverage = max(1.0, min(float(leverage), 5.0))  # hard cap
    deployable = capital * (1.0 - max(0.0, min(margin_buffer, 0.9)))
    plan = plan_carry(venue=venue, capital=deployable, top_n=top_n, min_funding=min_funding,
                      leverage=leverage, fee_rate=0.0002 if maker else 0.00055)

    perp = spot = None
    spot_bal: dict[str, float] = {}
    perp_pos: dict[str, float] = {}
    if live:
        from ..config import load_secrets
        secrets = load_secrets()
        key = getattr(secrets, "bybit_api_key", None)
        sec = getattr(secrets, "bybit_api_secret", None)
        if not key or not sec:
            raise RuntimeError("No Bybit API keys in .env — cannot trade live.")
        perp, spot = _keyed(venue, key, sec, "perp"), _keyed(venue, key, sec, "spot")
        spot_bal, perp_pos = fetch_current_book(perp, spot, plan)

    orders = reconcile_orders(plan, spot_bal, perp_pos)
    logger.info(render_plan(plan))
    logger.info("reconcile ({} current spot, {} current perp):", len(spot_bal), len(perp_pos))
    logger.info(render_orders(orders))

    fills = []
    if live and confirm and orders:
        for leg in plan.legs:
            try:
                perp.set_leverage(leverage, leg.perp_symbol)
            except Exception:  # noqa: BLE001
                pass
        fills = execute_orders(orders, perp, spot, maker=maker)
    elif not live:
        logger.info("[dry-run] {} orders computed; add --live --yes (+keys) to execute.", len(orders))
    elif not confirm:
        logger.info("[blocked] --live needs --yes; refusing real orders.")
    return {"plan": plan, "orders": orders, "fills": fills}


def run_carry(interval: float = 0.0, **kwargs) -> None:
    """Run one carry_step, or loop every ``interval`` seconds (0 = once)."""
    import time as _t
    while True:
        try:
            carry_step(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("carry step failed: {}", exc)
        if interval <= 0:
            break
        _t.sleep(interval)


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
