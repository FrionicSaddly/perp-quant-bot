"""`pqb` command-line interface."""
from __future__ import annotations

import importlib

import typer

from .config import load_config, load_secrets
from .logging_conf import setup_logging

app = typer.Typer(add_completion=False, help="perp-quant-bot: data, train, backtest, paper-trade")
logger = setup_logging()


@app.command()
def doctor() -> None:
    """Check Python deps, config, and API-key presence."""
    typer.echo("Dependencies:")
    for pkg in ["ccxt", "pandas", "numpy", "sklearn", "lightgbm", "pydantic", "typer", "yaml"]:
        try:
            m = importlib.import_module(pkg)
            typer.echo(f"  [ok]      {pkg} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  [MISSING] {pkg}: {exc}")

    cfg = load_config()
    typer.echo(
        f"Config OK: exchange={cfg.exchange.id} testnet={cfg.exchange.testnet} "
        f"symbols={cfg.universe.symbols} tf={cfg.universe.timeframe} mode={cfg.execution.mode}"
    )
    sec = load_secrets()
    typer.echo(
        f"Keys: bybit={'set' if sec.bybit_api_key else 'MISSING'} "
        f"opennews={'set' if sec.opennews_token else 'missing'}"
    )


@app.command()
def download() -> None:
    """Download OHLCV (+ funding/OI) for the configured universe."""
    from .data import load_or_download_funding, load_or_download_ohlcv
    from .data.exchange import make_data_exchange

    cfg = load_config()
    ex = make_data_exchange(cfg)
    for s in cfg.universe.symbols:
        load_or_download_ohlcv(cfg, s, ex)
        load_or_download_funding(cfg, s, ex)
    typer.echo("Download complete.")


@app.command()
def train() -> None:
    """Train models with purged walk-forward validation and save them."""
    from .pipeline.train import train_all

    res = train_all()
    typer.echo("\n=== Out-of-sample results ===")
    for sym, r in res.items():
        m = r["oos"]["metrics"]
        typer.echo(
            f"{sym}: sharpe={m['sharpe']:.2f} ret={m['total_return']:.1%} "
            f"maxDD={m['max_drawdown']:.1%} hit={m.get('hit_rate', float('nan')):.1%}"
        )


@app.command()
def backtest(symbol: str = typer.Option("", help="Specific symbol; empty = all")) -> None:
    """Backtest saved model(s). NOTE: in-sample if run on the training history."""
    import pandas as pd

    from .backtest import backtest_signal
    from .backtest.report import save_report
    from .data.exchange import make_data_exchange
    from .models import LightGBMModel
    from .pipeline.train import model_path, prepare_dataset

    cfg = load_config()
    ex = make_data_exchange(cfg)
    symbols = [symbol] if symbol else cfg.universe.symbols
    for s in symbols:
        p = model_path(cfg, s)
        if not p.exists():
            typer.echo(f"{s}: no model — run `pqb train` first")
            continue
        model = LightGBMModel.load(p)
        ds = prepare_dataset(cfg, s, ex)
        sig = pd.Series(model.predict_signal(ds["X"]), index=ds["X"].index)
        bt = backtest_signal(ds["ohlcv"].loc[ds["X"].index], sig, ds["atr_pct"], cfg, ds["funding"])
        m = bt["metrics"]
        save_report(bt["results"], m, cfg.path("reports"), name=s.replace("/", "-").replace(":", "-"))
        typer.echo(
            f"{s}: sharpe={m['sharpe']:.2f} psr={m.get('psr', float('nan')):.2f} "
            f"ret={m['total_return']:.1%} maxDD={m['max_drawdown']:.1%} "
            f"trades={m.get('n_trades', 0)}  (in-sample)"
        )


@app.command()
def leakcheck(symbol: str = typer.Option("", help="Symbol; empty = first in universe")) -> None:
    """Empirical leakage detector: clean vs shuffled-labels vs injected future leak."""
    from .pipeline.diagnostics import leak_check

    cfg = load_config()
    res = leak_check(cfg, symbol or None)
    typer.echo(
        f"{res['symbol']}: clean={res['clean']:.2f}  shuffled={res['shuffled']:.2f}  "
        f"leaked={res['leaked']:.2f}  -> {res['verdict']}"
    )
    typer.echo("(shuffled near 0 AND leaked >> clean  =>  pipeline is honest)")


@app.command()
def importance(symbol: str = typer.Option("", help="Symbol; empty = first in universe")) -> None:
    """Out-of-sample permutation importance (find noise features to prune)."""
    from .pipeline.diagnostics import permutation_importance

    cfg = load_config()
    res = permutation_importance(cfg, symbol or None)
    imp = res["importance"]
    typer.echo(f"{res['symbol']}: base accuracy={res['base_accuracy']:.3f}")
    typer.echo("Top features (most accuracy lost when shuffled):")
    for name, val in imp.head(12).items():
        typer.echo(f"  {name:<24} {val:+.4f}")
    typer.echo("Weakest (prune candidates):")
    for name, val in imp.tail(6).items():
        typer.echo(f"  {name:<24} {val:+.4f}")


@app.command()
def xsection(
    timeframe: str = typer.Option("1d", help="Bar timeframe for the factor"),
    lookback: int = typer.Option(30, help="Momentum lookback in bars"),
    top_frac: float = typer.Option(0.30, help="Long-top / short-bottom fraction"),
) -> None:
    """Cross-sectional momentum, market-neutral (validated on real history)."""
    from .strategies import run_cross_sectional

    res = run_cross_sectional(timeframe=timeframe, lookback=lookback, top_frac=top_frac)
    m = res["metrics"]
    typer.echo(
        f"cross-sectional ({m['n_symbols']} names): sharpe={m['sharpe']:.2f} "
        f"psr={m.get('psr', float('nan')):.2f} dsr={m.get('deflated_sharpe', float('nan')):.2f} "
        f"ret={m['total_return']:.1%} maxDD={m['max_drawdown']:.1%}"
    )
    typer.echo(
        f"  vs equal-weight market: ret={m.get('benchmark_eqw_return', 0.0):.1%} | "
        f"avg turnover/bar={m.get('avg_turnover', 0.0):.2f}"
    )


@app.command()
def carry(
    funding_venue: str = typer.Option("mexc", help="Venue with deep funding history"),
    top_frac: float = typer.Option(0.30, help="Long-low / short-high funding fraction"),
) -> None:
    """Cross-sectional funding-carry, market-neutral (validated on real funding history)."""
    from .strategies import run_funding_carry

    res = run_funding_carry(funding_venue=funding_venue, top_frac=top_frac)
    m = res["metrics"]
    typer.echo(
        f"funding-carry ({m['n_symbols']} names): sharpe={m['sharpe']:.2f} "
        f"psr={m.get('psr', float('nan')):.2f} dsr={m.get('deflated_sharpe', float('nan')):.2f} "
        f"ret={m['total_return']:.1%} maxDD={m['max_drawdown']:.1%}"
    )
    typer.echo(
        f"  funding share of PnL={m.get('funding_pnl_share', float('nan')):.0%} | "
        f"avg turnover/bar={m.get('avg_turnover', 0.0):.2f}"
    )


@app.command(name="funding-now")
def funding_now(venue: str = typer.Option("bybit", help="Exchange for the live funding snapshot")) -> None:
    """Live funding snapshot (carry signal: positive => LONG spot + SHORT perp)."""
    from .strategies.funding_carry import current_funding

    df = current_funding(venue=venue)
    if df.empty:
        typer.echo("No funding data.")
        return
    typer.echo(f"Live funding @ {venue} (positive = collect by LONG spot + SHORT perp):")
    for _, r in df.iterrows():
        flag = "  <- carry" if r["funding_rate"] > 0 else ""
        typer.echo(
            f"  {r['symbol']:18} {r['funding_rate'] * 100:+.4f}%/8h "
            f"(~{r['annualized_pct']:+6.1f}%/yr){flag}"
        )


@app.command()
def basis(
    source: str = typer.Option("binance_vision", help="binance_vision (deep multi-year) | mexc"),
    venue: str = typer.Option("mexc", help="ccxt venue when source=mexc"),
    top_k: int = typer.Option(0, help="Hold only the K richest-funding names (0 = full basket)"),
    weight: str = typer.Option("equal", help="equal | funding (full deploy, skew to richest funding)"),
) -> None:
    """Delta-neutral funding (basis) carry: long spot + short perp (validated).

    Concentration + weighting sweeps are always printed; pass --top-k / --weight to set
    the headline run (top-8 + funding-weight squeezes the most %; check the OOS column).
    """
    from .strategies import run_basis_carry

    res = run_basis_carry(venue=venue, source=source, top_k=top_k if top_k > 0 else None, weight_mode=weight)
    m = res["metrics"]
    typer.echo(
        f"basis-carry ({m['n_symbols']} names, daily): GROSS sharpe={m['gross_sharpe']:.2f} "
        f"ret={m['gross_total_return']:.1%} (funding {m['funding_total_return']:.1%}) | "
        f"turnover/bar={m['avg_turnover']:.2f}"
    )
    typer.echo("  net by fee (per side):")
    for row in res.get("fee_table", []):
        typer.echo(
            f"    {row['fee_bps']:>4.1f} bp -> sharpe={row['net_sharpe']:6.2f} "
            f"ret={row['net_return']:7.1%} PSR={row['psr']:.2f} DSR={row['dsr']:.2f}"
        )
    typer.echo("  (0bp=gross; 1-2bp~maker [how basis-arb is run]; 5.5bp=taker)")
    oos = res.get("oos", {})
    if oos:
        h1, h2 = oos.get("h1_in", {}), oos.get("h2_oos", {})
        typer.echo(
            f"  OOS @maker1bp: H1(in) sharpe={h1.get('sharpe', float('nan')):.2f} "
            f"ret={h1.get('return', float('nan')):.1%} | "
            f"H2(oos) sharpe={h2.get('sharpe', float('nan')):.2f} ret={h2.get('return', float('nan')):.1%}"
        )
    wc = res.get("weight_cmp", [])
    if wc:
        typer.echo("  weighting (equal vs funding, maker 1bp):")
        for row in wc:
            typer.echo(
                f"    {row['mode']:<8} -> sharpe={row['net_sharpe']:6.2f} ret={row['net_return']:7.1%} "
                f"DSR={row['dsr']:.2f} turn={row['turnover']:.2f} | OOS-H2 sharpe={row['oos_sharpe']:6.2f}"
            )
    lev = res.get("leverage", [])
    if lev:
        typer.echo("  leverage (ann return / maxDD; OPTIMISTIC - funding flips & frictions not modeled):")
        for row in lev:
            flag = "  !! LIQUIDATION RISK" if row["liquidation_risk"] else ""
            typer.echo(
                f"    {row['leverage']:>2}x -> ann={row['ann_return']:7.1%} "
                f"maxDD={row['max_dd']:7.1%} sharpe={row['sharpe']:5.2f}{flag}"
            )


@app.command()
def xfunding(
    venue_a: str = typer.Option("bybit", help="venue to SHORT when its funding is higher"),
    venue_b: str = typer.Option("okx", help="venue to LONG (lower funding)"),
    lookback_days: int = typer.Option(365, help="funding history depth"),
) -> None:
    """Cross-exchange funding-spread carry: perp-perp, delta-neutral, no spot.

    Collects the inter-venue funding differential. A second market-neutral edge to
    stack with the basis carry.
    """
    from .strategies import run_cross_exchange

    res = run_cross_exchange(venue_a=venue_a, venue_b=venue_b, lookback_days=lookback_days)
    m = res["metrics"]
    typer.echo(
        f"cross-exchange {venue_a} vs {venue_b} ({m['n_symbols']} names): "
        f"GROSS sharpe={m['gross_sharpe']:.2f} ret={m['gross_total_return']:.1%} | "
        f"engaged={m['pct_engaged']:.0%} turnover/bar={m['avg_turnover']:.2f}"
    )
    for row in res.get("fee_table", []):
        typer.echo(
            f"    {row['fee_bps']:>4.1f} bp -> sharpe={row['net_sharpe']:6.2f} "
            f"ret={row['net_return']:7.1%} PSR={row['psr']:.2f} DSR={row['dsr']:.2f}"
        )
    oos = res.get("oos", {})
    if oos:
        h1, h2 = oos.get("h1_in", {}), oos.get("h2_oos", {})
        typer.echo(
            f"  OOS @maker1bp: H1 sharpe={h1.get('sharpe', float('nan')):.2f} "
            f"ret={h1.get('return', float('nan')):.1%} | "
            f"H2 sharpe={h2.get('sharpe', float('nan')):.2f} ret={h2.get('return', float('nan')):.1%}"
        )


@app.command()
def pairs() -> None:
    """Statistical-arbitrage pairs: market-neutral mean reversion of beta-hedged spreads."""
    from .strategies import run_pairs

    res = run_pairs()
    m = res["metrics"]
    typer.echo(
        f"pairs stat-arb ({m['n_pairs']} pairs): GROSS sharpe={m['gross_sharpe']:.2f} "
        f"ret={m['gross_total_return']:.1%} | turnover/bar={m['avg_turnover']:.2f}"
    )
    for row in res.get("fee_table", []):
        typer.echo(
            f"    {row['fee_bps']:>4.1f} bp -> sharpe={row['net_sharpe']:6.2f} "
            f"ret={row['net_return']:7.1%} PSR={row['psr']:.2f} DSR={row['dsr']:.2f}"
        )


@app.command()
def cpcv(
    symbol: str = typer.Argument(..., help="e.g. ETH/USDT:USDT"),
    n_groups: int = typer.Option(6, help="time blocks"),
    k_test: int = typer.Option(2, help="test blocks per combination"),
) -> None:
    """Combinatorial Purged CV: OOS Sharpe DISTRIBUTION across all C(n,k) paths.

    One walk-forward number can be luck; this shows the spread. A robust edge stays
    positive across most paths — most of our directional configs do not.
    """
    import numpy as np
    import pandas as pd

    from .backtest import backtest_signal
    from .pipeline.train import build_model, prepare_dataset
    from .validation import combinatorial_purged_splits

    cfg = load_config()
    ds = prepare_dataset(cfg, symbol)
    X, y, t1, w = ds["X"], ds["y"], ds["t1"], ds["w"]
    splits = combinatorial_purged_splits(X.index, t1, n_groups, k_test, cfg.validation.embargo_bars)
    if not splits:
        typer.echo("not enough data for CPCV")
        return
    sharpes = []
    for tr, te in splits:
        m = build_model(cfg)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr].to_numpy())
        ti = X.index[te]
        sig = pd.Series(m.predict_signal(X.iloc[te]), index=ti)
        bt = backtest_signal(ds["ohlcv"].loc[ti], sig, ds["atr_pct"].loc[ti], cfg, ds["funding"])
        sharpes.append(bt["metrics"]["sharpe"])
    a = np.array([s for s in sharpes if np.isfinite(s)])
    typer.echo(f"CPCV {symbol}: {len(a)} paths (C({n_groups},{k_test}))")
    typer.echo(
        f"  OOS sharpe: mean={a.mean():.2f} median={np.median(a):.2f} std={a.std():.2f} "
        f"min={a.min():.2f} max={a.max():.2f} | %>0={np.mean(a > 0):.0%}"
    )
    typer.echo("  A real edge stays >0 across most paths; ~50% and mean~0 = no edge.")


@app.command()
def newslog(
    once: bool = typer.Option(False, help="Single poll then exit"),
    interval: int = typer.Option(90, help="Seconds between polls"),
) -> None:
    """Continuously log OpenNews items to disk (builds backtestable history)."""
    from .pipeline.news_logger import run_logger

    run_logger(once=once, interval=interval)


@app.command()
def paper(once: bool = typer.Option(False, help="Run a single iteration then exit")) -> None:
    """Run the paper/testnet trading loop (no real money)."""
    from .pipeline.trade import run_paper_loop

    run_paper_loop(once=once)


@app.command(name="carry-trade")
def carry_trade(
    venue: str = typer.Option("bybit", help="Exchange (must have spot + USDT perp + funding)"),
    capital: float = typer.Option(200.0, help="Total USDT to deploy across the book"),
    top_n: int = typer.Option(5, help="Number of highest-funding symbols to carry"),
    leverage: float = typer.Option(2.0, help="Perp leverage (capped at 5x for safety)"),
    min_funding: float = typer.Option(0.00005, help="Min funding/8h to engage (~5.5%/yr)"),
    margin_buffer: float = typer.Option(0.25, help="Fraction of capital kept idle (liquidation buffer)"),
    maker: bool = typer.Option(False, help="Post-only limit orders (cheaper; fills not guaranteed)"),
    interval: float = typer.Option(0.0, help="Loop every N seconds (0 = run once)"),
    live: bool = typer.Option(False, help="Place REAL orders (needs keys + --yes)"),
    yes: bool = typer.Option(False, help="Confirm real-money execution"),
) -> None:
    """Run the delta-neutral basis carry: LONG spot + SHORT perp, with position
    reconciliation (idempotent), margin buffer, leverage cap and funding-flip exits.

    DRY-RUN by default: prints the target book + the exact reconcile orders. No money
    moves without --live AND --yes AND Bybit keys in .env. Start tiny.
    """
    from .execution.carry_executor import run_carry

    run_carry(
        interval=interval, venue=venue, capital=capital, top_n=top_n, leverage=leverage,
        min_funding=min_funding, margin_buffer=margin_buffer, maker=maker,
        live=live, confirm=yes,
    )


@app.command()
def microlog(
    venue: str = typer.Option("bybit", help="Exchange to collect microstructure from"),
    interval: float = typer.Option(15.0, help="Seconds between polls"),
    once: bool = typer.Option(False, help="Single poll then exit (for testing/CI)"),
    duration: float = typer.Option(0.0, help="Run this many seconds then stop (0 = forever)"),
) -> None:
    """Collect perp microstructure (order-book imbalance, CVD, OI, funding) -> daily CSV.

    The short-horizon signals bar data lacks. Run on an always-on host to build
    history; later this feeds a microstructure-aware model. Ctrl-C stops cleanly.
    """
    from .pipeline.microstructure_logger import run_microstructure_logger

    run_microstructure_logger(
        venue=venue, interval=interval, once=once,
        duration=duration if duration > 0 else None,
    )


@app.command()
def liqlog(
    venue: str = typer.Option("bybit", help="Exchange to stream liquidations from"),
    duration: float = typer.Option(600.0, help="Seconds to stream (WebSocket)"),
) -> None:
    """Stream perp liquidation events over WebSocket (ccxt.pro) for a bounded window.

    Liquidation cascades are a strong short-horizon signal. Sporadic, so longer
    windows catch more. Persistence to a data branch is handled by the CI job.
    """
    from .pipeline.liquidations_logger import run_liquidations_logger

    events = run_liquidations_logger(venue=venue, duration=duration)
    typer.echo(f"collected {len(events)} liquidation events")
    for e in events[-10:]:
        typer.echo(f"  {e['datetime']} {e['symbol']} {e['side']} amt={e['amount']} @ {e['price']}")


if __name__ == "__main__":
    app()
