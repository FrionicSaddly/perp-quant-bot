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
) -> None:
    """Delta-neutral funding (basis) carry: long spot + short perp (validated).

    A concentration sweep is always printed; pass --top-k to make the headline run
    use that concentration (top-8 historically lifts return ~5.7%->7%/yr at similar risk).
    """
    from .strategies import run_basis_carry

    res = run_basis_carry(venue=venue, source=source, top_k=top_k if top_k > 0 else None)
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
    capital: float = typer.Option(1000.0, help="Total USDT to deploy across the book"),
    top_n: int = typer.Option(5, help="Number of highest-funding symbols to carry"),
    leverage: float = typer.Option(2.0, help="Perp leverage (margin = notional/leverage)"),
    min_funding: float = typer.Option(0.00003, help="Min funding/8h to engage (0.00003 ~= 3.3%/yr)"),
    maker: bool = typer.Option(True, help="Estimate fees at maker (else taker)"),
    live: bool = typer.Option(False, help="Place REAL orders (needs keys + --yes)"),
    yes: bool = typer.Option(False, help="Confirm real-money execution"),
) -> None:
    """Plan (and optionally execute) the delta-neutral basis carry: LONG spot + SHORT perp.

    Default is DRY-RUN: prints the exact book + expected funding income. No money moves
    without --live AND --yes AND Bybit keys in .env.
    """
    from .execution.carry_executor import execute_live, plan_carry, render_plan

    fee = 0.0002 if maker else 0.00055
    plan = plan_carry(
        venue=venue, capital=capital, top_n=top_n, min_funding=min_funding,
        leverage=leverage, fee_rate=fee,
    )
    typer.echo(render_plan(plan))

    if not live:
        typer.echo("\n[dry-run] No orders placed. Add --live --yes (with keys in .env) to execute.")
        return
    if not plan.legs:
        typer.echo("\nNothing to execute (no positive carry right now).")
        return
    if not yes:
        typer.echo("\n[blocked] --live needs explicit --yes. Refusing to spend real money.")
        return
    typer.echo("\n[LIVE] Placing real two-leg orders...")
    try:
        fills = execute_live(plan, confirm=True)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[error] {exc}")
        raise typer.Exit(code=1) from exc
    for f in fills:
        typer.echo(f"  {f}")


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
