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
    from .data.exchange import make_exchange

    cfg = load_config()
    ex = make_exchange(cfg)
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
    from .data.exchange import make_exchange
    from .models import LightGBMModel
    from .pipeline.train import model_path, prepare_dataset

    cfg = load_config()
    ex = make_exchange(cfg)
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
def paper(once: bool = typer.Option(False, help="Run a single iteration then exit")) -> None:
    """Run the paper/testnet trading loop (no real money)."""
    from .pipeline.trade import run_paper_loop

    run_paper_loop(once=once)


if __name__ == "__main__":
    app()
