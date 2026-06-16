"""Persist backtest artifacts: metrics JSON, equity CSV, and an optional PNG plot."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..logging_conf import setup_logging

logger = setup_logging()


def save_report(results: pd.DataFrame, metrics: dict, out_dir, name: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    metrics_path = out / f"{name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
            fh,
            indent=2,
        )
    paths["metrics"] = str(metrics_path)

    equity_path = out / f"{name}_equity.csv"
    results[["equity"]].to_csv(equity_path)
    paths["equity_csv"] = str(equity_path)

    # Optional plot (requires the [plot] extra: matplotlib)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        results["equity"].plot(ax=ax, title=f"{name} — equity curve")
        ax.set_ylabel("equity")
        fig.tight_layout()
        plot_path = out / f"{name}_equity.png"
        fig.savefig(plot_path, dpi=110)
        plt.close(fig)
        paths["plot"] = str(plot_path)
    except Exception as exc:  # noqa: BLE001
        logger.info("Plot skipped ({}). Install with: pip install -e \".[plot]\"", exc)

    logger.info("Report written: {}", ", ".join(paths.values()))
    return paths
