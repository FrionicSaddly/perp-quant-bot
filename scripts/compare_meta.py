"""Honest A/B: directional model vs meta-labeling on the same OOS folds."""
from __future__ import annotations

import copy
import sys

from perp_quant_bot.config import load_config
from perp_quant_bot.pipeline.train import train_symbol


def run(symbol: str) -> None:
    base = load_config()
    rows = []
    configs = [
        ("directional", False, None),
        ("meta-mom", True, "momentum"),
        ("meta-rev", True, "reversion"),
    ]
    for label, meta, kind in configs:
        cfg = copy.deepcopy(base)
        cfg.model.meta_labeling = meta
        if kind:
            cfg.model.primary_kind = kind
        res = train_symbol(cfg, symbol, exchange=None)
        m = res["meta"]
        om = m["oos_metrics"]
        rows.append({
            "mode": label,
            "DSR": m["deflated_sharpe"],
            "sharpe": om.get("sharpe"),
            "hit": om.get("hit_rate"),
            "ret": om.get("total_return"),
            "traded": m.get("oos_pct_traded"),
            "regimes": m.get("oos_regimes", {}),
        })

    print("\n================ HONEST OOS COMPARISON:", symbol, "================")
    print(f"{'mode':<12}{'DSR':>7}{'sharpe':>8}{'hit':>7}{'ret':>9}{'traded':>8}")
    for r in rows:
        print(f"{r['mode']:<12}{r['DSR']:>7.2f}{(r['sharpe'] or 0):>8.2f}"
              f"{(r['hit'] or 0):>7.1%}{(r['ret'] or 0):>9.1%}{(r['traded'] or 0):>8.1%}")
    for r in rows:
        reg = r["regimes"]
        if reg:
            print(f"\n{r['mode']} hit-rate by regime:")
            for k, v in reg.items():
                print(f"  {k:<10} n={v['n']:<6} hit={v['hit_rate']}  meanRet={v['mean_ret_bps']}bps")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "ETH/USDT:USDT")
