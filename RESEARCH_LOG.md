# Research log — what we tried and what the data actually said

Honest, dated notes. The point of this project is **out-of-sample truth**, so
negative results are recorded, not hidden. Every number below is purged
walk-forward OOS, after fees + slippage + funding, with a Deflated Sharpe (DSR)
that penalises multiple testing.

## 2026-06: directional 1h prediction + meta-labeling

**Goal:** raise directional accuracy on Bybit perps (BTC, ETH, 1h).

**What was built (all leak-safe, kept in the codebase, default off):**
- Richer perp features: funding z-score (72/168), funding-sign persistence,
  OI–price divergence (`features/microstructure.py`).
- Meta-labeling (`labeling/meta.py`): a parameter-free momentum/reversion
  **primary** picks the side; a secondary GBM predicts *whether to act*
  (P(side correct)); trade only above a probability threshold.
- Regime-conditioned OOS diagnostics (`pipeline/train.py: regime_breakdown`):
  hit-rate / mean next-bar return split by volatility and trend.
- A/B harness: `scripts/compare_meta.py`.

**Results (OOS, BTC + ETH, 1h):**

| symbol | mode | DSR | sharpe | hit | net ret | traded |
|---|---|---|---|---|---|---|
| ETH | directional | 0.09 | 0.51 | 47% | +63% | 5% |
| ETH | meta-momentum | 0.00 | -0.69 | 26% | -72% | 34% |
| ETH | meta-reversion | 0.00 | -1.18 | **60%** | **-86%** | 32% |
| BTC | directional | 0.00 | 0.53 | 41% | +67% | 30% |
| BTC | meta-momentum | 0.00 | -1.41 | 23% | -91% | 37% |
| BTC | meta-reversion | 0.00 | -1.62 | 54% | -93% | 30% |

**Verdict (confirmed on two symbols):**
1. **No robust directional edge on 1h OHLCV+funding+OI.** Every config sits at
   DSR ≈ 0 — indistinguishable from luck after costs and multiple testing.
2. **Hit-rate is a vanity metric.** Meta-reversion hit 54–60% and still lost
   86–93%: small frequent wins, rare large losses (negative skew), and fees
   dominate. We judge by DSR / net return, never raw hit-rate.
3. **No robust regime edge.** ETH showed a weak high-vol pocket; it did **not**
   replicate on BTC, so it was noise.

**Decision:** stop tuning 1h direction — more model tricks here would be
overfitting. The honest levers for *more* accuracy are **better data** (Bybit
microstructure: order-flow imbalance, liquidations, order-book depth, trade-level
CVD — which must be collected forward, like the OpenNews logger), not fancier
models on bar data. Meanwhile the one **validated** edge remains the
delta-neutral **basis carry** (`pqb basis`, `GO_LIVE.md`).
