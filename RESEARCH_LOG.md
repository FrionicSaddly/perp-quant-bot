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

## 2026-06: order-flow + positioning on history (does microstructure predict?)

**Goal:** without waiting for live order-book data, test the microstructure
hypothesis NOW. Binance publishes historically: per-bar **taker-buy volume** in
klines (order-flow / CVD) and 5-min **metrics** (taker buy/sell ratio, top-trader
& retail long/short ratios, OI). `scripts/microstructure_history_test.py` adds
these to the technical features and compares on the same purged-WF OOS folds.

**Results (BTCUSDT, 2025-06 .. 2026-05, purged walk-forward):**

| bars | model | fee | DSR | hit | net ret |
|---|---|---|---|---|---|
| 5m | technical-only | taker | 0.00 | 46.6% | -69% |
| 5m | + order-flow | taker | 0.00 | 50.7% | -66% |
| 5m | + flow + positioning | taker | 0.00 | 54.0% | -74% |
| 5m | technical-only | maker | 0.00 | 51.0% | -58% |
| 5m | + order-flow | maker | 0.00 | 54.2% | -46% |
| 5m | + flow + positioning | maker | 0.01 | **59.1%** | -47% |
| 1h | + flow + positioning | taker | 0.00 | 49.8% | -55% |

**Verdict:**
1. **Microstructure carries real directional signal.** Adding order-flow then
   positioning lifts hit-rate **monotonically 51 -> 54 -> 59%** at 5m — the
   strongest predictive content found in this project.
2. **The signal is short-horizon.** At 1h the lift vanishes (hit ~50%). The edge
   lives at minutes, not hours.
3. **It still does not pay** — even at maker fees, DSR ~ 0 and returns are
   negative. A 59% hit with negative payoff skew + per-trade costs loses. Once
   more: **hit-rate is not edge.**

**Implication:** a standalone microstructure *direction* bot is unlikely to be
profitable at these costs. Realistic uses are as a **filter/overlay** or combined
with the (untestable-historically) **live order-book depth** now being collected.
Validated earner remains the basis carry. The live collectors keep running so the
depth angle and an overlay can be evaluated once enough varied-regime data exists.

## 2026-06: improving the validated edge + harder validation

**Carry concentration (top-K).** Broadened the Binance universe to 20 names and
added leak-safe top-K selection (hold the K richest-funding names each bar). On
4.4y at maker fees: full basket 27.7% total (~5.7%/yr, Sharpe 9.8); **top-8 lifts
to 35.1% (~7%/yr) at near-identical risk** (Sharpe 9.6, OOS-H2 10.8). top-3/5 raise
return more but lose Sharpe to rotation turnover. `pqb basis --top-k 8`. A real,
modest, market-neutral improvement to the money-maker.

**CPCV + PBO.** Upgraded validation: `combinatorial_purged_splits` gives a
DISTRIBUTION of OOS performance (all C(n,k) paths), and `probability_of_backtest_
overfitting` (CSCV) quantifies overfitting. `pqb cpcv` demo on ETH 1h directional:
15 paths -> mean Sharpe -0.25, median -0.36, only 40% of paths >0 -> the "no edge"
verdict is robust to path choice, not a single unlucky split.

**Microstructure: costs or skew? (decisive gross test.)** Re-ran the 5m flow +
positioning test at ZERO cost. Even gross it loses: technical -52%, +flow -36%,
+flow+positioning **-32% with a 60.7% hit-rate**. So the unprofitability is NOT
execution cost (maker/rebates won't fix a gross-negative strategy) — it's negative
payoff skew in the triple-barrier directional framing. The microstructure features
clearly help (hit 52->61%, loss halved, Sharpe -1.0->-0.4), confirming real signal,
but this construction can't monetize it. **Conclusion: use microstructure as a
filter/overlay (e.g. gate carry entries), not a standalone directional bot.**
