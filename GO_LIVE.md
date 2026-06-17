# Going live — honest checklist

Read this before risking real money. It is deliberately blunt.

## What does and does NOT have an edge

- ❌ **Directional prediction** (`pqb train` / `pqb paper`, the LightGBM signal): **no validated edge.**
  Proven here — out-of-sample DSR ≈ 0, hit-rate < 50%, and `pqb leakcheck` confirms the
  pipeline is honest (the result is real, not a bug). **Do NOT trade this with real money.**
  It will pay fees to lose.
- ✅ **Basis carry** (`pqb basis`): the one validated edge. Delta-neutral **LONG spot + SHORT perp**
  to harvest funding. On 4.4y of Binance data: gross Sharpe ~11, **net Sharpe ~7.8 even at taker
  fees, DSR = 1.0, positive out-of-sample in both halves.** Return is ~all funding (price hedged).

## What the carry actually is (set expectations)

- It is **market-neutral arbitrage**, not a futures-direction bot.
- Historical return ≈ **5–7%/yr** (low volatility → high Sharpe, but modest absolute return).
- It is a **known, competitive trade**. Funding compresses as capital enters it — recent funding
  is lower than 2022–2024. The edge is real but **decaying**; monitor it (`pqb funding-now`).

## Requirements to run it for real

1. **An accessible exchange with spot + USDT perp + funding.** Bybit mainnet is reachable from
   your network (testnet is geo-blocked). Bybit has both spot and perps — workable. (OKX/MEXC are
   alternatives.)
2. **API keys** with **trade** permission, ideally on a **sub-account**, with an **IP allowlist**
   and **withdrawals disabled**. Put them in `.env` (git-ignored, never committed).
3. **Capital on BOTH legs**: cash for the spot long + margin for the perp short. The legs hedge
   each other on price; you collect funding.
4. **Start tiny** (smallest size), confirm fills/funding/PnL for days, then scale.

## Real risks (not modeled fully in the backtest)

- **Funding turns negative** in bear/neutral regimes → the carry stops; the bot must sit out
  (it only engages when funding > 0).
- **Liquidation / margin** on the short perp if not enough margin buffer (price spikes). Keep a
  large margin buffer; the long spot offsets PnL but margin is per-venue.
- **Execution / fills**: taker fees already make it thin; maker fills aren't guaranteed.
- **Exchange counterparty risk**, API outages, the (occasionally flaky) Bybit spot endpoint.
- **Edge decay**: more capital chasing carry shrinks it.

## Suggested path

1. `copy .env.example .env` → add Bybit keys (trade perms, sub-account, IP allowlist, no withdrawal).
2. `pqb funding-now --venue bybit` → see where the carry is right now.
3. Paper-run the carry first; verify it behaves.
4. Tiny live size; watch fills, funding accrual, margin for several days.
5. Scale only if it behaves as expected. Keep a kill-switch and daily-loss limit.

## Running the carry on Bybit (`pqb carry-trade`)

The executor is **delta-neutral** (LONG spot + SHORT perp), **reconciles** to a target
book (idempotent — safe to re-run; it trades only the delta), keeps a **margin buffer**
idle, **caps leverage at 5x**, and **exits a leg when its funding flips negative**.

```
# 1) DRY-RUN (default): prints the target book + exact reconcile orders, no keys needed
pqb carry-trade --capital 200 --leverage 2 --top-n 5

# 2) LIVE, tiny, once (needs Bybit keys in .env):
pqb carry-trade --capital 200 --leverage 2 --margin-buffer 0.25 --live --yes

# 3) LIVE loop (rebalance every hour), maker orders to cut fees:
pqb carry-trade --capital 200 --leverage 2 --maker --interval 3600 --live --yes
```

Knobs: `--capital` (USDT), `--leverage` (≤5), `--margin-buffer` (idle fraction, default
0.25), `--top-n` (richest-funding names), `--min-funding` (engage threshold),
`--maker` (post-only limits — cheaper, but fills aren't guaranteed → brief leg risk).

**Bybit-specific cautions:**
- The Bybit **spot** endpoint is intermittently geo-blocked from some networks (we hit
  403s). The carry NEEDS the spot leg — run the live loop on a **reliable host** (cloud
  VM / where Bybit is fully reachable), not a flaky connection.
- Use a **Unified Trading Account** so spot + linear perp share margin (cleaner hedge).
- Market orders guarantee the hedge fills together; `--maker` saves fees but can leave
  one leg unfilled — only use it if you monitor and accept brief delta.
- Start at the **smallest size**, confirm funding actually accrues and margin is safe
  for several days, then scale. Leverage multiplies funding-flip and liquidation risk.

**Not financial advice. You own the risk. Crypto leverage can lose your capital.**
