# Architecture & Design

This document is the deep dive: how the bot is structured, what to feed it, what
to connect for prediction and advanced analysis, and — critically — how to keep
the "accuracy" honest instead of fooling yourself.

---

## 0. The honest truth about "maximum prediction accuracy"

Price prediction on liquid crypto perps is hard for structural reasons:

- **Low signal-to-noise.** Most of a 1h/1d return is unpredictable noise.
- **Non-stationarity / regime change.** What worked in a bull trend dies in chop.
- **Adversarial & reflexive.** Other bots arbitrage away obvious patterns; your
  own orders move the book.
- **Fat tails & funding.** A few liquidation cascades dominate PnL.

So the goal is **not** a high accuracy number. The goal is a **small, stable,
out-of-sample edge** that survives **fees + funding + slippage**, combined with
**risk management** that keeps you alive through losing streaks. A model with 54%
hit-rate and disciplined sizing beats a "70% backtest" that was overfit and leaks
the future.

**Where accuracy is actually won or lost (in priority order):**

1. **No leakage / no lookahead** (most "great" backtests are bugs).
2. **Honest validation** (purged walk-forward, embargo, out-of-sample only).
3. **Realistic costs** (taker fees, funding payments, slippage, latency).
4. **Good labels** (triple-barrier beats naive "next-bar up/down").
5. **Risk & position sizing** (vol targeting, stops, max drawdown).
6. *Then* model sophistication (features, ensembles, deep nets).

Most beginners invert this list. We do not.

---

## 1. System overview

```
                ┌─────────────────────────────────────────────────────────┐
                │                      config/config.yaml                  │
                └─────────────────────────────────────────────────────────┘
                                          │
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  DATA LAYER  │──▶│   FEATURES   │──▶│   LABELING   │──▶│    MODELS    │
   │ ccxt OHLCV   │   │ technical +  │   │ triple-      │   │ LightGBM     │
   │ funding, OI  │   │ microstruct. │   │ barrier      │   │ (+seq stub)  │
   └──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                     │
        ┌────────────────────────────────────────────────────────────┘
        ▼                                   ▼
   ┌──────────────┐   purged walk-     ┌──────────────┐
   │  VALIDATION  │◀──forward CV──────▶│  BACKTEST    │  fees+funding+slippage
   └──────────────┘                    └──────┬───────┘
                                               │ metrics: Sharpe/Sortino/MaxDD/...
                                               ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │     RISK     │──▶│  EXECUTION   │──▶│  EXCHANGE    │  (paper / testnet)
   │ sizing/stops │   │ paper/testnet│   │   Bybit      │
   └──────────────┘   └──────────────┘   └──────────────┘
```

Two entry pipelines:

- **`pipeline/train.py`** — offline: data → features → labels → CV → fit → save.
- **`pipeline/trade.py`** — online loop: fetch latest → features → predict →
  risk → broker order. Runs in `paper` or `testnet` mode.

---

## 2. Data layer — what to feed it

`src/perp_quant_bot/data/`. Everything is timestamped **UTC**, stored as parquet,
and cached so you download once.

**Minimum viable (implemented):**

- **OHLCV** per symbol/timeframe (`data/ohlcv.py`). The backbone.
- **Funding rate history** (`data/funding.py`). Perp-specific; strong feature and
  a real PnL cost when holding positions.
- **Open interest history** (`data/funding.py`). Crowding / positioning proxy.

**High-value additions (connect as you grow — see §13):**

- **Order-book / L2 snapshots** (depth imbalance, spread) — best short-horizon edge.
- **Trades tape** (aggressor side, CVD = cumulative volume delta).
- **Liquidations** feed (cascade detection).
- **Cross-asset**: BTC dominance, ETH/BTC, total market cap, DXY, SPX, gold.
- **On-chain**: exchange net flows, stablecoin supply, active addresses.
- **News & sentiment**: the `opennews` MCP you installed (impact score + long/short
  signal), funding-rate anomalies, social volume.
- **Macro**: rates, CPI prints (the `us-gov-open-data` MCP / FRED).

**Data-quality rules (non-negotiable):**

- Deduplicate, sort by time, forward-fill only *non-anticipating* series.
- Align lower-frequency series (funding is 8h) by **as-of merge** (no future bleed).
- Store the **bar close timestamp**; a feature at bar `t` may only use info known
  at `close[t]`. Labels look forward; features never do.

---

## 3. Feature engineering — `features/`

Features are computed so that **feature row `t` uses only data up to `close[t]`**.

**`features/technical.py` (price/volume):**

- Log returns over each window in `features.windows`.
- Realized volatility (rolling std of log returns) per window.
- Momentum / ROC, distance from rolling mean (z-score).
- RSI, MACD (line/signal/hist), ATR (also used by labels & sizing).
- Bollinger %B and bandwidth.
- Volume z-score, dollar-volume, up/down volume ratio.
- Candle geometry: body/range, upper/lower wick fractions.
- Time-of-day / day-of-week cyclical encodings (sin/cos) — crypto has session effects.

**`features/microstructure.py` (perp-specific):**

- Funding rate level, rolling mean, and **funding momentum** (carry signal).
- Open-interest change & OI/price divergence (positioning).
- Basis (perp vs spot/index) when available.
- (Hook) order-book imbalance, CVD, liquidation intensity once you wire those feeds.

**`features/pipeline.py`:**

- Assembles the matrix, as-of merges micro features, drops warmup NaNs, and
  returns `X` (features) aligned to the OHLCV index plus the ATR series used
  downstream. Keeps an explicit **feature list** saved with the model so train
  and inference can never drift.

**Why this set:** it spans the predictive axes that actually matter on perps —
trend, mean-reversion, volatility regime, carry (funding), positioning (OI), and
seasonality — while staying cheap and leak-free.

---

## 4. Labeling — `labeling/triple_barrier.py`

Naive labels ("next bar up = 1") are noisy and ignore *how* you'd exit. We use the
**triple-barrier method** (López de Prado, *Advances in Financial ML*):

For each bar `t`, set three barriers over a forward window of `horizon_bars`:

- **Upper (profit-take):** `close[t] * (1 + pt_atr_mult * ATR[t]/close[t])`
- **Lower (stop-loss):** `close[t] * (1 - sl_atr_mult * ATR[t]/close[t])`
- **Vertical (time):** at `t + horizon_bars`.

The **first barrier touched** sets the label:

- touch upper first → `+1` (long pays)
- touch lower first → `-1` (short pays)
- neither → sign of the return at the vertical barrier, or `0` (neutral) if
  `|ret| < min_ret`.

ATR-scaled barriers adapt to volatility regime, so labels are comparable across
calm and wild markets. This produces a **3-class** target (`-1/0/1`) that mirrors
how the bot actually trades (take-profit / stop / timeout).

**Advanced (roadmap):** *meta-labeling* — a second model that predicts whether to
*act* on the primary signal (sizing/precision filter), and **sample-uniqueness
weights** to down-weight overlapping labels.

---

## 5. Models — `models/`

- **`models/gbm.py` — LightGBM (baseline & default).** Gradient-boosted trees are
  the workhorse for tabular financial features: robust to scale, capture
  nonlinearities/interactions, fast, and hard to beat OOS. 3-class softmax →
  per-class probabilities → signal via `prob_threshold` (only trade when the model
  is confident enough; otherwise stay flat).
- **`models/sequence.py` — optional LSTM/TCN/Transformer stub** (`[deep]` extra).
  Use only after the GBM baseline is honest and profitable on testnet; sequence
  nets overfit easily and need far more data and care.

**Beyond a single model (roadmap):**

- **Probability calibration** (isotonic/Platt) so thresholds mean what they say.
- **Ensembling** across symbols/timeframes/seeds; average or stack.
- **Per-regime models** (e.g., trend vs range gated by a volatility/Hurst filter).
- **Feature importance / SHAP** for sanity and pruning.

---

## 6. Validation — `validation/walk_forward.py`

This is where most "accurate" bots are exposed as fiction.

- **Purged walk-forward:** train on the past, test on the *future*, walking
  forward in `n_splits` folds. Never shuffle time-series.
- **Embargo:** drop `embargo_bars` between train and test so a label whose
  horizon overlaps the test window can't leak.
- **Purge:** remove training samples whose label end-time `t1` reaches into the
  test set.
- **Out-of-sample only:** report metrics on test folds, never on training data.

**Roadmap:** Combinatorial Purged Cross-Validation (CPCV) for a distribution of
OOS paths and a **Deflated Sharpe Ratio** to penalize multiple-testing /
backtest-overfitting.

**Leakage checklist (run it every time):**

- [ ] No feature uses `t+1..` data (check shifts).
- [ ] Scalers/encoders fit on train folds only.
- [ ] Labels' forward window purged + embargoed from test.
- [ ] No symbol uses another symbol's future (as-of merges only).
- [ ] Backtest fills use next-bar price, not the signal bar's close.

---

## 7. Backtest realism — `backtest/`

A signal is worthless until it survives costs. `backtest/engine.py` simulates:

- **Decision/fill timing:** decide at `close[t]`, **fill at `open/close[t+1]`**
  (no acting on a price you couldn't have traded).
- **Fees:** `fee_rate` per side on position changes (taker by default).
- **Slippage:** `slippage_bps` per side.
- **Funding:** pay/receive funding while holding across funding timestamps.
- **Equity curve & trade log**, then `metrics.py`: total/CAGR, **Sharpe, Sortino,
  max drawdown, Calmar**, hit-rate, avg win/loss, exposure, turnover.

Compare every result against **buy-and-hold** and a **random/shuffled-signal**
baseline. If you can't beat those OOS after costs, the edge isn't real.

---

## 8. Risk management — `risk/manager.py`

Risk logic is independent of the model and applies in both backtest and live:

- **Vol-targeted / ATR position sizing:** size so that hitting the stop loses
  `risk_per_trade` of equity (`risk_per_trade`, `atr_stop_mult`).
- **Leverage cap** (`max_leverage`) and **max concurrent positions**.
- **Stop-loss / take-profit** mirroring the label barriers.
- **Daily max loss** (`daily_max_loss`): stop opening new trades after a bad day.
- (Roadmap) portfolio vol target, correlation-aware exposure, kill-switch.

Position sizing matters **more than the model** for long-run survival.

---

## 9. Execution — `execution/`

- **`broker.py`** — abstract interface: balance, position, set_leverage, order.
- **`paper.py`** — `PaperBroker` (in-memory fills at last price; default, zero
  risk) and `CcxtBroker` for **Bybit testnet** (real API, play money).
- **`live.py`** — intentionally **disabled**; flipping it on is a deliberate,
  reviewed step (idempotent orders, reconciliation, rate limits, error handling).

---

## 10. From signal to order — the live loop (`pipeline/trade.py`)

```
every poll_seconds:
  1. fetch latest OHLCV (+ funding/OI) for each symbol
  2. build features for the most recent CLOSED bar
  3. model.predict_proba -> signal in {-1,0,1} (respecting prob_threshold)
  4. RiskManager: sizing, leverage cap, max positions, daily-loss guard
  5. reconcile target vs current position -> Broker.create_order (paper/testnet)
  6. log decision, fills, equity; persist state
```

Only acts on **closed** bars (never the in-progress candle). Crash-safe: state is
reloadable so a restart doesn't double-trade.

---

## 11. MLOps — keeping it alive

- **Retraining cadence** (e.g., weekly) on a rolling window; version each model.
- **Model registry:** save model + feature list + config hash + train window +
  OOS metrics in `models/`.
- **Drift monitoring:** feature distribution drift, live hit-rate vs backtest,
  realized vs expected slippage.
- **Alerting & kill-switch:** halt on drawdown breach, data gaps, or API errors.

---

## 12. Advanced analysis & alpha ideas (roadmap)

- **Order-flow:** book imbalance, CVD, absorption, liquidation cascades.
- **Cross-sectional:** rank many perps, long top / short bottom (market-neutral).
- **Carry:** funding-rate harvesting with delta hedges.
- **Regime detection:** HMM / volatility / Hurst gating which model is active.
- **Sentiment & events:** `opennews` impact scores, macro prints (`us-gov-open-data`).
- **Change-point / anomaly detection** on OI and funding for early regime flips.

---

## 13. Data sources to connect (mapping to your MCPs)

| Need | Source | Status |
|---|---|---|
| OHLCV, funding, OI | Bybit via `ccxt` | implemented |
| News + AI long/short signal, funding/liq/OI feeds | `opennews` MCP (6551) | hook (needs token) |
| Macro (rates, CPI, DXY) | `us-gov-open-data` MCP / FRED | hook (optional) |
| Order book / trades tape | exchange websocket | roadmap |
| On-chain flows | Glassnode/Dune-style API | roadmap |

The MCPs help you **research and prototype features** inside chat; the bot itself
reads data through its own `data/` clients so it can run headless 24/7.

---

## 14. Pitfalls that silently kill real accuracy

- Acting on the **signal bar's close** instead of the next executable price.
- Fitting scalers/feature stats on the **whole** dataset.
- Ignoring **funding** (it can exceed your gross edge on held positions).
- Optimizing hyperparameters on the **test** set (multiple-testing bias).
- Tiny samples + many features → overfit. Prefer fewer, robust features.
- Reporting **in-sample** or single-split results. Always OOS, always after costs.
- Forgetting **minimum notional / tick / lot** rounding at the exchange.
