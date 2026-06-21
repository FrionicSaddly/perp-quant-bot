# perp-quant-bot

Research-grade crypto **perpetual-futures** trading bot. End-to-end pipeline:
market data → feature engineering → ML signal → realistic backtest → risk-managed
**paper/testnet** execution. Built for **Bybit** (via `ccxt`), exchange-agnostic by design.

> **Mode:** paper / testnet only. There is no enabled live-trading path in this
> codebase. Live execution is a guarded stub you must deliberately implement.

---

## Honest expectations (read this first)

No bot guarantees "maximum accuracy" or profit. Crypto perp markets are noisy,
near-efficient, adversarial, and full of regime changes. A realistic, well-built
ML signal lands around **52–56% directional hit-rate**; the edge comes from
**risk management, costs control, and not blowing up**, not from a magic model.

This project is engineered to **maximize robustness and avoid the things that
destroy real-world accuracy**: lookahead bias, data leakage, overfitting,
ignoring fees/funding/slippage, and survivorship in validation. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design and the "how to actually
improve edge" roadmap.

Futures use leverage. You can lose everything (and on some venues more than your
deposit). Trade testnet until your **out-of-sample** results are stable.

---

## Quickstart

```bash
# 1. Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the package (editable)
pip install -e .

# 3. (optional) copy env template and add Bybit TESTNET keys
copy .env.example .env    # then edit .env

# 4. Sanity-check the environment & config
pqb doctor

# 5. Download historical data for the configured universe
pqb download

# 6. Train models with purged walk-forward validation
pqb train

# 7. Backtest the trained signal with fees + funding + slippage
pqb backtest

# 8. Run the paper-trading loop (no real money)
pqb paper
```

All behavior is driven by [`config/config.yaml`](config/config.yaml).

---

## Project layout

```
perp-quant-bot/
├── config/config.yaml          # all knobs: universe, features, model, risk, backtest
├── ARCHITECTURE.md             # detailed design (the deep dive)
├── src/perp_quant_bot/
│   ├── config.py               # typed config + secrets loader
│   ├── data/                   # ccxt exchange, OHLCV + funding/OI download
│   ├── features/               # technical + microstructure feature builders
│   ├── labeling/               # triple-barrier labels
│   ├── models/                 # LightGBM baseline (+ sequence-model stub)
│   ├── validation/             # purged walk-forward CV
│   ├── backtest/               # vectorized backtester + metrics
│   ├── risk/                   # position sizing + risk limits
│   ├── execution/              # paper / testnet brokers
│   ├── pipeline/               # train.py + trade.py orchestration
│   └── cli.py                  # `pqb` command-line interface
├── tests/                      # smoke tests
└── data/, models/              # artifacts (gitignored)
```

## Commands

| Command | What it does |
|---|---|
| `pqb doctor` | Check Python, deps, config, and (optional) exchange keys |
| `pqb download` | Fetch OHLCV (+ funding/OI) for the universe → parquet |
| `pqb train` | Build features+labels, run walk-forward CV, fit + save models |
| `pqb backtest` | Simulate the signal with realistic costs; saves metrics/equity report |
| `pqb leakcheck` | Empirical leak detector: clean vs shuffled-labels vs injected future leak |
| `pqb paper` | Live paper/testnet loop with daily-loss + data-staleness guards |
| `pqb basis` | Backtest the delta-neutral basis carry (long spot + short perp) — the one validated edge |
| `pqb funding-now` | Live funding snapshot: where the carry is right now (positive ⇒ long spot + short perp) |
| `pqb carry-trade` | Plan the live carry book (dry-run by default); real orders only with `--live --yes` + keys. See `GO_LIVE.md` |
| `pqb microlog` | Collect perp microstructure (order-book imbalance, CVD, OI, funding) → daily CSV. The short-horizon data bar OHLCV lacks; run on an always-on host to accumulate history |

Methodology guardrails for features, labels, validation, and backtests live in
[`docs/quant-methodology.md`](docs/quant-methodology.md).

## License

MIT. Provided for research/education. **Not financial advice.** Use at your own risk.
