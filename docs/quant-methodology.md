# Accuracy-first methodology (non-negotiable)

The #1 priority is HONEST out-of-sample accuracy. Most "great" backtests are bugs.
Preserve these invariants in ANY change to features, labels, validation, or backtest.

## No lookahead / no leakage
- A feature at bar `t` may use data only up to `close[t]`. Never use `t+1..`.
- Rolling / ewm / `pct_change` look backward only; never apply negative shifts to inputs.
- Lower-frequency series (funding/OI) align by as-of / forward-fill — never future-fill.
- No target leakage: a feature must NOT derive from the label, a future return, or `t1`.

## Labels & validation
- Labels: triple-barrier (`labeling/`). Judge changes on OOS folds only — never in-sample.
- CV: purged walk-forward + embargo (`validation/walk_forward.py`). Purge training samples
  whose label end `t1` reaches into the test window.
- Fit scalers / encoders / statistics on the TRAIN fold only, then apply to test.

## Backtest realism
- Decide at `close[t]`, FILL on the next executable bar (`t+1`) — never the signal bar.
- Always include fees + funding + slippage. Compare vs buy-&-hold and a random/shuffled
  signal; if it can't beat those OOS after costs, the edge isn't real.

## Modeling discipline
- Prefer few robust features over many; beware overfitting and multiple-testing bias.
- Never tune hyperparameters on the test set. Report Sharpe / Sortino / MaxDD / hit-rate OOS.
- Risk sizing/limits (`risk/`) matter more than the model — keep them intact.

## Example: act next bar, don't leak the fill
```python
# BAD — uses the decision bar's own return
pnl = position * close.pct_change()
# GOOD — signal at t is applied to the t -> t+1 return
pos_used = position.shift(1)
pnl = pos_used * close.pct_change()
```

## Secrets
- API keys only from environment / `.env` (git-ignored). Never hardcode or commit keys.
