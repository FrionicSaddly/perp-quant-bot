# Security

This is a **research / portfolio** project. It is designed so that **no secrets
ever live in the repository**.

## Threat model in one line

Your money lives behind your exchange API keys. If keys never enter git, a public
repo cannot leak them. This project keeps keys out of git by construction.

## How secrets are handled

- **Keys live only in `.env`**, which is git-ignored. The repo ships `.env.example`
  with empty placeholders — copy it to `.env` and fill it locally:
  ```bash
  copy .env.example .env   # Windows
  ```
- **No key is ever read from code or config** — only from environment variables
  (`src/perp_quant_bot/config.py::Secrets`).
- `.gitignore` blocks `.env`, `*.key`, `*.pem`, `*.p12`, `credentials.*`,
  `secrets.*`, plus all data and model artifacts.
- A **local `pre-commit` hook** (`.git/hooks/pre-commit`) refuses any commit that
  contains a secret-looking file or an inline API key/secret. It is a safety net,
  not a replacement for care.

## Use testnet first

- Configure **Bybit testnet** keys (https://testnet.bybit.com), not mainnet, while
  developing. Testnet keys control only play money.
- This codebase has **no enabled live-trading path**; live execution is a guarded
  stub you must implement deliberately.

## Recommended key hygiene

- Create API keys with the **minimum permissions** needed (no withdrawal rights).
- Restrict keys by **IP allowlist** on the exchange where possible.
- Use a **separate sub-account** for the bot.
- **Rotate keys** periodically and immediately if you suspect exposure.

## If a key is ever exposed

1. **Revoke / rotate the key on the exchange immediately** (this is what actually
   protects you — deleting the commit is not enough).
2. Remove it from history (`git filter-repo` / BFG) and force-push if already
   pushed.
3. Generate fresh keys with restricted permissions.

## Reporting

This is a personal project; open a private issue or contact the repo owner for
security concerns. Do not include real keys in any report.
