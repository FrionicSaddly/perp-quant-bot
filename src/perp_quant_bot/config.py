"""Typed configuration and secrets loading.

`config/config.yaml` holds all non-secret knobs; `.env` holds API keys.
Everything in the codebase reads settings through :func:`load_config`.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeCfg(BaseModel):
    id: str = "bybit"
    testnet: bool = True
    market_type: str = "linear"
    quote: str = "USDT"


class UniverseCfg(BaseModel):
    symbols: list[str]
    timeframe: str = "1h"
    since: str = "2022-01-01T00:00:00Z"


class DataCfg(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class FeaturesCfg(BaseModel):
    windows: list[int] = [6, 12, 24, 48, 72, 168]
    regime_windows: list[int] = [24, 168]  # windows for trend/regime stats (kept small)
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    include_funding: bool = True
    include_open_interest: bool = True
    use_cross_asset: bool = True
    anchor_symbol: str = "BTC/USDT:USDT"  # market driver; alts get its return/vol as features


class LabelingCfg(BaseModel):
    method: str = "triple_barrier"
    horizon_bars: int = 24
    pt_atr_mult: float = 2.0
    sl_atr_mult: float = 1.5
    min_ret: float = 0.0


class ModelCfg(BaseModel):
    type: str = "lightgbm"
    prob_threshold: float = 0.40
    params: dict = Field(default_factory=dict)


class ValidationCfg(BaseModel):
    scheme: str = "purged_walk_forward"
    n_splits: int = 5
    embargo_bars: int = 24


class BacktestCfg(BaseModel):
    fee_rate: float = 0.00055
    slippage_bps: float = 2.0
    apply_funding: bool = True
    initial_capital: float = 10000.0
    fill: str = "next_open"  # next_open (realistic) | close (optimistic approximation)


class RiskCfg(BaseModel):
    risk_per_trade: float = 0.01
    atr_stop_mult: float = 1.5
    max_leverage: float = 3.0
    max_positions: int = 2
    daily_max_loss: float = 0.05


class ExecutionCfg(BaseModel):
    mode: str = "paper"  # paper | testnet | live(disabled)
    poll_seconds: int = 60
    order_type: str = "market"


class Config(BaseModel):
    exchange: ExchangeCfg
    universe: UniverseCfg
    data: DataCfg
    features: FeaturesCfg
    labeling: LabelingCfg
    model: ModelCfg
    validation: ValidationCfg
    backtest: BacktestCfg
    risk: RiskCfg
    execution: ExecutionCfg

    project_root: Path = Field(default=Path("."), exclude=True)

    def path(self, *parts: str) -> Path:
        """Resolve a path relative to the project root."""
        return self.project_root.joinpath(*parts)

    def raw_dir(self) -> Path:
        p = self.path(self.data.raw_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def processed_dir(self) -> Path:
        p = self.path(self.data.processed_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def models_dir(self) -> Path:
        p = self.path("models")
        p.mkdir(parents=True, exist_ok=True)
        return p


class Secrets(BaseSettings):
    """API keys, read from environment / .env (never committed)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bybit_api_key: str | None = None
    bybit_api_secret: str | None = None
    opennews_token: str | None = None
    fred_api_key: str | None = None


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until we find pyproject.toml (the project root)."""
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cur


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the YAML config.

    If *path* is None we look for ``config/config.yaml`` under the project root.
    """
    root = find_project_root()
    cfg_path = Path(path).resolve() if path else (root / "config" / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(**raw)
    cfg.project_root = cfg_path.parent.parent
    return cfg


def load_secrets() -> Secrets:
    return Secrets()
