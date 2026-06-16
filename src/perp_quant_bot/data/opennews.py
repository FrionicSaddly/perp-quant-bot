"""Minimal sync client for the 6551 / OpenNews REST API (token from .env).

Used by the logger to accumulate a timestamped feature stream (news impact scores,
trading signals, market/funding/liquidation events) for FUTURE backtesting. The API
is real-time / recent only, so historical backtest needs data collected over time.
"""
from __future__ import annotations

import httpx

from ..config import load_secrets
from ..logging_conf import setup_logging

logger = setup_logging()

API_BASE = "https://ai.6551.io"


class OpenNewsClient:
    def __init__(self, token: str | None = None, base: str = API_BASE, timeout: float = 30.0):
        self.token = token or (load_secrets().opennews_token or "")
        if not self.token:
            raise RuntimeError("OPENNEWS_TOKEN not set (.env). Get one at https://6551.io/mcp")
        self.base = base.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def engine_tree(self) -> dict:
        r = self._client.get(f"{self.base}/open/news_type")
        r.raise_for_status()
        return r.json()

    def search(
        self,
        coins: list[str] | None = None,
        engine_types: dict[str, list[str]] | None = None,
        query: str | None = None,
        score: int | None = None,
        limit: int = 100,
        page: int = 1,
    ) -> dict:
        body: dict = {"limit": limit, "page": page}
        if coins:
            body["coins"] = coins
        if engine_types:
            body["engineTypes"] = engine_types
        if query:
            body["q"] = query
        if score is not None:
            body["score"] = score
        r = self._client.post(f"{self.base}/open/news_search", json=body)
        r.raise_for_status()
        return r.json()

    def latest(self, limit: int = 100, engine_types: dict[str, list[str]] | None = None) -> list[dict]:
        data = self.search(engine_types=engine_types, limit=limit, page=1).get("data")
        if isinstance(data, dict):
            data = data.get("list") or data.get("items")
        return data or []
