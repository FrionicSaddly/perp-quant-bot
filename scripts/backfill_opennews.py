"""Backfill OpenNews history by paginating the API, merge into the opennews-data
branch parquet (dedup by id), and push it back via git plumbing (keeps main clean).

Re-runnable. Reads OPENNEWS_TOKEN from .env. Run from the repo root after
`git fetch origin opennews-data`.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

API_BASE = "https://ai.6551.io"
BRANCH = "opennews-data"
DATA_FILE = "opennews_log.parquet"


def _record(it: dict, now: str) -> dict:
    ai = it.get("aiRating") or {}
    coins = it.get("coins")
    raw = {k: v for k, v in it.items() if k != "text"}
    return {
        "id": str(it.get("id")),
        "ts": it.get("ts"),
        "engineType": it.get("engineType"),
        "newsType": it.get("newsType"),
        "source": it.get("source"),
        "score": it.get("score"),
        "ai_signal": ai.get("signal"),
        "ai_grade": ai.get("grade"),
        "ai_score": ai.get("score"),
        "coins_json": json.dumps(coins, ensure_ascii=False) if coins is not None else None,
        "raw": json.dumps(raw, ensure_ascii=False),
        "text": (it.get("text") or it.get("description") or "")[:300],
        "logged_at": now,
    }


def _existing_branch_df() -> pd.DataFrame:
    out = subprocess.run(["git", "show", f"origin/{BRANCH}:{DATA_FILE}"], capture_output=True)
    if out.returncode == 0 and out.stdout:
        try:
            return pd.read_parquet(io.BytesIO(out.stdout))
        except Exception as exc:  # noqa: BLE001
            print("could not read existing branch file:", exc, file=sys.stderr)
    return pd.DataFrame()


def _push(parquet_path: str, message: str) -> None:
    blob = subprocess.run(
        ["git", "hash-object", "-w", parquet_path], capture_output=True, text=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "mktree"], input=f"100644 blob {blob}\t{DATA_FILE}\n", capture_output=True, text=True
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", "--verify", f"origin/{BRANCH}"], capture_output=True, text=True
    ).stdout.strip()
    args = ["git", "commit-tree", tree, "-m", message]
    if parent:
        args = ["git", "commit-tree", tree, "-p", parent, "-m", message]
    commit = subprocess.run(args, capture_output=True, text=True).stdout.strip()
    push = subprocess.run(
        ["git", "push", "origin", f"{commit}:refs/heads/{BRANCH}"], capture_output=True, text=True
    )
    print(f"push rc={push.returncode}: {(push.stderr or push.stdout).strip()[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    token = os.environ.get("OPENNEWS_TOKEN", "")
    if not token:
        print("OPENNEWS_TOKEN not set", file=sys.stderr)
        return 1
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    now = pd.Timestamp.now(tz="UTC").isoformat()

    existing = _existing_branch_df()
    seen = set(existing["id"].astype(str)) if (not existing.empty and "id" in existing) else set()
    print(f"existing branch rows: {len(existing)}")

    recs: list[dict] = []
    empty_streak = 0
    with httpx.Client(timeout=30.0) as c:
        for page in range(1, args.max_pages + 1):
            try:
                r = c.post(
                    f"{API_BASE}/open/news_search",
                    headers=headers,
                    json={"limit": args.limit, "page": page},
                )
                r.raise_for_status()
                items = r.json().get("data") or []
            except Exception as exc:  # noqa: BLE001
                print(f"page {page} error: {exc}")
                break
            if not items:
                print(f"page {page}: empty -> stop")
                break
            new = 0
            for it in items:
                rid = str(it.get("id"))
                if rid and rid not in seen and rid != "None":
                    seen.add(rid)
                    recs.append(_record(it, now))
                    new += 1
            print(f"page {page}: {len(items)} items, +{new} new")
            empty_streak = empty_streak + 1 if new == 0 else 0
            if empty_streak >= 2:
                print("no new items for 2 pages -> stop")
                break
            time.sleep(0.3)

    merged = pd.concat([existing, pd.DataFrame(recs)], ignore_index=True) if recs else existing
    try:
        t = pd.to_datetime(merged["ts"], format="ISO8601", utc=True, errors="coerce")
        print(f"merged rows: {len(merged)} | backfilled +{len(recs)} | ts span: {t.min()} -> {t.max()}")
    except Exception:  # noqa: BLE001
        print(f"merged rows: {len(merged)} | backfilled +{len(recs)}")

    out = "_backfill_opennews.parquet"
    merged.to_parquet(out)
    if args.push and len(merged):
        _push(out, f"backfill opennews +{len(recs)} ({now})")
    if os.path.exists(out):
        os.remove(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
