"""Thin Alpaca client: option chains and stock snapshots.

Credentials come from the environment (APCA_API_KEY_ID / APCA_API_SECRET_KEY)
or from a `.env` file in this folder -- copy `.env.example` to `.env`.
Nothing here prints or returns them.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from typing import Dict, Iterable, List, Optional

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
ENV_PATH = os.path.join(ROOT, ".env")

DATA = "https://data.alpaca.markets"
TRADE = "https://paper-api.alpaca.markets"


def load_env(path: str = ENV_PATH) -> int:
    """Populate os.environ from the local .env file. Shell vars win."""
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v
                n += 1
    return n


def headers() -> Dict[str, str]:
    load_env()
    key = os.environ.get("APCA_API_KEY_ID", "")
    sec = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not sec:
        raise RuntimeError(
            f"No Alpaca credentials. Expected APCA_API_KEY_ID / "
            f"APCA_API_SECRET_KEY in the environment or in {ENV_PATH}")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec,
            "accept": "application/json"}


def _get(url: str, params: Optional[dict] = None, tries: int = 4) -> dict:
    for attempt in range(tries):
        r = requests.get(url, params=params, headers=headers(), timeout=60)
        if r.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            try:
                msg = r.json().get("message") or r.text[:300]
            except Exception:
                msg = r.text[:300]
            raise RuntimeError(f"Alpaca {r.status_code} on {url}: {msg}")
        return r.json()
    raise RuntimeError(f"Alpaca rate limited {tries}x on {url}")


# --------------------------------------------------------------------------- #
# disk cache -- a scan is re-run many times while the analysis changes, and the
# underlying quotes are a fixed snapshot of one close. Re-pulling them each time
# would be slow and, worse, would silently mix two different snapshots.
# --------------------------------------------------------------------------- #
def cached(name: str, fn, refresh: bool = False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.json")
    if os.path.exists(path) and not refresh:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    val = fn()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(val, fh)
    return val


# --------------------------------------------------------------------------- #
def assets(refresh: bool = False) -> List[dict]:
    """All active US equities, with has_options / borrow flags."""
    return cached("assets", lambda: _get(
        f"{TRADE}/v2/assets",
        {"status": "active", "asset_class": "us_equity"}), refresh)


def stock_snapshots(symbols: Iterable[str], chunk: int = 400) -> Dict[str, dict]:
    """Latest trade/quote/bars for many symbols, in chunks."""
    syms = list(symbols)
    out: Dict[str, dict] = {}
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        js = _get(f"{DATA}/v2/stocks/snapshots",
                  {"symbols": ",".join(part), "feed": "iex"})
        out.update(js.get("snapshots", js) or {})
        time.sleep(0.05)
    return out


def option_chain(underlying: str, feed: str = "indicative",
                 max_pages: int = 20) -> Dict[str, dict]:
    """Full option chain snapshot: every contract's latest NBBO + greeks."""
    out: Dict[str, dict] = {}
    token = None
    for _ in range(max_pages):
        p = {"feed": feed, "limit": 1000}
        if token:
            p["page_token"] = token
        js = _get(f"{DATA}/v1beta1/options/snapshots/{underlying}", p)
        out.update(js.get("snapshots") or {})
        token = js.get("next_page_token")
        if not token:
            break
        time.sleep(0.05)
    return out


# --------------------------------------------------------------------------- #
def occ_parse(symbol: str) -> dict:
    body = symbol.strip().upper()
    return {"underlying": body[:-15],
            "expiry": dt.datetime.strptime(body[-15:-9], "%y%m%d").date(),
            "right": body[-9],
            "strike": int(body[-8:]) / 1000.0}
