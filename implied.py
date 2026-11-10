"""Event-implied moves from live option chains (Alpaca indicative snapshot).

For an event on day E, take the expiry just before E (T1) and the first on or
after it (T2). Forward variance between them = event variance + ordinary
variance for the other days in the gap:

    var_event = iv2^2 T2 - iv1^2 T1 - iv1^2 (T2 - T1 - 1/252)

using iv1 as the ordinary daily level. sqrt(var_event) is the one-standard-
deviation move the market prices for the event day itself. SPY lists an expiry
every weekday, so its macro events are isolated almost exactly; single names
list Mon/Wed/Fri or weekly, which is enough to bracket an earnings report.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import alpaca

ROOT = os.path.dirname(os.path.abspath(__file__))

ASOF = dt.date(2026, 9, 25)          # quotes are Friday's close
R = 0.042                             # ~3-month T-bill; small effect at these tenors


def bs(S, K, T, sig, call):
    if T <= 0 or sig <= 0:
        return max(S - K, 0) if call else max(K - S, 0)
    d1 = (math.log(S / K) + (R + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)
    return K * math.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def iv(price, S, K, T, call):
    intrinsic = max(S - K, 0) if call else max(K - S, 0)
    if price <= intrinsic + 1e-4 or T <= 0:
        return np.nan
    try:
        return brentq(lambda s: bs(S, K, T, s, call) - price, 1e-3, 5.0)
    except ValueError:
        return np.nan


def atm_table(sym: str, spot: float) -> pd.DataFrame:
    chain = alpaca.option_chain(sym)
    rows = []
    for occ, snap in chain.items():
        m = alpaca.occ_parse(occ)
        q = (snap or {}).get("latestQuote") or {}
        b, a = q.get("bp"), q.get("ap")
        if not b or not a or b <= 0 or a < b:
            continue
        rows.append((m["expiry"], m["strike"], m["right"], 0.5 * (b + a), (a - b) / (0.5 * (a + b))))
    df = pd.DataFrame(rows, columns=["expiry", "strike", "right", "mid", "rel"])
    out = []
    for e, g in df.groupby("expiry"):
        T = max((e - ASOF).days, 0.5) / 365.0
        ks = sorted(set(g[g.right == "C"].strike) & set(g[g.right == "P"].strike))
        if not ks:
            continue
        k = min(ks, key=lambda x: abs(x - spot))
        c = g[(g.strike == k) & (g.right == "C")].iloc[0]
        p = g[(g.strike == k) & (g.right == "P")].iloc[0]
        ivs = [iv(c.mid, spot, k, T, True), iv(p.mid, spot, k, T, False)]
        ivs = [x for x in ivs if np.isfinite(x)]
        if not ivs or max(c.rel, p.rel) > 0.25:
            continue
        out.append({"expiry": e, "tyr": T, "strike": k, "straddle": c.mid + p.mid,
                    "iv": float(np.mean(ivs))})
    return pd.DataFrame(out).sort_values("expiry").reset_index(drop=True)


def event_move(tab: pd.DataFrame, event_day: dt.date) -> dict | None:
    """1-sd implied move for the session `event_day` (the reaction day)."""
    before = tab[tab.expiry < event_day]
    after = tab[tab.expiry >= event_day]
    if not len(before) or not len(after):
        return None
    a, b = before.iloc[-1], after.iloc[0]
    fwd = b.iv ** 2 * b.tyr - a.iv ** 2 * a.tyr
    ordinary = a.iv ** 2 * max(b.tyr - a.tyr - 1 / 252, 0)
    ev = fwd - ordinary
    if ev <= 0:
        return {"move_pct": 0.0, "e1": str(a.expiry), "e2": str(b.expiry), "iv1": a.iv, "iv2": b.iv}
    return {"move_pct": 100 * math.sqrt(ev), "e1": str(a.expiry), "e2": str(b.expiry),
            "iv1": a.iv, "iv2": b.iv, "gap_days": (b.expiry - a.expiry).days}


def spot_of(sym: str) -> float:
    s = alpaca.stock_snapshots([sym])[sym]
    return float((s.get("dailyBar") or {}).get("c") or s["latestTrade"]["p"])


EVENTS_SPY = {"Jobs report (Sep)": dt.date(2026, 10, 2), "CPI (Sep)": dt.date(2026, 10, 14),
              "FOMC": dt.date(2026, 10, 28), "Hyperscaler earnings reaction": dt.date(2026, 10, 29),
              "Midterm election reaction": dt.date(2026, 11, 4), "Jobs report (Oct)": dt.date(2026, 11, 6),
              "CPI (Oct)": dt.date(2026, 11, 10), "Nvidia earnings reaction": dt.date(2026, 11, 18)}

EARN = {"MSFT": dt.date(2026, 10, 29), "GOOGL": dt.date(2026, 10, 29), "META": dt.date(2026, 10, 29),
        "AMZN": dt.date(2026, 10, 30), "AAPL": dt.date(2026, 10, 30), "NVDA": dt.date(2026, 11, 18),
        "ORCL": dt.date(2026, 12, 11), "AVGO": dt.date(2026, 12, 10), "TSM": dt.date(2026, 10, 15),
        "AMD": dt.date(2026, 11, 4), "UNH": dt.date(2026, 10, 13), "LLY": dt.date(2026, 10, 29),
        "JNJ": dt.date(2026, 10, 13), "NEE": dt.date(2026, 10, 27), "PLD": dt.date(2026, 10, 15)}

SECTORS = ["SPY", "QQQ", "XLK", "SMH", "XLV", "XLU", "XLRE", "XLE", "USO", "TLT"]


def main():
    res = {"asof": str(ASOF), "spy_events": {}, "earnings": {}, "sectors": {}}
    spot = spot_of("SPY")
    tab = atm_table("SPY", spot)
    tab.to_csv(os.path.join(ROOT, "cache", "spy_atm.csv"), index=False)
    print(f"SPY {spot:.2f}: {len(tab)} expiries")
    for name, d in EVENTS_SPY.items():
        r = event_move(tab, d)
        res["spy_events"][name] = r | {"date": str(d)} if r else None
        if r:
            print(f"  {name:<30} {d}  1-sd {r['move_pct']:.2f}%  (${spot * r['move_pct'] / 100:.2f})  {r['e1']}->{r['e2']}")
    # base: median 1-sd implied move of ordinary SPY days in the next 3 weeks
    for sym, d in EARN.items():
        try:
            sp = spot_of(sym)
            t = atm_table(sym, sp)
            r = event_move(t, d)
            res["earnings"][sym] = (r or {}) | {"date": str(d), "spot": sp}
            if r:
                print(f"  {sym:<6} earnings reaction {d}: 1-sd {r['move_pct']:.2f}%  {r['e1']}->{r['e2']}")
        except Exception as ex:                                  # noqa: BLE001
            print(sym, "ERR", ex)
    for sym in SECTORS:
        try:
            sp = spot_of(sym)
            t = atm_table(sym, sp)
            t["days"] = [(e - ASOF).days for e in t.expiry]
            pick = {}
            for label, target in (("1m", 30), ("3m", 91)):
                i = (t.days - target).abs().idxmin()
                pick[label] = {"expiry": str(t.expiry.at[i]), "iv": float(t.iv.at[i])}
            res["sectors"][sym] = {"spot": sp} | pick
            print(f"  {sym:<5} spot {sp:>8.2f}  IV 1m {100 * pick['1m']['iv']:.1f}%  3m {100 * pick['3m']['iv']:.1f}%")
        except Exception as ex:                                  # noqa: BLE001
            print(sym, "ERR", ex)
    with open(os.path.join(ROOT, "results", "implied.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
