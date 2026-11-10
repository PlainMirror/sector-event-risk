"""Re-derive event moves from the expiries immediately around each event.

The full-chain snapshot is paginated and came back with gaps, so events in the
same week shared one bracket. Here each candidate expiry within a week of the
event is requested on its own (expiration_date filter, strikes within 5% of
spot), which recovers the daily / Mon-Wed-Fri expiries that sit right against
the event.
"""
import datetime as dt, json, os, time
import numpy as np, pandas as pd
import implied as I
import alpaca

def expiry_quotes(sym, spot, d):
    js = alpaca._get(f"{alpaca.DATA}/v1beta1/options/snapshots/{sym}",
                     {"feed": "indicative", "limit": 1000, "expiration_date": d.isoformat(),
                      "strike_price_gte": round(spot * 0.95, 2), "strike_price_lte": round(spot * 1.05, 2)})
    return js.get("snapshots") or {}

def atm_iv(sym, spot, d):
    snaps = expiry_quotes(sym, spot, d)
    if not snaps:
        return None
    rows = []
    for occ, s in snaps.items():
        m = alpaca.occ_parse(occ); q = s.get("latestQuote") or {}
        b, a = q.get("bp"), q.get("ap")
        if b and a and b > 0 and a >= b:
            rows.append((m["strike"], m["right"], (a + b) / 2, (a - b) / ((a + b) / 2)))
    df = pd.DataFrame(rows, columns=["k", "r", "mid", "rel"])
    ks = sorted(set(df[df.r == "C"].k) & set(df[df.r == "P"].k))
    if not ks:
        return None
    k = min(ks, key=lambda x: abs(x - spot))
    T = max((d - I.ASOF).days, 0.5) / 365
    c = df[(df.k == k) & (df.r == "C")].iloc[0]; p = df[(df.k == k) & (df.r == "P")].iloc[0]
    v = [x for x in (I.iv(c.mid, spot, k, T, True), I.iv(p.mid, spot, k, T, False)) if np.isfinite(x)]
    if not v or max(c.rel, p.rel) > 0.25:
        return None
    return {"expiry": d, "tyr": T, "iv": float(np.mean(v)), "straddle": c.mid + p.mid, "k": k}

def bracket(sym, spot, event, span=8):
    rows = []
    for off in range(-span, span + 1):
        d = event + dt.timedelta(days=off)
        if d.weekday() >= 5 or d <= I.ASOF:
            continue
        r = atm_iv(sym, spot, d)
        if r:
            rows.append(r)
        time.sleep(0.05)
    tab = pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True) if rows else pd.DataFrame()
    return I.event_move(tab, event) if len(tab) else None, tab

EV = {"SPY": {"Jobs report (Sep)": dt.date(2026,10,2), "CPI (Sep)": dt.date(2026,10,14),
              "FOMC": dt.date(2026,10,28), "Hyperscaler earnings (MSFT/GOOGL/META)": dt.date(2026,10,29),
              "AMZN/AAPL earnings": dt.date(2026,10,30), "Midterm election": dt.date(2026,11,4),
              "Jobs report (Oct)": dt.date(2026,11,6), "CPI (Oct)": dt.date(2026,11,10),
              "Nvidia earnings": dt.date(2026,11,18)},
      "QQQ": {"Hyperscaler earnings (MSFT/GOOGL/META)": dt.date(2026,10,29), "Nvidia earnings": dt.date(2026,11,18),
              "Midterm election": dt.date(2026,11,4)}}
EV.update({s: {"earnings": d} for s, d in I.EARN.items()})

out = {}
for sym, evs in EV.items():
    spot = I.spot_of(sym)
    for name, d in evs.items():
        r, tab = bracket(sym, spot, d)
        out[f"{sym}|{name}"] = (r or {}) | {"sym": sym, "event": name, "date": str(d), "spot": spot}
        if r:
            print(f"{sym:<5} {name:<40} {d}  1-sd {r['move_pct']:5.2f}%  ${spot*r['move_pct']/100:7.2f}   {r['e1']} -> {r['e2']}", flush=True)
        else:
            print(f"{sym:<5} {name:<40} {d}  no bracket", flush=True)
json.dump(out, open("results/implied_precise.json", "w"), indent=1, default=str)
