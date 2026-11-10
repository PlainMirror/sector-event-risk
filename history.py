"""How each sector has reacted to scheduled macro events, 2015-2026.

Spike = a day whose return is at least 2 standard deviations of the sector's
trailing 20-day volatility (known before the day). For each sector and event
type: the spike rate with a Jeffreys Beta(0.5, 0.5) posterior 90% interval,
the posterior probability that event days spike more often than ordinary days,
and the variance multiplier -- mean squared standardised return on event days
over ordinary days, with a bootstrap 90% interval. The multiplier is what the
Monte Carlo in spikes.py applies on each scheduled date.

Event days that coincide are counted under each event they carry; ordinary
days carry none of the three.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(3)
SECTORS = {"Tech (XLK)": "XLK", "AI / semis (SMH)": "SMH", "Hyperscalers": "HYPER",
           "Healthcare (XLV)": "XLV", "Utilities (XLU)": "XLU", "Real estate (XLRE)": "XLRE",
           "S&P 500 (SPY)": "SPY"}
HYPER = ["MSFT", "GOOGL", "AMZN", "META", "ORCL"]
ETFS = ["QQQ", "SMH", "SPY", "TLT", "USO", "XLE", "XLK", "XLRE", "XLU", "XLV"]


def dates(name: str) -> set:
    txt = open(os.path.join(ROOT, "cache", f"{name}_dates.txt")).read().strip()
    return {dt.date.fromisoformat(x) for x in txt.split(",")}


def load() -> pd.DataFrame:
    path = os.path.join(ROOT, "cache", "etf_daily.parquet")
    if os.path.exists(path):
        px = pd.read_parquet(path)
    else:
        px = yf.download(ETFS, start="2014-01-01", end="2026-09-26", auto_adjust=True,
                         progress=False)["Close"]
        px.to_parquet(path)
    path = os.path.join(ROOT, "cache", "hyper_daily.parquet")
    if os.path.exists(path):
        h = pd.read_parquet(path)
    else:
        h = yf.download(HYPER, start="2014-01-01", end="2026-09-26", auto_adjust=True,
                        progress=False)["Close"]
        h.to_parquet(path)
    r = np.log(px).diff()
    # equal-weight daily-rebalanced basket of the five hyperscalers
    r["HYPER"] = np.log(h).diff().mean(axis=1)
    return r


def beta_ci(k: int, n: int, a: float = 0.5, draws: int = 20000):
    s = RNG.beta(k + a, n - k + a, draws)
    return s, np.percentile(s, [5, 95])


def main():
    r = load()
    ev = {"CPI": dates("cpi"), "Jobs report": dates("nfp"), "FOMC": dates("fomc")}
    res, rows = {}, []
    for label, col in SECTORS.items():
        x = r[col].dropna()
        sd = x.rolling(20).std().shift(1)
        z = (x / sd).dropna()
        z = z[z.index >= "2015-01-01"]
        day = pd.Index([d.date() for d in z.index])
        is_ev = {k: day.isin(list(v)) for k, v in ev.items()}
        # FOMC history only starts 2021: compare it against ordinary days of
        # the same span so the base rate is not from a calmer era
        any_ev = np.zeros(len(z), bool)
        for v in is_ev.values():
            any_ev |= v
        spike = np.abs(z.to_numpy()) >= 2
        z2 = z.to_numpy() ** 2
        for k in ev:
            span = day >= (dt.date(2021, 1, 1) if k == "FOMC" else dt.date(2015, 1, 1))
            m_ev = is_ev[k] & span
            m_no = ~any_ev & span
            ke, ne = int(spike[m_ev].sum()), int(m_ev.sum())
            kn, nn = int(spike[m_no].sum()), int(m_no.sum())
            se, cie = beta_ci(ke, ne)
            sn, cin = beta_ci(kn, nn)
            p_more = float((se > sn).mean())
            # variance multiplier with a bootstrap over event days
            a, b = z2[m_ev], z2[m_no]
            mult = a.mean() / b.mean()
            boot = [RNG.choice(a, len(a)).mean() / RNG.choice(b, len(b)).mean() for _ in range(3000)]
            rows.append({"sector": label, "event": k, "n": ne, "spikes": ke,
                         "rate": ke / ne, "ci_lo": cie[0], "ci_hi": cie[1],
                         "base_rate": kn / nn, "p_more_than_base": p_more,
                         "var_mult": mult, "mult_lo": np.percentile(boot, 5),
                         "mult_hi": np.percentile(boot, 95)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    df.to_csv(os.path.join(ROOT, "results", "event_history.csv"), index=False)


if __name__ == "__main__":
    main()
