"""Probability of spike days per sector, Oct 1 - Dec 31 2026.

1. GJR-GARCH(1,1) with Student-t errors per sector, fitted on 2015-2026 daily
   returns: volatility clusters, and down days raise it more than up days.
2. Scheduled dates scale that day's conditional variance by an event
   multiplier -- mean squared standardised return on past event days over
   ordinary days (history.py for macro; past report-reaction days here for
   earnings), each with a bootstrap standard error.
3. Empirical-Bayes shrinkage of every log-multiplier toward the pooled mean,
   weight tau^2 / (tau^2 + se^2): a cell measured on 46 noisy days cannot
   claim an extreme value on its own.
4. Election day (Nov 4 reaction) has no usable history (two midterms), so its
   multiplier comes from SPY options: the election-week excess variance net of
   the jobs report, haircut to 0.8 for the implied-over-realised premium.
5. 20,000 paths per sector; counts of |return| >= 2%, >= 3%, >= 5% days in the
   window, against the same counts under the options' 3-month implied vol.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t as student_t

import history

ROOT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(11)
N_PATHS = 20000
HOLIDAYS = {dt.date(2026, 11, 26), dt.date(2026, 12, 25)}
START_SIM, WIN0, WIN1 = dt.date(2026, 9, 28), dt.date(2026, 10, 1), dt.date(2026, 12, 31)

MACRO = {"CPI": [dt.date(2026, 10, 14), dt.date(2026, 11, 10), dt.date(2026, 12, 10)],
         "Jobs report": [dt.date(2026, 10, 2), dt.date(2026, 11, 6), dt.date(2026, 12, 4)],
         "FOMC": [dt.date(2026, 10, 28), dt.date(2026, 12, 9)]}
# upcoming reaction day per company (report after close -> next session)
EARN_NEXT = {"MSFT": dt.date(2026, 10, 29), "GOOGL": dt.date(2026, 10, 29), "META": dt.date(2026, 10, 29),
             "AMZN": dt.date(2026, 10, 30), "AAPL": dt.date(2026, 10, 30), "NVDA": dt.date(2026, 11, 18),
             "ORCL": dt.date(2026, 12, 11), "AVGO": dt.date(2026, 12, 10), "TSM": dt.date(2026, 10, 15),
             "AMD": dt.date(2026, 11, 4), "UNH": dt.date(2026, 10, 13), "LLY": dt.date(2026, 10, 29),
             "JNJ": dt.date(2026, 10, 13), "NEE": dt.date(2026, 10, 27), "PLD": dt.date(2026, 10, 15)}
# which companies' reports move which sector (membership, not weights --
# the multiplier measured on past reaction days already carries the weight)
EARN_MAP = {"Tech (XLK)": ["MSFT", "AAPL", "NVDA", "AVGO", "ORCL", "AMD"],
            "AI / semis (SMH)": ["NVDA", "TSM", "AVGO", "AMD"],
            "Hyperscalers": ["MSFT", "GOOGL", "META", "AMZN", "ORCL"],
            "Healthcare (XLV)": ["LLY", "UNH", "JNJ"],
            "Utilities (XLU)": ["NEE"], "Real estate (XLRE)": ["PLD"],
            "S&P 500 (SPY)": ["MSFT", "GOOGL", "META", "AMZN", "AAPL", "NVDA"]}
ELECTION = dt.date(2026, 11, 4)


def trading_days(a: dt.date, b: dt.date):
    d, out = a, []
    while d <= b:
        if d.weekday() < 5 and d not in HOLIDAYS:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def multiplier(z: pd.Series, days: set, ordinary: np.ndarray):
    idx = np.array([d.date() in days for d in z.index])
    a = z.to_numpy()[idx] ** 2
    b = z.to_numpy()[ordinary] ** 2
    if len(a) < 8:
        return None
    m = a.mean() / b.mean()
    boot = np.log([RNG.choice(a, len(a)).mean() / RNG.choice(b, len(b)).mean() for _ in range(2000)])
    return {"n": int(len(a)), "mult": float(m), "log": float(np.log(m)), "se": float(boot.std())}


def shrink(cells: dict) -> dict:
    logs = np.array([c["log"] for c in cells.values()])
    ses = np.array([c["se"] for c in cells.values()])
    mu = logs.mean()
    tau2 = max(logs.var() - (ses ** 2).mean(), 0.0)
    for c in cells.values():
        w = tau2 / (tau2 + c["se"] ** 2) if tau2 > 0 else 0.0
        c["shrunk"] = float(np.exp(mu + w * (c["log"] - mu)))
        c["weight"] = float(w)
    return {"mu": float(np.exp(mu)), "tau2": float(tau2)}


def main():
    r = history.load()
    ev_hist = {"CPI": history.dates("cpi"), "Jobs report": history.dates("nfp"),
               "FOMC": history.dates("fomc")}
    reacts = {k: {dt.date.fromisoformat(x) for x in v}
              for k, v in json.load(open(os.path.join(ROOT, "cache", "earnings_reactions.json"))).items()}
    imp = json.load(open(os.path.join(ROOT, "results", "implied_precise.json")))
    iv3 = json.load(open(os.path.join(ROOT, "results", "implied.json")))["sectors"]

    # ---- election multiplier from SPY options --------------------------------
    spy_iv = iv3["SPY"]["1m"]["iv"]
    ordinary_var = (100 * spy_iv) ** 2 / 252                       # %^2 per day
    week = imp["SPY|Midterm election"]["move_pct"] ** 2            # excess, %^2
    cells, fits = {}, {}
    sectors = history.SECTORS
    for label, col in sectors.items():
        x = r[col].dropna()
        x = x[x.index >= "2015-01-01"] * 100
        sd = (x / 100).rolling(20).std().shift(1)
        z = ((x / 100) / sd).dropna()
        any_ev = np.zeros(len(z), bool)
        for v in ev_hist.values():
            any_ev |= np.array([d.date() in v for d in z.index])
        for v in reacts.values():
            any_ev |= np.array([d.date() in v for d in z.index])
        ordinary = ~any_ev
        for k, v in ev_hist.items():
            c = multiplier(z[z.index >= ("2021-01-01" if k == "FOMC" else "2015-01-01")], v, ordinary[-len(z[z.index >= ("2021-01-01" if k == "FOMC" else "2015-01-01")]):])
            if c:
                cells[(label, k)] = c
        for co in EARN_MAP[label]:
            c = multiplier(z, reacts[co], ordinary)
            if c:
                cells[(label, f"{co} earnings")] = c
        am = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        fits[label] = am.fit(disp="off")
    pooled = shrink(cells)

    # SPY jobs-report share of election week, then the election's own excess
    nfp_spy = cells[("S&P 500 (SPY)", "Jobs report")]["shrunk"]
    elect_excess = max(week - (nfp_spy - 1) * ordinary_var, 0) * 0.8
    elect_mult = 1 + elect_excess / ordinary_var

    days = trading_days(START_SIM, WIN1)
    in_win = np.array([d >= WIN0 for d in days])
    out = []
    for label, res in fits.items():
        p = res.params
        mu, om, al, ga, be, nu = p["mu"], p["omega"], p["alpha[1]"], p["gamma[1]"], p["beta[1]"], p["nu"]
        h = float(res.conditional_volatility.iloc[-1] ** 2)
        e_prev = float(res.resid.iloc[-1])
        # Combine events on one day by EXCESS variance, not by multiplying:
        # the largest earnings effect that day (co-reporting names were
        # measured on overlapping days, so their multipliers already include
        # each other), plus each macro release's excess, plus the election's.
        earn_ex = np.zeros(len(days))
        other_ex = np.zeros(len(days))
        for k, ds in MACRO.items():
            c = cells.get((label, k))
            for d in ds:
                if c and d in days:
                    other_ex[days.index(d)] += c["shrunk"] - 1
        for co in EARN_MAP[label]:
            c = cells.get((label, f"{co} earnings"))
            d = EARN_NEXT[co]
            if c and d in days:
                i = days.index(d)
                earn_ex[i] = max(earn_ex[i], c["shrunk"] - 1)
        if ELECTION in days:
            other_ex[days.index(ELECTION)] += elect_mult - 1
        mult = 1 + earn_ex + other_ex
        scale = np.sqrt((nu - 2) / nu)
        H = np.full(N_PATHS, h)
        E = np.full(N_PATHS, e_prev)
        R = np.empty((N_PATHS, len(days)))
        for i in range(len(days)):
            H = om + (al + ga * (E < 0)) * E ** 2 + be * H
            eps = np.sqrt(H * mult[i]) * student_t.rvs(nu, size=N_PATHS, random_state=RNG) * scale
            R[:, i] = mu + eps
            E = eps
        W = np.abs(R[:, in_win])
        n_days = W.shape[1]
        ivs = iv3.get({"Tech (XLK)": "XLK", "AI / semis (SMH)": "SMH", "Healthcare (XLV)": "XLV",
                       "Utilities (XLU)": "XLU", "Real estate (XLRE)": "XLRE",
                       "S&P 500 (SPY)": "SPY"}.get(label, ""), {}).get("3m", {}).get("iv")
        def opt_p(th):
            if not ivs:
                return None
            sd_d = 100 * ivs / np.sqrt(252)
            q = 2 * norm.sf(th / sd_d)
            return 1 - (1 - q) ** n_days
        row = {"sector": label, "garch_vol_now": float(np.sqrt(h * 252)),
               "exp_days_2pct": float((W >= 2).sum(1).mean()),
               "p_any_3pct": float(((W >= 3).sum(1) > 0).mean()),
               "p_any_5pct": float(((W >= 5).sum(1) > 0).mean()),
               "median_worst_day": float(np.median(W.max(1))),
               "opt_p_any_3pct": opt_p(3), "opt_p_any_5pct": opt_p(5),
               "iv3m": ivs, "event_days": int((mult > 1).sum())}
        xs = r[sectors[label]].dropna()
        xs = xs[xs.index >= "2015-01-01"].abs() * 100
        q = xs.groupby(xs.index.to_period("Q")).max()
        row["hist_q_any_3pct"] = float((q >= 3).mean())
        row["hist_q_any_5pct"] = float((q >= 5).mean())
        row["hist_quarters"] = int(len(q))
        out.append(row)
        print(f"{label:<20} GARCH vol now {row['garch_vol_now']:5.1f}%  E[#days>=2%] {row['exp_days_2pct']:5.1f}  "
              f"P(any>=3%) {100*row['p_any_3pct']:5.1f}%  P(any>=5%) {100*row['p_any_5pct']:5.1f}%  "
              f"worst day med {row['median_worst_day']:.1f}%  | hist quarters P3 {100*row['hist_q_any_3pct']:4.0f}% P5 {100*row['hist_q_any_5pct']:4.0f}% | options P3 {'' if ivs is None else f'{100*opt_p(3):5.1f}%'}  P5 {'' if ivs is None else f'{100*opt_p(5):5.1f}%'}")
    print(f"\npooled event multiplier {pooled['mu']:.2f}, tau^2 {pooled['tau2']:.3f}; election multiplier {elect_mult:.2f}")
    cdf = pd.DataFrame([{"sector": k[0], "event": k[1]} | v for k, v in cells.items()])
    print(cdf.sort_values("shrunk", ascending=False).head(14).round(2).to_string(index=False))
    json.dump({"results": out, "pooled": pooled, "election_mult": elect_mult,
               "cells": cdf.to_dict("records")}, open(os.path.join(ROOT, "results", "spikes.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
