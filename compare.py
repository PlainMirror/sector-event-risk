"""Score every spike-probability model on the same unseen days (2023-2026).

Models: constant 5% (the training base rate, by construction of the
threshold), GARCH-t, GARCH-EVT, Hawkes with events only, + self-excitation,
+ cross-excitation, and a 50/50 average of GARCH-EVT and the full Hawkes. Scores: Brier (mean squared error of the probability) and
log loss, both reported as SKILL against the constant: 1 - score/score_const,
so 0 = no better than always saying 5%, positive = better.

Uncertainty: a block bootstrap over dates (20-session blocks, shared across
series, so a market-wide spike day is resampled together), 2,000 draws.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(8)
MODELS = ["p_const", "p_garch_t", "p_garch_evt", "p_hawkes_events", "p_hawkes_self", "p_hawkes_full", "p_blend"]
NAMES = {"p_const": "Constant 5%", "p_garch_t": "GARCH-t", "p_garch_evt": "GARCH + EVT tails",
         "p_hawkes_events": "Hawkes: events only", "p_hawkes_self": "Hawkes: + self-excitation",
         "p_hawkes_full": "Hawkes: + cross-sector", "p_blend": "50/50 GARCH-EVT + Hawkes"}


def brier(p, y):
    return ((p - y) ** 2).mean()


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def main():
    g = pd.read_parquet(os.path.join(ROOT, "results", "pred_garch.parquet"))
    h = pd.read_parquet(os.path.join(ROOT, "results", "pred_hawkes.parquet"))
    g["date"] = pd.to_datetime(g["date"]); h["date"] = pd.to_datetime(h["date"])
    d = g.merge(h.drop(columns=["spike"]), on=["series", "date"], how="inner")
    d["p_const"] = 0.05
    # the two model families miss different days, so their average is scored too
    d["p_blend"] = 0.5 * (d["p_garch_evt"] + d["p_hawkes_full"])
    dates = np.array(sorted(d["date"].unique()))
    blocks = [dates[i:i + 20] for i in range(0, len(dates), 20)]
    print(f"{d['series'].nunique()} series x {len(dates)} test sessions "
          f"({pd.Timestamp(dates[0]).date()}..{pd.Timestamp(dates[-1]).date()}), "
          f"spike rate {d['spike'].mean():.3f}")

    def skills(frame):
        y = frame["spike"].to_numpy()
        b0, l0 = brier(frame["p_const"].to_numpy(), y), logloss(frame["p_const"].to_numpy(), y)
        return {m: (1 - brier(frame[m].to_numpy(), y) / b0, 1 - logloss(frame[m].to_numpy(), y) / l0) for m in MODELS}

    point = skills(d)
    boot = {m: [] for m in MODELS}
    by_date = {k: v for k, v in d.groupby("date")}
    for _ in range(2000):
        pick = RNG.integers(0, len(blocks), len(blocks))
        fr = pd.concat([by_date[x] for b in pick for x in blocks[b]])
        s = skills(fr)
        for m in MODELS:
            boot[m].append(s[m])
    rows = []
    for m in MODELS:
        bs = np.array(boot[m])
        rows.append({"model": NAMES[m], "brier_skill": point[m][0], "bs_lo": np.percentile(bs[:, 0], 5),
                     "bs_hi": np.percentile(bs[:, 0], 95), "log_skill": point[m][1],
                     "ls_lo": np.percentile(bs[:, 1], 5), "ls_hi": np.percentile(bs[:, 1], 95)})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nPOOLED, all series (skill vs constant; 90% block-bootstrap interval)")
    print(out.round(4).to_string(index=False))
    per = []
    for s, fr in d.groupby("series"):
        sk = skills(fr)
        per.append({"series": s, "spike_rate": fr["spike"].mean(),
                    **{NAMES[m]: sk[m][1] for m in MODELS[1:]}})
    per = pd.DataFrame(per)
    print("\nLOG-SCORE SKILL by series")
    print(per.round(3).to_string(index=False))
    # calibration of the best pooled model and GARCH-t
    cal = []
    for m in ("p_garch_evt", "p_hawkes_full"):
        q = pd.qcut(d[m].rank(method="first"), 10, labels=False)
        for k, fr in d.groupby(q):
            cal.append({"model": NAMES[m], "decile": int(k) + 1, "predicted": fr[m].mean(), "observed": fr["spike"].mean(), "n": len(fr)})
    cal = pd.DataFrame(cal)
    print("\nCALIBRATION (deciles of predicted probability)")
    print(cal.round(3).to_string(index=False))
    out.to_csv(os.path.join(ROOT, "results", "compare_pooled.csv"), index=False)
    per.to_csv(os.path.join(ROOT, "results", "compare_series.csv"), index=False)
    cal.to_csv(os.path.join(ROOT, "results", "compare_calibration.csv"), index=False)


if __name__ == "__main__":
    main()
