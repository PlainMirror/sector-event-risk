"""GARCH-EVT (McNeil & Frey, 2000): extreme-value tails on filtered residuals.

1. GJR-GARCH(1,1)-t per series, fitted on TRAIN (2015-2022) only; the fitted
   parameters then filter TEST (2023-2026-09) without refitting, so every test
   day's volatility uses information up to the day before.
2. Standardised residuals z = eps / sigma. Each tail (losses and gains
   separately -- they differ) gets a generalised Pareto fit to exceedances over
   its 95th percentile; the body stays empirical. A shape xi > 0 is a heavy
   (power-law) tail; the Student-t's own tail index is 1/nu.
3. Out-of-sample checks on TEST:
   - one-day-ahead probability of a spike, |return| >= c, where c is the
     series' 95th-percentile absolute move in TRAIN (the same spike definition
     hawkes.py uses), saved for scoring in compare.py;
   - 99% and 99.5% one-day loss quantiles (VaR) for t vs EVT, with Kupiec's
     proportion-of-failures test (is the breach count right?), Christoffersen's
     independence test (do breaches bunch together?) and the two combined.
4. Oct-Dec 2026 Monte Carlo with EVT innovations (filtered historical
   simulation with GPD tails) and the same scheduled event multipliers as
   spikes.py, refitted on the FULL sample for forecasting.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.special import xlogy
from scipy.stats import chi2, genpareto, t as student_t

import history
import spikes

ROOT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(21)
TRAIN_END, TEST_START = "2022-12-31", "2023-01-01"
SERIES = {"Tech (XLK)": "XLK", "AI / semis (SMH)": "SMH", "Hyperscalers": "HYPER",
          "Healthcare (XLV)": "XLV", "Utilities (XLU)": "XLU", "Real estate (XLRE)": "XLRE",
          "Treasuries (TLT)": "TLT", "Oil (USO)": "USO"}
TAIL_Q = 0.95


def returns() -> pd.DataFrame:
    r = history.load()                     # log returns incl. HYPER basket
    return r[r.index >= "2015-01-01"]


def spike_threshold(x: pd.Series) -> float:
    return float(x[x.index <= TRAIN_END].abs().quantile(0.95))


class SemiParam:
    """Empirical body + GPD tails for standardised residuals."""

    def __init__(self, z: np.ndarray):
        z = z[np.isfinite(z)]
        self.z = np.sort(z)
        self.uL = np.quantile(-z, TAIL_Q)          # loss-tail threshold (positive)
        self.uU = np.quantile(z, TAIL_Q)
        exL = -z[-z > self.uL] - self.uL
        exU = z[z > self.uU] - self.uU
        self.xiL, _, self.bL = genpareto.fit(exL, floc=0)
        self.xiU, _, self.bU = genpareto.fit(exU, floc=0)
        self.pL, self.pU = len(exL) / len(z), len(exU) / len(z)
        self.nL, self.nU = len(exL), len(exU)

    def sf(self, x: float) -> float:
        """P(Z > x)."""
        if x >= self.uU:
            return self.pU * genpareto.sf(x - self.uU, self.xiU, scale=self.bU)
        return float((self.z > x).mean())

    def cdf(self, x: float) -> float:
        """P(Z < x)."""
        if x <= -self.uL:
            return self.pL * genpareto.sf(-x - self.uL, self.xiL, scale=self.bL)
        return float((self.z < x).mean())

    def ppf_low(self, a: float) -> float:
        """Lower-tail quantile, a < pL."""
        return -(self.uL + genpareto.ppf(1 - a / self.pL, self.xiL, scale=self.bL))

    def draw(self, n: int) -> np.ndarray:
        u = RNG.random(n)
        out = RNG.choice(self.z[(self.z > -self.uL) & (self.z < self.uU)], n)
        lo, hi = u < self.pL, u > 1 - self.pU
        out[lo] = -(self.uL + genpareto.rvs(self.xiL, scale=self.bL, size=lo.sum(), random_state=RNG))
        out[hi] = self.uU + genpareto.rvs(self.xiU, scale=self.bU, size=hi.sum(), random_state=RNG)
        return out


def fit_filter(x: pd.Series, train_end: str | None):
    """Fit GJR-GARCH-t on data up to train_end, filter the whole series."""
    xp = x * 100
    fit_on = xp if train_end is None else xp[xp.index <= train_end]
    res = arch_model(fit_on, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    full = arch_model(xp, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fix(res.params)
    return res, full


def kupiec_lr(hits: np.ndarray, p: float):
    """Breach rate and Kupiec's likelihood ratio for 'the breach rate equals p'.

    xlogy(0, 0) = 0, so a sample with no breaches (or all breaches) is handled:
    zero breaches where several were expected is a rejection, not a pass.
    """
    n, x = len(hits), int(np.sum(hits))
    ph = x / n
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_obs = xlogy(n - x, 1 - ph) + xlogy(x, ph)
    return ph, float(max(-2 * (ll_null - ll_obs), 0.0))


def kupiec(hits: np.ndarray, p: float):
    """Unconditional coverage: is the number of breaches right? Returns (rate, p-value)."""
    ph, lr = kupiec_lr(hits, p)
    return ph, float(chi2.sf(lr, 1))


def christoffersen_lr(hits: np.ndarray) -> float:
    """Christoffersen's (1998) likelihood ratio for independence of breaches.

    Kupiec only counts breaches. This asks whether a breach today makes one
    tomorrow more likely, which is what a volatility model is supposed to
    prevent. It compares one breach probability for every day against two:
    one after a quiet day (pi01) and one after a breach (pi11).
    """
    h = np.asarray(hits).astype(int)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum()); n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum()); n11 = int(((prev == 1) & (cur == 1)).sum())
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    ll_one = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_two = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    return float(max(-2 * (ll_one - ll_two), 0.0))


def var_tests(hits: np.ndarray, p: float) -> dict:
    """Breach rate with three p-values: coverage, independence, and both jointly."""
    rate, lr_uc = kupiec_lr(hits, p)
    lr_ind = christoffersen_lr(hits)
    return {"rate": rate, "p": float(chi2.sf(lr_uc, 1)), "ind_p": float(chi2.sf(lr_ind, 1)),
            "cc_p": float(chi2.sf(lr_uc + lr_ind, 2))}


def main():
    r = returns()
    preds, tails, var_rows = [], [], []
    for label, col in SERIES.items():
        x = r[col].dropna()
        c = spike_threshold(x)
        res, full = fit_filter(x, TRAIN_END)
        mu, nu = res.params["mu"], res.params["nu"]
        sig = full.conditional_volatility                 # % units, one step ahead
        z = (full.resid / sig)
        ztr = z[z.index <= TRAIN_END].to_numpy()
        sp = SemiParam(ztr)
        tscale = np.sqrt((nu - 2) / nu)
        # --- tail comparison on TRAIN residuals --------------------------------
        emp = lambda k: float((np.abs(ztr) > k).mean())
        t_tail = lambda k: float(2 * student_t.sf(k / tscale, nu))
        e_tail = lambda k: sp.sf(k) + sp.cdf(-k)
        tails.append({"series": label, "xi_loss": sp.xiL, "xi_gain": sp.xiU,
                      "xi_loss_se": (1 + sp.xiL) / np.sqrt(sp.nL), "xi_gain_se": (1 + sp.xiU) / np.sqrt(sp.nU),
                      "nu": nu, "t_tail_index": 1 / nu, "threshold_pct": 100 * c,
                      **{f"p_gt{k}_emp": emp(k) for k in (3, 4, 5)},
                      **{f"p_gt{k}_t": t_tail(k) for k in (3, 4, 5)},
                      **{f"p_gt{k}_evt": e_tail(k) for k in (3, 4, 5)}})
        # --- one-day-ahead spike probabilities on TEST -------------------------
        te = x[x.index >= TEST_START]
        s_te = sig[sig.index >= TEST_START]
        cp = 100 * c
        p_t = [float(student_t.sf((cp - mu) / s / tscale, nu) + student_t.cdf((-cp - mu) / s / tscale, nu)) for s in s_te]
        p_e = [sp.sf((cp - mu) / s) + sp.cdf((-cp - mu) / s) for s in s_te]
        preds.append(pd.DataFrame({"series": label, "date": te.index, "spike": (te.abs() >= c).astype(int).to_numpy(),
                                   "p_garch_t": p_t, "p_garch_evt": p_e}))
        # --- 99% / 99.5% loss quantiles, OOS -----------------------------------
        for a in (0.01, 0.005):
            q_t = mu + s_te * tscale * student_t.ppf(a, nu)
            q_e = mu + s_te * sp.ppf_low(a)
            hits_t = (100 * te.to_numpy() < q_t.to_numpy())
            hits_e = (100 * te.to_numpy() < q_e.to_numpy())
            vt, ve = var_tests(hits_t, a), var_tests(hits_e, a)
            var_rows.append({"series": label, "level": 1 - a, "n": len(te), "expected": a,
                             "t_rate": vt["rate"], "t_p": vt["p"], "evt_rate": ve["rate"], "evt_p": ve["p"],
                             "t_ind_p": vt["ind_p"], "evt_ind_p": ve["ind_p"],
                             "t_cc_p": vt["cc_p"], "evt_cc_p": ve["cc_p"]})
        print(f"{label:<20} spike>= {cp:.2f}%  xi loss {sp.xiL:+.2f}  gain {sp.xiU:+.2f}  nu {nu:.1f}", flush=True)
    pd.concat(preds).to_parquet(os.path.join(ROOT, "results", "pred_garch.parquet"))
    tdf, vdf = pd.DataFrame(tails), pd.DataFrame(var_rows)
    tdf.to_csv(os.path.join(ROOT, "results", "evt_tails.csv"), index=False)
    vdf.to_csv(os.path.join(ROOT, "results", "evt_var_backtest.csv"), index=False)
    pd.set_option("display.width", 220)
    print(tdf[["series", "xi_loss", "xi_loss_se", "xi_gain", "xi_gain_se", "t_tail_index",
               "p_gt4_emp", "p_gt4_t", "p_gt4_evt", "p_gt5_emp", "p_gt5_t", "p_gt5_evt"]].round(4).to_string(index=False))
    print(vdf.round(4).to_string(index=False))
    q4(r)


def q4(r: pd.DataFrame):
    """Oct-Dec 2026 with EVT innovations, full-sample fit, same event calendar."""
    cells = pd.DataFrame(json.load(open(os.path.join(ROOT, "results", "spikes.json")))["cells"])
    elect = json.load(open(os.path.join(ROOT, "results", "spikes.json")))["election_mult"]
    days = spikes.trading_days(spikes.START_SIM, spikes.WIN1)
    in_win = np.array([d >= spikes.WIN0 for d in days])
    out = []
    for label, col in SERIES.items():
        x = r[col].dropna()
        c = 100 * spike_threshold(x)
        res, full = fit_filter(x, None)
        p = res.params
        sig = full.conditional_volatility
        sp = SemiParam((full.resid / sig).to_numpy())
        ex_e, ex_o = np.zeros(len(days)), np.zeros(len(days))
        for k, ds in spikes.MACRO.items():
            m = cells[(cells.sector == label) & (cells.event == k)]
            for d in ds:
                if len(m) and d in days:
                    ex_o[days.index(d)] += float(m.shrunk.iloc[0]) - 1
        for co in spikes.EARN_MAP.get(label, []):
            m = cells[(cells.sector == label) & (cells.event == f"{co} earnings")]
            d = spikes.EARN_NEXT[co]
            if len(m) and d in days:
                ex_e[days.index(d)] = max(ex_e[days.index(d)], float(m.shrunk.iloc[0]) - 1)
        if spikes.ELECTION in days:
            ex_o[days.index(spikes.ELECTION)] += elect - 1
        mult = 1 + ex_e + ex_o
        H = np.full(spikes.N_PATHS, float(sig.iloc[-1] ** 2))
        E = np.full(spikes.N_PATHS, float(full.resid.iloc[-1]))
        R = np.empty((spikes.N_PATHS, len(days)))
        for i in range(len(days)):
            H = p["omega"] + (p["alpha[1]"] + p["gamma[1]"] * (E < 0)) * E ** 2 + p["beta[1]"] * H
            eps = np.sqrt(H * mult[i]) * sp.draw(spikes.N_PATHS)
            R[:, i] = p["mu"] + eps
            E = eps
        W = np.abs(R[:, in_win])
        out.append({"series": label, "threshold_pct": c, "p_any_spike": float((W >= c).any(1).mean()),
                    "exp_spikes": float((W >= c).sum(1).mean()),
                    "p_any_3pct": float((W >= 3).any(1).mean()), "p_any_5pct": float((W >= 5).any(1).mean())})
        print(f"Q4 EVT {label:<20} P(any >= {c:.2f}%) {100*out[-1]['p_any_spike']:.1f}%  "
              f"E[#] {out[-1]['exp_spikes']:.1f}  P(any 3%) {100*out[-1]['p_any_3pct']:.1f}%  P(any 5%) {100*out[-1]['p_any_5pct']:.1f}%", flush=True)
    pd.DataFrame(out).to_csv(os.path.join(ROOT, "results", "evt_q4.csv"), index=False)


if __name__ == "__main__":
    main()
