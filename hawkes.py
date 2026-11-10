"""Discrete-time multivariate Hawkes model of spike days, with scheduled events.

Spike X_it = 1 when |return| >= c_i, the series' 95th-percentile absolute
daily move over 2015-2022 (the same definition evt.py scores). For target i:

    lambda_it = exp(m_i + g_i . D_t)  +  sum_j alpha_ij * S_ij,t
    S_ij,t    = delta_i * S_ij,t-1 + (1 - delta_i) * X_j,t-1      (delta_i = e^-beta_i)
    P(X_it=1) = 1 - exp(-lambda_it)

D_t = CPI, jobs report, FOMC, and the target's own earnings-reaction days.
The kernel sums to one over lags, so alpha_ij is roughly the expected number
of extra i-spikes one j-spike triggers; the spectral radius of alpha is the
system's branching ratio (< 1 = stationary). Half-life ln2 / beta_i days.
Excitation starts the day AFTER a spike: same-day co-moves are correlation,
which a copula would model, not contagion.

The likelihood factorises by target, so each series is fitted on its own.
Three nested variants are scored out of sample (fit 2015-2022, predict
2023-2026 one day ahead): events only, + self-excitation, + cross-excitation.
Cross terms are tested by likelihood ratio on the full sample (boundary-
corrected, Benjamini-Hochberg across all 56 pairs).
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import chi2

import evt
import history
import spikes

ROOT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(5)
SER = list(evt.SERIES.items())               # 6 sectors + TLT + USO
EVENTS = ["CPI", "Jobs report", "FOMC", "Earnings"]
EARN_MEMBERS = {**spikes.EARN_MAP}


def dates_of(name):
    p = "fomc_dates_full" if name == "fomc" else f"{name}_dates"
    return {dt.date.fromisoformat(x) for x in open(os.path.join(ROOT, "cache", f"{p}.txt")).read().strip().split(",")}


def build():
    r = evt.returns()
    cols = [c for _, c in SER]
    R = r[cols].dropna()
    thr = np.array([evt.spike_threshold(r[c].dropna()) for c in cols])
    X = (R.abs().to_numpy() >= thr).astype(float)
    days = [d.date() for d in R.index]
    reacts = {k: {dt.date.fromisoformat(x) for x in v}
              for k, v in json.load(open(os.path.join(ROOT, "cache", "earnings_reactions.json"))).items()}
    macro = {"CPI": dates_of("cpi"), "Jobs report": dates_of("nfp"), "FOMC": dates_of("fomc")}
    D = {}
    for label, _ in SER:
        mem = EARN_MEMBERS.get(label, [])
        earn = set().union(*[reacts[m] for m in mem]) if mem else set()
        D[label] = np.array([[d in macro["CPI"], d in macro["Jobs report"], d in macro["FOMC"], d in earn]
                             for d in days], float)
    return R.index, X, D, thr


def exc(X, delta):
    """S_t = delta S_{t-1} + (1-delta) X_{t-1}, all sources at once."""
    Xl = np.vstack([np.zeros((1, X.shape[1])), X[:-1]])
    return lfilter([1 - delta], [1, -delta], Xl, axis=0)


def unpack(th, D, mask):
    m, g = th[0], th[1:1 + D]
    a = np.full(len(mask), -30.0)
    a[mask] = th[1 + D:1 + D + mask.sum()]
    b = th[-1]
    return m, g, a, b


def nll(th, y, X, Dm, mask):
    m, g, a, b = unpack(th, Dm.shape[1], mask)
    delta = np.exp(-np.exp(b))
    lam = np.exp(m + Dm @ g) + exc(X, delta) @ np.exp(a)
    lam = np.clip(lam, 1e-9, 50)
    return -(y * np.log(-np.expm1(-lam)) - (1 - y) * lam).sum()


N_STARTS = 4        # the likelihood is not convex, so one starting point is not enough
ALPHA_OFF = -20.0   # log-alpha at its lower bound: that source is switched off


def fit(y, X, Dm, mask, start=None, n_starts=N_STARTS):
    """Maximum likelihood from several starting points; returns (theta, loglik, converged).

    The default start, an optional warm start, and a few random ones drawn from
    a fixed seed (so a rerun gives the same fit). The best likelihood wins.
    """
    nd, k = Dm.shape[1], int(mask.sum())
    bounds = [(-12, 2)] + [(-3, 3)] * nd + [(ALPHA_OFF, 2)] * k + [(np.log(0.01), np.log(3))]
    lo, hi = np.array(bounds).T
    base = np.r_[np.log(max(y.mean(), 1e-3) * 0.6), np.zeros(nd), np.full(k, np.log(0.05)), np.log(0.15)]
    starts = [base] + ([np.asarray(start, float)] if start is not None else [])
    rng = np.random.default_rng(11)
    for _ in range(n_starts - 1):
        s = base.copy()
        s[0] += rng.normal(0, 0.5)
        s[1 + nd:1 + nd + k] = rng.uniform(np.log(0.005), np.log(0.3), k)
        s[-1] = rng.uniform(np.log(0.03), np.log(1.0))
        starts.append(s)
    best = None
    for s in starts:
        res = minimize(nll, np.clip(s, lo, hi), args=(y, X, Dm, mask), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best.x, -best.fun, bool(best.success)


def predict(th, X, Dm, mask):
    m, g, a, b = unpack(th, Dm.shape[1], mask)
    lam = np.exp(m + Dm @ g) + exc(X, np.exp(-np.exp(b))) @ np.exp(a)
    return -np.expm1(-lam)


def main():
    idx, X, D, thr = build()
    N = X.shape[1]
    tr = np.array(idx <= evt.TRAIN_END)
    labels = [l for l, _ in SER]
    print(f"{len(idx)} common sessions {idx[0].date()}..{idx[-1].date()}; spike thresholds "
          + ", ".join(f"{l.split(' (')[0]} {100*c:.2f}%" for l, c in zip(labels, thr)))
    # ---------------- out-of-sample: events only / self / full ---------------
    preds, n_fits, n_failed, n_refit = [], 0, 0, 0
    for i, label in enumerate(labels):
        y = X[:, i]
        out = {"series": label, "date": idx[~tr], "spike": y[~tr].astype(int)}
        for name, mask in (("events", np.zeros(N, bool)), ("self", np.eye(N, dtype=bool)[i]),
                           ("full", np.ones(N, bool))):
            th, _, ok = fit(y[tr], X[tr], D[label][tr], mask)
            n_fits += 1; n_failed += not ok
            p = predict(th, X, D[label], mask)
            out[f"p_hawkes_{name}"] = p[~tr]
        preds.append(pd.DataFrame(out))
    pd.concat(preds).to_parquet(os.path.join(ROOT, "results", "pred_hawkes.parquet"))
    # ---------------- full-sample fit + LR tests -----------------------------
    full_th, alpha, half, gam, lls = {}, np.zeros((N, N)), {}, {}, {}
    lr_rows, n_neg = [], 0
    for i, label in enumerate(labels):
        y, mask = X[:, i], np.ones(N, bool)
        th, ll, ok = fit(y, X, D[label], mask)
        n_fits += 1; n_failed += not ok
        # Each restricted model drops one source and starts from the full fit
        # with that source removed, so the two fits are compared like for like.
        nd, restricted = D[label].shape[1], {}
        for j in range(N):
            if j == i:
                continue
            mk = mask.copy(); mk[j] = False
            th0, ll0, ok0 = fit(y, X, D[label], mk, start=np.delete(th, 1 + nd + j))
            n_fits += 1; n_failed += not ok0
            restricted[j] = (th0, ll0)
        # The full model contains every restricted one, so it can never fit
        # worse. If a restricted fit scored higher, the full fit was stuck:
        # restart it from that restricted solution with the source switched off.
        jb = max(restricted, key=lambda j: restricted[j][1])
        if restricted[jb][1] > ll + 1e-8:
            th2, ll2, ok2 = fit(y, X, D[label], mask, start=np.insert(restricted[jb][0], 1 + nd + jb, ALPHA_OFF))
            n_fits += 1; n_refit += 1
            if ll2 > ll:
                th, ll, ok = th2, ll2, ok2
        full_th[label], lls[label] = th, ll
        m, g, a, b = unpack(th, 4, mask)
        alpha[i] = np.exp(a)
        half[label] = float(np.log(2) / np.exp(b))
        gam[label] = dict(zip(EVENTS, np.exp(g)))
        for j, (_, ll0) in restricted.items():
            n_neg += ll0 > ll + 1e-6
            lr = max(2 * (ll - ll0), 0)
            lr_rows.append({"target": label, "source": labels[j], "alpha": float(alpha[i, j]),
                            "lr": lr, "p": 0.5 * chi2.sf(lr, 1)})
        print(f"{label:<20} self {alpha[i, i]:.3f}  half-life {half[label]:.1f}d  "
              f"event lift x: " + ", ".join(f"{k} {v:.2f}" for k, v in gam[label].items()), flush=True)
    lr = pd.DataFrame(lr_rows).sort_values("p").reset_index(drop=True)
    m = len(lr)
    lr["bh_q"] = (lr["p"] * m / (np.arange(m) + 1))[::-1].cummin()[::-1].clip(upper=1)
    rho = float(max(abs(np.linalg.eigvals(alpha))))
    print(f"\nspectral radius (branching ratio) {rho:.3f}")
    print(f"optimiser: {n_fits} fits x {N_STARTS} starts, {n_failed} did not converge, "
          f"{n_refit} full fits restarted from a restricted one, {n_neg} likelihood ratios still negative")
    print(lr.head(15).round(4).to_string(index=False))
    # excitation state on the last observed day: where the simulation starts from
    S_end = {l: exc(X, np.exp(-np.exp(unpack(full_th[l], 4, np.ones(N, bool))[3])))[-1] for l in labels}
    # ---------------- Q4 2026 simulation --------------------------------------
    days = spikes.trading_days(spikes.START_SIM, spikes.WIN1)
    in_win = np.array([d >= spikes.WIN0 for d in days])
    Df = {}
    for label in labels:
        mem = EARN_MEMBERS.get(label, [])
        earn = {spikes.EARN_NEXT[mm] for mm in mem}
        Df[label] = np.array([[d in spikes.MACRO["CPI"], d in spikes.MACRO["Jobs report"],
                               d in spikes.MACRO["FOMC"], d in earn] for d in days], float)
    P = spikes.N_PATHS
    deltas = np.array([np.exp(-np.exp(unpack(full_th[l], 4, np.ones(N, bool))[3])) for l in labels])
    S = np.stack([np.tile(S_end[l], (P, 1)) for l in labels])       # target x path x source
    last_x = np.tile(X[-1], (P, 1))
    hits = np.zeros((P, N, len(days)), bool)
    for t in range(len(days)):
        S = deltas[:, None, None] * S + (1 - deltas)[:, None, None] * last_x[None, :, :]
        lam = np.empty((P, N))
        for i, l in enumerate(labels):
            m, g, a, b = unpack(full_th[l], 4, np.ones(N, bool))
            lam[:, i] = np.exp(m + Df[l][t] @ g) + S[i] @ np.exp(a)
        xnew = RNG.random((P, N)) < -np.expm1(-lam)
        hits[:, :, t] = xnew
        last_x = xnew.astype(float)
    W = hits[:, :, in_win]
    # historical: share of calendar quarters with >= 1 spike, and mean count
    Xdf = pd.DataFrame(X, index=idx, columns=labels)
    q_any = Xdf.groupby(Xdf.index.to_period("Q")).max().mean()
    q_cnt = Xdf.groupby(Xdf.index.to_period("Q")).sum().mean()
    # current state: next-5-session spike probability vs unconditional
    nxt5 = hits[:, :, :5].any(2).mean(0)
    uncond5 = 1 - (1 - Xdf.mean()) ** 5
    q4 = pd.DataFrame({"series": labels, "threshold_pct": 100 * thr,
                       "p_any_q4": W.any(2).mean(0), "exp_spikes_q4": W.sum(2).mean(0),
                       "hist_q_any": q_any.to_numpy(), "hist_q_count": q_cnt.to_numpy(),
                       "p_next5": nxt5, "uncond_next5": uncond5.to_numpy(),
                       "self_alpha": np.diag(alpha), "half_life_d": [half[l] for l in labels]})
    print(q4.round(3).to_string(index=False))
    q4.to_csv(os.path.join(ROOT, "results", "hawkes_q4.csv"), index=False)
    lr.to_csv(os.path.join(ROOT, "results", "hawkes_cross.csv"), index=False)
    pd.DataFrame(alpha, index=labels, columns=labels).to_csv(os.path.join(ROOT, "results", "hawkes_alpha.csv"))
    json.dump({"rho": rho, "gamma": gam, "half_life": half,
               "optimiser": {"fits": n_fits, "starts_per_fit": N_STARTS, "not_converged": int(n_failed),
                             "full_fits_restarted": n_refit, "negative_lr": int(n_neg)}}, open(os.path.join(ROOT, "results", "hawkes_fit.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
