"""Checks on the pieces the results rest on. No market data needed.

    python tests/test_models.py        # or: pytest tests

VaR backtests are checked against hand-computable cases, the tail model against
its own definitions, and the Hawkes fit against data simulated from known
parameters, since a fit that cannot recover what was planted proves nothing
about what it finds in real returns.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evt      # noqa: E402
import hawkes   # noqa: E402


# ------------------------------------------------------------------ VaR tests
def test_kupiec_rejects_zero_breaches():
    """936 days at 99% should see about 9 breaches. None is far too conservative."""
    rate, p = evt.kupiec(np.zeros(936, bool), 0.01)
    assert rate == 0.0
    assert p < 0.001, p


def test_kupiec_passes_exact_rate():
    hits = np.zeros(1000, bool)
    hits[::100] = True                      # exactly 1%
    rate, p = evt.kupiec(hits, 0.01)
    assert abs(rate - 0.01) < 1e-12 and p > 0.999


def test_kupiec_matches_binomial_likelihood():
    """Same statistic computed independently from the binomial pmf."""
    n, x, p0 = 250, 7, 0.01
    hits = np.zeros(n, bool)
    hits[:x] = True
    _, lr = evt.kupiec_lr(hits, p0)
    want = -2 * (binom.logpmf(x, n, p0) - binom.logpmf(x, n, x / n))
    assert abs(lr - want) < 1e-9, (lr, want)


def test_christoffersen_sees_clustering():
    """Ten breaches in a row and ten spread out have the same count, so Kupiec
    cannot tell them apart. The independence test must."""
    bunched = np.zeros(1000, bool); bunched[500:510] = True
    spread = np.zeros(1000, bool); spread[50::100] = True
    assert evt.kupiec(bunched, 0.01)[1] == evt.kupiec(spread, 0.01)[1]
    assert evt.var_tests(bunched, 0.01)["ind_p"] < 1e-6
    assert evt.var_tests(spread, 0.01)["ind_p"] > 0.5


# ------------------------------------------------------------------ EVT tails
def test_semiparametric_tails_are_consistent():
    z = np.random.default_rng(0).standard_t(4, 20000)
    sp = evt.SemiParam(z)
    # the tail probability at each threshold is the share of points beyond it
    assert abs(sp.sf(sp.uU) - sp.pU) < 1e-12
    assert abs(sp.cdf(-sp.uL) - sp.pL) < 1e-12
    # quantile and cdf are inverses in the loss tail
    for a in (0.01, 0.005, 0.001):
        assert abs(sp.cdf(sp.ppf_low(a)) - a) < 1e-9
    # a Student-t with 4 degrees of freedom has tail index 1/4
    assert 0.1 < sp.xiL < 0.4 and 0.1 < sp.xiU < 0.4, (sp.xiL, sp.xiU)


# --------------------------------------------------------------------- Hawkes
def test_excitation_uses_only_the_past():
    X = (np.random.default_rng(1).random((300, 2)) < 0.1).astype(float)
    S = hawkes.exc(X, 0.7)
    X2 = X.copy()
    X2[200] = 1 - X2[200]                   # change day 200
    S2 = hawkes.exc(X2, 0.7)
    assert np.array_equal(S[:201], S2[:201])        # days up to and including 200 unchanged
    assert not np.array_equal(S[201], S2[201])      # the day after sees it


def test_kernel_sums_to_one():
    X = np.zeros((2000, 1)); X[0] = 1.0     # one spike
    assert abs(hawkes.exc(X, 0.8).sum() - 1.0) < 1e-6


def _simulate(T, m, g, alpha, beta, seed):
    """Spike days from the model in hawkes.py, for N series with one event type."""
    rng = np.random.default_rng(seed)
    N = len(m)
    D = (rng.random((T, 1)) < 0.05).astype(float)
    delta = np.exp(-np.asarray(beta))
    X, S = np.zeros((T, N)), np.zeros((N, N))       # S[i, j]: target i, source j
    prev = np.zeros(N)
    for t in range(T):
        S = delta[:, None] * S + (1 - delta)[:, None] * prev[None, :]
        lam = np.exp(np.asarray(m) + np.asarray(g) * D[t, 0]) + (alpha * S).sum(1)
        prev = (rng.random(N) < -np.expm1(-lam)).astype(float)
        X[t] = prev
    return X, D


def test_fit_recovers_planted_parameters():
    """Series 1 excites series 0 and itself; series 0 excites nothing."""
    alpha = np.array([[0.0, 0.4], [0.0, 0.5]])
    X, D = _simulate(40000, m=[np.log(0.03), np.log(0.03)], g=[np.log(3), 0.0],
                     alpha=alpha, beta=[0.3, 0.3], seed=2)
    both = np.ones(2, bool)
    th0, ll0, ok0 = hawkes.fit(X[:, 0], X, D, both)
    m, g, a, b = hawkes.unpack(th0, 1, both)
    assert ok0
    assert abs(np.exp(a[1]) - 0.4) < 0.12, np.exp(a)        # the planted cross link
    assert np.exp(a[0]) < 0.08, np.exp(a)                   # no self-excitation was planted
    assert abs(g[0] - np.log(3)) < 0.35, g                  # event lift
    assert abs(np.exp(m) - 0.03) < 0.012, np.exp(m)         # baseline
    assert 0.15 < np.exp(b) < 0.6, np.exp(b)                # decay rate
    th1, ll1, _ = hawkes.fit(X[:, 1], X, D, both)
    a1 = hawkes.unpack(th1, 1, both)[2]
    assert abs(np.exp(a1[1]) - 0.5) < 0.12 and np.exp(a1[0]) < 0.08, np.exp(a1)


def test_restricted_model_never_beats_full():
    """Dropping a source cannot raise the likelihood when the fits are sound."""
    alpha = np.array([[0.2, 0.3], [0.0, 0.4]])
    X, D = _simulate(15000, m=[np.log(0.03)] * 2, g=[0.5, 0.5], alpha=alpha, beta=[0.3, 0.3], seed=3)
    both = np.ones(2, bool)
    th, ll, _ = hawkes.fit(X[:, 0], X, D, both)
    for j in (0, 1):
        mk = both.copy(); mk[j] = False
        _, ll0, _ = hawkes.fit(X[:, 0], X, D, mk, start=np.delete(th, 1 + 1 + j))
        assert ll0 <= ll + 1e-6, (j, ll0, ll)
    # and dropping the real link costs far more than chance would
    mk = both.copy(); mk[1] = False
    _, ll_no_cross, _ = hawkes.fit(X[:, 0], X, D, mk)
    assert 2 * (ll - ll_no_cross) > 10


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print("ALL PASS" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
