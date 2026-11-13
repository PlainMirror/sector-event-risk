# Sector Event Risk Models

How likely is a large one-day move in each sector, and how much of it comes from scheduled events (CPI, jobs reports, FOMC, earnings)? Three model families, scored on the same unseen days, January 2023 to September 2026.

Research only, not investment advice.

Series: Tech (XLK), semis (SMH), a hyperscaler basket (MSFT, GOOGL, AMZN, META, ORCL), Healthcare (XLV), Utilities (XLU), Real estate (XLRE), S&P 500 (SPY), Treasuries (TLT) and Oil (USO).

## The models

1. **GJR-GARCH(1,1) with Student-t errors** (`spikes.py`), with each scheduled date's variance scaled by an event multiplier shrunk toward the pooled mean (empirical Bayes).
2. **GARCH-EVT** (`evt.py`): GJR-GARCH fitted on 2015-2022 only, with generalised Pareto tails on the standardised residuals, losses and gains fitted separately.
3. **Multivariate Hawkes** (`hawkes.py`): a spike in one series raises the chance of spikes in the next days, in itself and the others, with event days in the baseline. Fitted by maximum likelihood from several starting points, with cross-sector links tested by likelihood ratio and a Benjamini-Hochberg correction.

A spike is a day beyond the series' 95th-percentile absolute move over 2015-2022.

## Results

**Event days.** Since 2021, FOMC days carry about 2.6 times the variance of an ordinary day for the S&P 500 (90% interval 1.18 to 4.68), and jobs reports about 1.4 times. Earnings move the hyperscaler basket far more than any macro release.

**Value at Risk, 2023-2026** (936 days per series, 16 tests at 99% and 99.5%):

| Model | Fails Kupiec (coverage) | Fails Christoffersen (independence) | Fails both together |
|---|---|---|---|
| Student-t | 4 of 16 | 0 of 16 | 4 of 16 |
| EVT tails | 1 of 16 | 0 of 16 | 0 of 16 |

**Contagion.** The Hawkes branching ratio is 0.85. Only 3 of 56 cross-sector links survive the correction: healthcare to tech, real estate to utilities, and tech to semis.

**Out-of-sample spike forecasts** (`compare.py`), as skill against always predicting 5%, with a 20-day block bootstrap:

| Model | Log-score skill [90% interval] |
|---|---|
| GARCH-t | 7.4% [4.8%, 9.9%] |
| GARCH-EVT | 7.4% [4.8%, 9.9%] |
| Hawkes, events only | 1.1% [-0.1%, 2.5%] |
| Hawkes + self-excitation | 6.5% [3.7%, 9.3%] |
| Hawkes + cross-sector | 5.6% [2.9%, 8.5%] |
| **50/50 GARCH-EVT + Hawkes** | **9.1% [6.6%, 11.6%]** |

Volatility clustering carries most of the skill; the event calendar alone barely beats a constant. No model beats the constant for utilities.

The models' probabilities of at least one spike in Q4 2026 are in `results/` and can be scored once the quarter ends.

## Limitations

- The 2026 election multiplier (3.7) is an assumption taken from SPY options.
- FOMC history here starts in 2021 (46 meetings).
- The 50/50 blend was scored on the same test days that suggested it.
- One Hawkes decay rate per target series.

## Running it

```bash
pip install -r requirements.txt
python history.py && python spikes.py && python evt.py && python hawkes.py && python compare.py
python tests/test_models.py      # no market data needed
```

The options step (`implied.py`, `implied_precise.py`) needs Alpaca keys in `.env`; its outputs are already in `results/`. Prices come from Yahoo and are cached in `cache/`.

| Path | What it is |
|---|---|
| `history.py`, `spikes.py` | event multipliers and the GARCH model |
| `evt.py` | GARCH-EVT tails and VaR back tests |
| `hawkes.py` | multivariate Hawkes fit and tests |
| `compare.py` | out-of-sample scoring |
| `tests/test_models.py` | 9 tests, including recovery of planted Hawkes parameters |

Tested with Python 3.13, arch 8.0, numpy 2.2 and scipy 1.16.
