# Sector Event Risk Models

I wanted to know how likely a big move is in each sector over a quarter, and how much of that comes from scheduled events: CPI releases, jobs reports, FOMC meetings, earnings, and the November 2026 midterm election. I built three kinds of model for it (GARCH with event multipliers, GARCH with extreme-value tails, and a Hawkes model where spikes can trigger more spikes) and then scored all of them on the same unseen days, January 2023 to September 2026.

Built in September 2026, with data up to the September 25 close. Research only, not investment advice.

Sectors: Tech (XLK), AI / semis (SMH), Hyperscalers (an equal-weight basket of MSFT, GOOGL, AMZN, META, ORCL), Healthcare (XLV), Utilities (XLU), Real estate (XLRE) and the S&P 500 (SPY). The EVT and Hawkes models also include Treasuries (TLT) and Oil (USO).

## Do scheduled events actually move sectors?

`history.py` compares event days to ordinary days from 2015 (FOMC from 2021). A "spike" is a day where the return is at least 2 standard deviations of the trailing 20-day volatility, and the variance multiplier is the average squared standardised return on event days divided by the same thing on ordinary days, with a bootstrap 90% interval.

| | CPI | Jobs report | FOMC |
|---|---|---|---|
| S&P 500 (SPY) | 0.98 [0.76, 1.24] | 1.36 [1.03, 1.74] | 2.58 [1.18, 4.68] |
| Tech (XLK) | 0.93 [0.73, 1.16] | 1.32 [1.01, 1.69] | 1.47 [0.98, 2.04] |
| Real estate (XLRE) | 1.46 [1.08, 1.92] | 1.01 [0.72, 1.38] | 1.85 [0.94, 3.07] |
| Utilities (XLU) | 1.23 [0.98, 1.51] | 1.42 [0.92, 2.02] | 1.12 [0.65, 1.69] |

The first thing that stands out is FOMC days for the S&P 500: since 2021 they carry about 2.6 times the variance of an ordinary day, and 13% of them were spikes against a 5.9% base rate. Jobs reports matter for SPY and tech too. However, CPI days were not more volatile than ordinary days for SPY or tech at all over this period, even though they were for the rate-sensitive sectors like real estate. (Note: the FOMC sample is only 46 meetings, which is why its intervals are so wide.)

Earnings are a much bigger deal than any macro release for the names that report. On the hyperscaler basket, MSFT and GOOGL reaction days carry about 6.4 times the ordinary variance. Since a single cell measured on 46 noisy days shouldn't be allowed to claim an extreme value on its own, `spikes.py` shrinks every multiplier toward the pooled mean (empirical Bayes), which brings those two down to about 3.4 to 3.5.

## The three models

1. `spikes.py`: GJR-GARCH(1,1) with Student-t errors per sector, so volatility clusters and down days raise it more than up days. Each scheduled date scales that day's variance by its shrunk multiplier, and the election day multiplier (3.7) comes from SPY options, since there is no usable history for it. Then 20,000 simulated paths for October to December 2026.
2. `evt.py`: the McNeil and Frey GARCH-EVT approach. It fits GJR-GARCH-t on 2015 to 2022 only, then puts generalised Pareto tails on the standardised residuals, separately for losses and gains since they differ.
3. `hawkes.py`: a discrete-time multivariate Hawkes model on spike days, where a spike in one series raises the chance of spikes in the following days, in itself and in the other series, with CPI, jobs, FOMC and earnings days as baseline effects.

For the EVT and Hawkes models a spike is a day where the absolute return is above that series' 95th percentile from 2015 to 2022 (3.04% for tech, 4.05% for semis, 1.78% for TLT, 5.00% for oil), so a constant 5% is the natural benchmark.

## Do fat tails matter?

Yes for risk measures, but not much for one-day spike probabilities. Most loss tails have a positive EVT shape parameter (around +0.14 to +0.19 for tech, hyperscalers, real estate, TLT and oil), which means power-law tails, although each one has a standard error of about 0.12, so individually they are not strongly significant.

Where it shows is the VaR backtest on 2023 to 2026 (936 days per series, 16 tests in total at the 99% and 99.5% levels). Using Kupiec's test at 5%, the plain Student-t model fails 4 of the 16 (tech and real estate at both levels, for example tech's 99% VaR was breached on 2.24% of days instead of 1%). The EVT version fails only 1 of 16, and that one (healthcare at 99%) is for too few breaches, not too many.

Kupiec's test only counts breaches, so I also ran Christoffersen's independence test, which asks whether a breach today makes one tomorrow more likely. Neither model fails it on any of the 16. So the problem with the Student-t is how many breaches there are, not that they bunch together, which means the GARCH part is doing its job and the tail shape is what's wrong. On the joint test of both conditions, Student-t fails 4 of 16 and EVT fails none. (Note: these are 16 tests with no correction for running that many, so one failure at 5% is about what chance alone would give.)

## Do spikes spread across sectors?

Some, but not much between sectors. The Hawkes branching ratio is 0.85, so the system is stationary but a fair share of spikes are follow-ons from earlier ones. Self-excitation is strongest in oil (0.52), Treasuries (0.49) and healthcare (0.43), and the effect fades with a half-life of about 3 to 7 days. The largest event effects are earnings, which multiply the baseline spike intensity by about 20 for the hyperscaler basket and 9.5 for tech, and FOMC days for real estate (about 7.9).

For contagion between sectors, out of 56 possible links only 3 survive a likelihood-ratio test with a Benjamini-Hochberg correction at q < 0.05: healthcare to tech, real estate to utilities, and tech to semis. So whatever co-movement there is between sectors mostly happens on the same day (which this model deliberately treats as correlation, not contagion), rather than one sector's spike leading to another's over the next few days.

The Hawkes likelihood is not convex, so a single optimiser run can stop at a poor solution, and a likelihood-ratio test built on two such runs can't be trusted. Each fit is therefore run from 4 starting points, and each restricted model (one source removed) starts from the full fit. Since the full model contains every restricted one, it can never fit worse, and the code checks that: in 7 of the 8 series a restricted fit first scored higher than the full one, the full fit was restarted from it, and after that no likelihood ratio was negative. All 95 fits converged. The three links and the branching ratio came out the same as before the check, so the result didn't depend on it.

## Which model is best?

`compare.py` scores every model on the same 8 series x 936 test days (spike rate 4.2%), as skill against always predicting 5%: 0 means no better than the constant and positive means better. The intervals come from a 20-day block bootstrap so that a market-wide spike day gets resampled together.

| Model | Brier skill | Log-score skill [90% interval] |
|---|---|---|
| GARCH-t | 4.0% | 7.4% [4.8%, 9.9%] |
| GARCH + EVT tails | 4.0% | 7.4% [4.8%, 9.9%] |
| Hawkes: events only | 1.0% | 1.1% [-0.1%, 2.5%] |
| Hawkes: + self-excitation | 3.1% | 6.5% [3.7%, 9.3%] |
| Hawkes: + cross-sector | 2.8% | 5.6% [2.9%, 8.5%] |
| 50/50 GARCH-EVT + Hawkes | 4.4% | 9.1% [6.6%, 11.6%] |

A few things stand out:

- Knowing the event calendar alone barely beats a constant (1.1%). Most of the skill comes from volatility clustering, which both GARCH and the self-exciting Hawkes model capture.
- Adding the cross-sector terms made the Hawkes model worse out of sample (5.6% against 6.5%), which fits with only 3 of the 56 links being real.
- The best result is the simple average of GARCH-EVT and Hawkes, at 9.1%, since the two families get different days wrong.
- None of the models worked for utilities, where every one of them scored below the constant (-0.3% to -8.6%).
- Calibration is decent at the top but not in the middle. GARCH-EVT's highest decile predicted 17.6% and saw 15.9%, but its 8th and 9th deciles predicted 5.2% and 7.8% and saw only 3.1% and 5.2%.

## What the models said about Q4 2026

Probability of at least one spike day between October 1 and December 31, 2026, as of the September 25 close:

| Series | Spike threshold | GARCH-EVT | Hawkes | Share of past quarters with one |
|---|---|---|---|---|
| Tech (XLK) | 3.04% | 67.9% | 78.9% | 61.4% |
| AI / semis (SMH) | 4.05% | 82.2% | 87.6% | 75.0% |
| Hyperscalers | 3.24% | 80.7% | 88.4% | 79.5% |
| Healthcare (XLV) | 2.15% | 74.8% | 70.6% | 59.1% |
| Utilities (XLU) | 2.23% | 86.1% | 83.7% | 77.3% |
| Real estate (XLRE) | 2.47% | 73.7% | 75.9% | 65.9% |
| Treasuries (TLT) | 1.78% | 73.7% | 74.5% | 65.9% |
| Oil (USO) | 5.00% | 80.9% | 75.4% | 70.5% |

Both models put every series above its historical rate, which makes sense if we look at the calendar: the quarter has the election, two FOMC meetings and the main earnings season, on top of the volatility at the time. Q4 2026 is still in progress, so these can be checked against what actually happens once the quarter ends.

For the simpler thresholds in `spikes.py`, the chance of at least one 3% day was 97% for semis, 85% for hyperscalers, 68% for tech and 36% for the S&P 500 (against 89%, 85%, 62% and 32% of past quarters).

## What options were pricing for each event

`implied.py` and `implied_precise.py` read event-implied moves off the option chain by bracketing each event between the expiry before it and the first one on or after it. For SPY at the September 25 close:

| Event | Date | 1-sd implied move | Expiries used |
|---|---|---|---|
| Jobs report (Sep) | Oct 2 | 0.80% | Oct 1 to Oct 2 |
| CPI (Sep) | Oct 14 | 0.98% | Oct 9 to Oct 16 |
| FOMC + hyperscaler and AMZN/AAPL earnings | Oct 28 to 30 | 1.59% | Oct 23 to Oct 30 |
| Midterm election + jobs report (Oct) | Nov 4 to 6 | 1.47% | Oct 30 to Nov 6 |

Only the first one is a clean single-day number. Alpaca only listed weekly expiries beyond about October 2, so the later figures cover a whole event week.

## Limitations

- The 50/50 blend wasn't tuned, but I only thought to score it after seeing that the two families miss different days, so it is reported on the same test days it was motivated by.
- The Hawkes model has one decay rate per target series, shared by every source that excites it.
- Each event multiplier is measured against a 20-day trailing volatility and then applied on top of the GARCH variance, and in the Q4 simulation an event-day shock also raises the next day's variance. Both could count part of an event twice.
- Option quotes are Alpaca's free indicative feed, not the full OPRA NBBO.
- The election multiplier (3.7, applied to every sector) is an assumption taken from SPY options, haircut by 0.8 for the usual implied-over-realised premium.
- FOMC history only starts in 2021 here, so it rests on 46 meetings.
- Earnings effects are mapped by membership (which companies' reports move which sector), not by index weights.
- FRED blocks scripted requests, so prices come from Yahoo and event dates come from the BLS and Federal Reserve calendars instead.

## Running it

```bash
pip install -r requirements.txt
python history.py
python spikes.py
python evt.py
python hawkes.py
python compare.py
python tests/test_models.py
```

The tests need no market data. They check the VaR tests against cases that can be worked by hand (zero breaches must be a rejection, ten breaches in a row must fail independence), that the tail model's quantiles and probabilities invert each other, that the Hawkes excitation only uses past days, and that the fit recovers parameters planted in simulated data.

The options step (`implied.py`, then `implied_precise.py`) needs Alpaca API keys in a `.env` file (see `.env.example`). Its outputs are already in `results/`, so everything else runs without keys. `history.py` downloads daily prices from Yahoo the first time and caches them in `cache/`. All random seeds are fixed, so with the same price data the results reproduce exactly. A fresh download can differ slightly since Yahoo revises adjusted prices.

| Path | What it is |
|---|---|
| `history.py` | event-day spike rates and variance multipliers, 2015 to 2026 |
| `spikes.py` | GJR-GARCH-t with shrunk event multipliers, Q4 Monte Carlo |
| `evt.py` | GARCH-EVT tails, VaR backtest, Q4 Monte Carlo with EVT innovations |
| `hawkes.py` | multivariate Hawkes fit, likelihood-ratio tests, Q4 simulation |
| `compare.py` | out-of-sample Brier and log-score skill, calibration |
| `implied.py`, `implied_precise.py` | event-implied moves from option chains |
| `tests/test_models.py` | checks on the VaR tests, the tail model and the Hawkes fit |
| `cache/` | CPI, jobs report and FOMC dates, and past earnings reaction days |
| `results/` | every output used above |

Tested with Python 3.13, arch 8.0, numpy 2.2 and scipy 1.16.
