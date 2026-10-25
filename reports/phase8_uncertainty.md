# Phase 8 — Severity uncertainty (quantile + CQR)

## Diagnosis of first attempt (2026-10-25)

The first implementation fit quantiles on `log1p(pure_premium_per_policy)`,
which has ~95% mass at exactly 0 (no claim). Quantile regression predicts
0 everywhere because in almost every leaf, the empirical quantile of a
mostly-zero distribution is 0. Result on val:

- Raw quantile coverage: 96.4%
- CQR coverage: 96.4% (identical)
- `q_hat` = 0.0000
- Interval widths: 0.00 in both log and EUR
- Median predicted premium: 0.00 EUR

The 96.4% coverage comes from the **zero mass**, not from working
intervals. The whole exercise is vacuous on this target.

**This is a subtle pitfall worth highlighting**: it demonstrates the
danger of applying standard quantile regression to actuarial targets
without understanding the mass-at-zero structure.

## Fix: fit on severity only, cover conditionally

1. Restrict quantile regression to claims with `ClaimAmount_sum > 0`
   (17,417 train rows, 3,755 val rows).
2. Target becomes `log(ClaimAmount_sum)`: heavy-tailed but continuous,
   no zero mass.
3. Measure coverage **conditionally**: "given that a claim occurs, my
   80% interval for the amount is [lo, hi]; on held-out check claims,
   does [lo, hi] contain the true amount 80% of the time?"
4. Frequency uncertainty (P(claim)) is separately quantified by Phase 6
   LGBM frequency model. Compound pure-premium interval and portfolio
   aggregation are Phase 9 concerns.

## Method: CQR (Conformalized Quantile Regression)

Standard pipeline from Romano, Patterson, Candes (2019):

1. Fit LightGBM quantile models q in {0.1, 0.5, 0.9} on `log(severity)`,
   using train claims only.
2. Split val claims into **cal** (50%, used for calibration) and **check**
   (50%, used only for reporting). Fixed seed=42.
3. Conformity scores on cal:
   `s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))`
4. Finite-sample q_hat:
   `q_hat = quantile(s, ceil((n_cal+1)(1-alpha))/n_cal)`, alpha = 0.20
5. Adjusted interval on check: `[q_lo(x) - q_hat, q_hi(x) + q_hat]`
6. Coverage = fraction of y_true in the adjusted interval, measured on
   check claims.

## Why conformal, not raw quantiles

Raw quantile regression is **only asymptotically valid**. With finite
calibration data and heavy tails, the raw interval typically under-covers.
Conformal calibration adds a non-negative `q_hat` to widen the interval,
restoring the coverage guarantee (>= 1-alpha with prob >= 1-alpha for
exchangeable data).

## Target choice: log(severity), not log1p(pure premium)

- `log1p(pure_premium)`: 95% mass at 0. Quantile regression predicts 0.
  **Failed.**
- `log(severity)`: continuous, heavy-tailed, no zero mass. Fit works.
  Coverage meaningful.

## Expected outcomes (after fix)

- **Raw coverage**: 70-78% (under-covers, finite-sample bias).
- **CQR coverage**: 78-82% (one-sided guarantee holds).
- **Median interval width**: 2,000-4,000 EUR (matches observed severity
  range 1-18k EUR).

Exact numbers in `reports/phase8_uncertainty.txt` from `make uncertainty`.

## Design decisions

- **Alpha = 0.20** (80% nominal interval). Change to 0.10 in Phase 9
  sensitivity for a 90% interval.
- **50/50 cal/check split on val claims**: ~1,877 cal + 1,877 check.
  Finite-sample CQR is valid at this size.
- **log target, exp back-transform**: preserves positivity; matches
  Phase 5/6 severity conventions.
- **No frequency classifier in Phase 8**: Phase 6 LGBM frequency model
  already gives P(claim). Compound composition is Phase 9.

## Downstream usage

- `q_hat` is frozen. Phase 9 applies it to test once, reports conditional
  coverage there.
- Phase 11 (FastAPI) `/predict` returns:
  - `expected_claim_prob` (from Phase 6 frequency model)
  - `severity_median` (from Phase 8 q=0.5)
  - `severity_lo` / `severity_hi` (from Phase 8 CQR)
- Compound pure-premium interval = `p_claim × [lo, hi] × exposure` with 0
  lower bound for the 95% no-claim mass.

## What this phase proves

- Whether raw quantile regression under-covers on severity (expected:
  yes, 70-78%).
- Whether CQR restores nominal coverage on the claim subpopulation
  (expected: yes, 78-82%).
- Whether interval widths are reasonable in EUR (visual sanity).

## Limitations

- **Marginal coverage only**, not conditional on covariates. A specific
  subpopulation (young drivers, high-density areas) may see different
  coverage. Phase 9 fairness analysis reports coverage per subgroup.
- **Severity-only intervals**. Frequency uncertainty is not in this
  interval. Compound composition is Phase 9.
- **Assumes exchangeability between cal and check**. Violated if temporal
  drift exists; untested with a single split.

## Portfolio message

Not "I implemented quantile regression" (junior). But "I noticed that
naive quantile regression on pure premium produces zero-width intervals
because of the mass-at-zero; split into frequency x severity; covered
severity conditionally; calibrated with CQR; measured conditional
coverage on held-out claims" (senior). That is the kind of detail that
distinguishes a candidate who has thought carefully about the problem
from one who has followed a tutorial.
