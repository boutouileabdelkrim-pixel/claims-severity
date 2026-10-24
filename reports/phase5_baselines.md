# Phase 5 — Baselines (corrected)

## Scope

Three baselines, same features, same splits, no touching test.

| Family | Dummy | GLM |
|--------|-------|-----|
| Frequency | constant rate = total Claims / total Exposure (train) | PoissonRegressor on RATE target, sample_weight = Exposure |
| Severity | constant per-claim mean (train, weighted by ClaimNb_sev) | GammaRegressor on RAW per-claim severity, sample_weight = ClaimNb_sev |
| Pure premium | rate * p_has_sev * Exposure * per-claim mean | rate_pred * p_has_sev * Exposure * per-claim_pred |

## The 9,116-claims problem (critical correction)

freMTPL2 has 34,060 policies reporting ClaimNb > 0, but only 24,944 policies
have a severity record. At claim level: 25,259 claims (train) of which only
18,453 have a sev amount (in train). The rest (9,116 across full data) are
documented dataset quirks.

**Naive formula** (rate_all_claims * sev_per_claim_of_sev_records) over-predicts
by exactly 1/p_has_sev, where p_has_sev = 18,453 / 25,259 = 0.731.

**Correct formula**:
    pure_premium = rate * p_has_sev * exposure * sev_per_claim

p_has_sev is computed on TRAIN only, frozen, applied to val (and later test).
A future model could predict P(has_sev | claim) feature-dependently, but for
a baseline a constant is the honest choice.

## Frequency model — rate target, not raw count

sklearn PoissonRegressor has NO offset. The current baseline fits on:
    y = ClaimNb / Exposure  (rate per policy)
    sample_weight = Exposure

This is exactly a log-link GLM with offset log(Exposure) for the Poisson
deviance. Predictions are rates; count = rate * exposure.

**Why not sample_weight=Exposure with y=ClaimNb**: that approach fits
"exposure-weighted expected count", which over-predicts policies with
low exposure and under-predicts high exposure. Empirically it over-predicts
total count by ~25% (AE = 0.80 in the first Phase 5 attempt). The rate
target fixes this.

## Severity target — raw, not capped

Phase 4 produced ClaimAmount_capped (p99.5 on train severity = 35,095.53
EUR, 125 policies capped in train). That cap is a **Phase 6 training
concern** for Gamma-objective GBMs — it stabilizes gradients without
discarding tail signal for the Tweedie-direct model.

**The baseline uses RAW ClaimAmount_sum.** Rationale:
- Baseline should be the simplest correct floor for Phase 6 to beat
- Capping at p99.5 would systematically under-predict raw severity
- AE should be ~1.0 on the true target (raw), not 0.73 on a manipulated one

## Numeric scaling — StandardScaler

The GLM features span wildly different ranges (VehPower 4-15, Density_log
0-10, one-hot 0-1). Without scaling, LBFGS failed to converge in 1000 iters
(ConvergenceWarning). Fix: StandardScaler fitted on train numerics, applied
to val. max_iter raised to 2000 for safety.

## Metrics

- Frequency: MAE on ClaimNb, Poisson deviance, actual/expected ratio
- Severity: MAE on per-claim, Gamma deviance, actual/expected ratio
- Pure premium: MAE, actual/expected ratio
- RMSE not used as primary (tail-dominated, see docs/CONTRACT.md)

Exact numbers are in `reports/phase5_baselines.txt`. Do not fill in guesses.

## Expected results

After corrections:
- Freq AE: dummy ≈ 1.00 (by construction), poisson ≈ 1.00 (rate target)
- Sev AE: dummy ≈ 1.00 (weighted mean), gamma ≈ 1.00
- PP AE: dummy ≈ 1.00, glm ≈ 1.00 (via p_has_sev)

If AE differs from 1.0 by more than 5%, that's a real calibration issue.

## Interpretation guide

- **GLM barely beats dummy on deviance (2-5%)**: expected and honest. Real
  gain from trees is also modest on freMTPL2 (5-10%).
- **GLM much worse than dummy**: bug. Check feature leak, split swap, target
  sign, exposure zeros.
- **AE far from 1.0**: under-fitting or distribution shift. Look at train vs
  val AE.

## Val-train severity distribution shift (important)

Val severity per policy is **+7.7% higher than train** (2,578 vs 2,394 EUR).
Cause: we stratified on ClaimNb > 0 (frequency), not on severity — and
severity is the target, so cannot be stratified on. With the heavy tail
(max claim 4.07M EUR), a few large claims landing in val vs train shifts
the mean.

**Consequence**: `dummy_va` sev AE = 1.0787 and `gamma_va` sev AE = 1.0902
are not model biases; they are the val split's severity distribution.

**Phase 6 implications**:
- Model selection must use DEVANCE (proper scoring rule), not AE
- AE on val can be 0.9-1.1 for a well-fit model
- Do not chase AE = 1.00 on val; it is unachievable by construction
- Phase 9 (test split) will confirm: test severity distribution vs train
- Document this in the Model Card (Phase 13)


## Phase 6 target

Beat the GLM baselines on val by at least 3% relative on the primary metric
(Poisson deviance for frequency, Gamma deviance for severity). Below that,
the GBM does not justify its complexity.
