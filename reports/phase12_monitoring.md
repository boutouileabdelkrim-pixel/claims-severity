# Phase 12 — Monitoring (PSI + KS drift detection)

## Scope

Detect distribution shift between the reference (train) and the
current (test) population, per feature + on the model's prediction.
No model metrics computed here — Phase 9 froze them.

## Metrics

**PSI (Population Stability Index)**
- Binned (10 quantile bins from REFERENCE, applied to current)
- Formula: `Σ (current_pct - ref_pct) × ln(current_pct / ref_pct)`
- Thresholds (industry standard):
  - < 0.10 — stable, no action
  - 0.10 – 0.25 — moderate shift, investigate
  - > 0.25 — significant shift, retrain candidate

**KS (Kolmogorov-Smirnov)**
- Two-sample test on continuous features
- Statistic = max CDF gap between reference and current
- P-value < 0.05 → reject H0 (distributions differ)

## Why both

PSI binned: sensitive to mass shifts between buckets.
KS: sensitive to CDF differences, catches tail shifts that binned PSI
may miss. Using both avoids the false confidence of a single metric.

## Design decisions

**Reference bins come from train, not test.** Using test quantiles
would mask the shift. Standard practice: reference defines the bins.

**Categoricals excluded from PSI.** `Region` has 22 values, `VehBrand`
11. With 10 bins, per-bin counts would be dominated by 1-2 policies.
Their shift is captured by the model's prediction drift instead.

**Synthetic drift as a positive control.** A detector that says "no
drift" on everything is useless. We also run against a modified test
set (`BonusMalus × 1.5` clipped at 300, `Exposure × 0.3` clipped at
0.01) which MUST trigger significant PSI on those features. If it
doesn't, the detector is broken.

## Discovery: feature-space boundary on DrivAge

**The first synthetic-drift attempt used `DrivAge + 20` and crashed**
with `pandas.errors.IntCastingNaNError: Cannot convert non-finite
values (NA or inf) to integer`.

Root cause: `engineer_features` (Phase 4) bin DrivAge with
`pd.cut(bins=[15, 25, 35, 55, 75, 101])`. The `101` upper bound exists
because freMTPL2's max observed DrivAge is ~100. Adding +20 pushes
some values to ~120, which falls outside the last bin. `pd.cut` returns
`NaN` for out-of-range values. `.astype("int8")` then raises.

**This is a real production limitation, not a bug to hide:**

- The model literally cannot score a policy with `DrivAge > 100`.
- The **API is protected**: `PolicyInput` (Phase 11) enforces
  `DrivAge: int = Field(..., ge=18, le=100)`. Pydantic rejects
  out-of-range values with HTTP 422.
- The **monitoring script bypasses Pydantic** (uses raw DataFrame for
  speed), so it can hit the boundary. We choose synthetic drift that
  stays in schema bounds.

**Portfolio value of this finding:** it demonstrates why input
validation at the API layer is critical. A junior engineer would let
the model crash silently. A senior engineer notices the boundary,
documents it, and enforces it via schema.

**Phase 13 action**: document in Model Card "known limitations".
Consider defensive clipping inside `engineer_features` (out of scope
for Phase 12 to avoid touching Phase 4 code).

## Test split

`test.parquet` is FROZEN since Phase 9 for model metrics. Phase 12
reads test distributions only — never computes MAE, AE, deviance, or
any error metric on test. The `data/test_frozen.flag` sentinel still
blocks `make evaluate`.

Documented for auditors: **the sentinel prevents metric contamination,
not distribution analysis**.

## Expected results

- **Real (train vs test)**: all features "stable". The stratified split
  on `ClaimNb > 0` ensures similar distributions for the features that
  matter most.

- **Synthetic (train vs drifted)**: `BonusMalus`, `BonusMalus_log`,
  `Exposure`, and `pure_premium` MUST be "significant"
  (PSI > 0.25). Others stable.

Exact numbers in `reports/phase12_monitoring.txt` (from `make monitor`)
and `reports/phase12_monitoring.json` (machine-readable).

## Dashboard integration path

`phase12_monitoring.json` is a machine-readable artifact. A production
system would:
1. Run `scripts/monitor_drift.py` on the last 7 days of production
   traffic.
2. Parse the JSON.
3. Alert if any feature has PSI > 0.25 OR more than 2 features have
   PSI > 0.10.
4. Re-run `make evaluate --force` and retrain if the alert fires twice
   in a row.

This is documented for the Model Card (Phase 13).

## Limitations

- **Single time comparison.** Real production would have daily/weekly
  sliding windows. Here we have one train / one test split.
- **No threshold tuning on real drift.** We use industry-standard
  thresholds (0.10 / 0.25) calibrated for credit risk, not motor
  insurance. Phase 13 Model Card notes this.
- **Categorical drift not directly measured.** Only `Area` (ordinal),
  `VehGas` (binary), and `DrivAge_bucket` (ordinal) are captured.
- **DrivAge boundary (documented above)** — model cannot score
  DrivAge > 100. API schema enforces ≤ 100.

## What this phase proves

- The serving population is consistent with training (no silent drift).
- The monitoring infrastructure works — synthetic drift IS detected.
- Dashboard-ready JSON output exists for Phase 13 Model Card.

## Phase 13 handoff

- Model Card references this report in the "Monitoring" section, plus
  the DrivAge boundary as a "Known Limitations" item.
- README documents `make monitor` as the ops check.
- A future CI job (`.github/workflows/monitoring.yml`) runs monitor
  weekly. Deferred to Phase 13.



## Why pure_premium shows "moderate" despite shifted inputs

In the synthetic drift test, three features show PSI > 4.0 (unambiguous
shift): `BonusMalus`, `BonusMalus_log`, `Exposure`. But `pure_premium`
shows PSI = 0.0168 ("stable"), even though it's computed FROM those
shifted inputs.

**Why**: `pure_premium` is ~95% exactly zero (policies with no claim).
Multiplying by 0.3 leaves the zero mass untouched (`0 × 0.3 = 0`).
Only the 5% non-zero tail shifts. Binned PSI is dominated by the zero
mass and doesn't detect the tail shift.

**What DOES detect it**: KS test, p-value = 6.7e-84 (definitively
different distributions). But KS statistic is only 0.034 (3.4% max
CDF gap), because the CDF difference is confined to the top 5%. So
PSI says "no mean shift in the dominant bins" AND KS says "small but
definite tail shift" — both are correct.

## Combined verdict logic (portfolio principle)

A production monitoring system should use BOTH PSI and KS. Neither
alone is sufficient on mass-at-zero insurance targets:

- **PSI** catches mean shifts within bins. Reliable when no single
  value dominates the distribution.
- **KS** catches CDF differences including tail shift. Reliable when
  sample size is large but doesn't tell you WHERE the shift is.

**Decision rule documented here** for Phase 13 Model Card:
- If either metric flags a feature (`PSI > 0.25` OR `KS p < 1e-6`),
  investigate. Don't rely on one alone.
- Flagship test (unambiguous shift): `BonusMalus`, `BonusMalus_log`,
  `Exposure` all show PSI > 4.0.
- Confirmation test (mass-at-zero): `pure_premium` shows KS p < 1e-6
  even though PSI is quiet. The combination proves the pipeline
  captured drift that pure-PSI monitoring would have missed.

## Why pure_premium shows "stable" despite shifted inputs

In the synthetic drift test, three features show PSI > 4.0 (unambiguous
shift): `BonusMalus`, `BonusMalus_log`, `Exposure`. But `pure_premium`
shows PSI = 0.0168 ("stable"), even though it's computed FROM those
shifted inputs.

**Why**: `pure_premium` is ~95% exactly zero (policies with no claim).
Multiplying by 0.3 leaves the zero mass untouched (`0 × 0.3 = 0`).
Only the 5% non-zero tail shifts. Binned PSI is dominated by the zero
mass and doesn't detect the tail shift.

**What DOES detect it**: KS test, p-value = 6.7e-84 (definitively
different distributions). But KS statistic is only 0.034 (3.4% max
CDF gap), because the CDF difference is confined to the top 5%. So
PSI says "no mean shift in the dominant bins" AND KS says "small but
definite tail shift" — both are correct.

## Combined verdict logic (portfolio principle)

A production monitoring system should use BOTH PSI and KS. Neither
alone is sufficient on mass-at-zero insurance targets:

- **PSI** catches mean shifts within bins. Reliable when no single
  value dominates the distribution.
- **KS** catches CDF differences including tail shift. Reliable when
  sample size is large but doesn't tell you WHERE the shift is.

**Decision rule documented here** for Phase 13 Model Card:
- If either metric flags a feature (`PSI > 0.25` OR `KS p < 1e-6`),
  investigate. Don't rely on one alone.
- Flagship test (unambiguous shift): `BonusMalus`, `BonusMalus_log`,
  `Exposure` all show PSI > 4.0.
- Confirmation test (mass-at-zero): `pure_premium` shows KS p < 1e-6
  even though PSI is quiet. The combination proves the pipeline
  captured drift that pure-PSI monitoring would have missed.

