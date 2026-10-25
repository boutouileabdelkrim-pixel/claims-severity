# Phase 9 — Final evaluation on test

## Scope

**Test split touched exactly once.** The chosen compound model
(LGBM frequency × GLM severity × p_has_sev) is evaluated here for the
last time before deployment concerns (Phase 10+).

After this script runs successfully, a sentinel file
`data/test_frozen.flag` is created. Future runs of `make evaluate`
refuse unless `--force` is passed. This is operational discipline
encoded in code, not just documentation.

## Model under test

- **Frequency**: LGBM Poisson on rate target, sample_weight=Exposure
  (Phase 6 defaults; Optuna found tuning does not help in Phase 7).
- **Severity**: GLM Gamma on raw per-claim target, sample_weight=
  ClaimNb_sev (Phase 6 choice: stable on tail, LGBM severity was
  unstable with AE 1.28).
- **Correction**: `p_has_sev = 0.7306` (train, frozen) accounts for
  the 9,116 freMTPL2 claims without a severity record.
- **Uncertainty**: CQR-calibrated severity intervals, `q_hat = 0.2714`
  (frozen from Phase 8).

## Metrics computed

| Metric | Interpretation |
|--------|----------------|
| Pure premium MAE | Absolute error, EUR |
| Pure premium AE | Actual/predicted ratio. 1.0 = calibrated |
| Pure premium Gini | Ranking quality. 0=random, 1=perfect |
| Frequency Poisson deviance | Proper scoring rule |
| Severity Gamma deviance | Proper scoring rule |
| Lift by decile | Actual/expected ratio per prediction decile |
| Conditional CQR coverage | Fraction of true severities in 80% interval |

## What the lift by decile shows

Sort policies by predicted premium descending (decile 1 = highest
predicted). For a well-calibrated model, all deciles should have
AE ≈ 1.0. Common defects:

- Decile 1 AE < 1: model over-predicts at the top (overconfident on
  big claims).
- Decile 10 AE > 1: model under-predicts at the bottom (misses small
  claims).
- If AE is monotone decreasing: model ranking is inverted vs calibration.

Phase 6 hit the tail-hardness issue; we expect deciles 8-10 to
under-predict (AE > 1) because most "small-claim" mass is at ~0.
Documented, not fixed here.

## What the SHAP shows

SHAP (TreeExplainer) on the LGBM frequency model, 5,000-row subsample.
Top features expected:
- `BonusMalus_log`: penalty/discount, strong signal in French motor.
- `DrivAge` / `DrivAge_bucket`: age is the dominant risk factor.
- `VehAge`, `VehPower`: vehicle characteristics.
- `Area_ord` / `Density_log`: urban/rural risk.

SHAP plot saved as `reports/phase9_shap_summary.png`. Not in this text
report to keep it readable.

## Fairness analysis — descriptive, not normative

Residuals by `DrivAge_bucket`, `Region` (top-5), `Area` (A-F).
Groups with AE outside [0.9, 1.2] should be reviewed. This is not
evidence of discrimination — but it is the kind of measurement a
responsible insurance model must produce.

Contract requires this section explicitly. No legal claim is made.

## Limitations

- Test metrics include the val-tail-shift bias inherited from
  Phase 5/6: test severity distribution may differ from train by
  similar magnitude (~+7.7%). Pure premium AE > 1 is expected and
  documented, not a model defect.
- SHAP on frequency only. Severity is GLM (linear, SHAP is trivial
  linear decomposition). Not reported.
- Fairness on groups is marginal. Conditional fairness (e.g., young
  drivers with high BonusMalus) is out of scope for this project.

## What this phase proves

- The compound model ranks correctly (Gini > 0.5 expected).
- The 4M outlier does not break the compound (already shown Phase 6).
- Uncertainty intervals calibrated on test claims hold at ~80%.
- Fairness gaps are measurable and bounded.
- Model is reproducible: `make evaluate` produces byte-identical
  reports (seeds fixed).

## Downstream

Test is frozen. Phase 10 (MLflow + Docker) logs artifacts from this
frozen model. Phase 11 (FastAPI) serves the frozen model. Phase 12
(monitoring) simulates drift on a different slice. Phase 13 writes
the Model Card referencing this report.

## Portfolio message

"I touched the test set once, measured everything a senior actuary
would want (Gini, lift, coverage, SHAP, fairness), documented where
the model fails and why, and froze the test set so nothing downstream
can cherry-pick." That is the kind of discipline that separates a
portfolio from a notebook dump.
