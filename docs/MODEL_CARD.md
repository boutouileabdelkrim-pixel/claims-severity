# Model Card — Claims Severity & Cost Prediction

## Model details

- **Name**: `claims-severity` v0.2.0
- **Type**: Compound pure-premium pricing model
  - Frequency: LightGBM Poisson (rate target, sample_weight=Exposure)
  - Severity: scikit-learn `GammaRegressor` (per-claim raw target)
  - Correction: `p_has_sev = 0.7306` (train-frozen, corrects for 9,116
    claims without a severity record in freMTPL2)
  - Uncertainty: Conformalized Quantile Regression (CQR, Romano 2019) on
    severity, `q_hat = 0.2714` frozen from Phase 8
- **Architecture components**:
  - `src/models/baseline.py` — Dummy, GLM Poisson/Gamma baselines
  - `src/models/gbm.py` — XGBoost + LightGBM (freq, sev, tweedie)
  - `src/models/uncertainty.py` — quantile + CQR
  - `src/models/actuarial.py` — Gini, Lorenz, lift by decile
  - `src/models/fairness.py` — residuals + coverage by group
  - `src/api/` — FastAPI serving (`/predict`, `/predict/batch`, `/health`, `/info`)
- **Model artifact**: `models/bundle_v1.pkl` (11.7 MB, gitignored)
- **MLflow run**: `8466b1dc3a174f4d9f6b6edcc03bebae`
- **Docker image**: `claims-severity:latest`, hash `8d05bc42a796`, 3 GB

## Intended use

**In-scope**:
- Motor third-party liability pricing in France (freMTPL2 domain)
- Pure-premium estimation for individual policies
- Portfolio risk ranking (via Gini on prediction)
- Uncertainty quantification for severity (80% conformal interval)
- Educational / portfolio demonstration

**Out-of-scope**:
- Other insurance lines (home, life, health) — model is freMTPL2-specific
- Real-time pricing under load > 100 req/s — FastAPI is single-worker
- Automated underwriting decisions — requires human review
- Regulatory pricing in production — see "Ethical considerations"

## Training data

**Source**: freMTPL2 via OpenML (`data_id` 41214 / 41215).
French motor third-party liability policies, 2014 vintage.

| Split | Rows | Claims | Exposure (yr) |
|-------|------|--------|---------------|
| Train | 474,609 | 25,259 | 250,891 |
| Val | 101,702 | 5,417 | 53,769 |
| Test | 101,702 | 5,417 | 53,839 |

**Severity subset**: 17,417 train claims with `ClaimAmount_sum > 0`.

**Split strategy**: stratified on `ClaimNb > 0` (frequency positive
class), seed=42, 70/15/15. One assignment per `IDpol`. Test split
frozen after Phase 9 (sentinel `data/test_frozen.flag`).

**Known dataset quirks** (documented in `reports/phase2_summary.md`):
- 9,116 policies report `ClaimNb > 0` but no severity record. Corrected
  via `p_has_sev = 0.7306` (constant from train).
- 6 orphan IDpols in sev but not freq. Dropped by LEFT JOIN.
- Max per-policy claim: 4,075,400 EUR. Kept (raw target), documented.

**Cap decision**: `ClaimAmount_sum` capped at p99.5 = 35,095.53 EUR
for the Gamma objective (Phase 4). Raw target preserved for evaluation.

## Evaluation

Metrics reported on test (touched ONCE in Phase 9, see
`reports/phase9_final_evaluation.txt`):

| Metric | Test value | Notes |
|--------|-----------:|-------|
| Pure premium MAE | 157.16 EUR | Per-policy absolute error |
| **Pure premium AE** | **0.9909** | Actual/predicted ratio — near-perfect calibration |
| Pure premium Gini | 0.2635 | Ranking quality (0=random, 1=perfect). **Honest: weak ranking** |
| Frequency Poisson deviance | 0.1507 | Matches Phase 6 val (0.1515), no overfit |
| Frequency AE | 0.9988 | Well-calibrated |
| Severity Gamma deviance | 1.6632 | |
| Severity AE | 0.9401 | Slight over-prediction (6%) |
| Conditional CQR coverage | **0.7975** | Nominal 80% — holds on test claims |

**Interpretation**:
- Global calibration is excellent (AE 0.99).
- **Ranking is weak (Gini 0.26)** — model knows the mean, not who pays more. This is a real freMTPL2 limitation, not a defect. A Gini > 0.40 would require external features (telematics, credit score) that are not in freMTPL2.
- Per-decile calibration drifts (decile 1 AE 0.68, decile 10 AE 14.49). Reported honestly in `reports/phase9_final_evaluation.txt`.

**Comparison to baselines** (val, Phase 5 → Phase 6):

| Model | Freq deviance | Sev deviance |
|-------|--------------:|-------------:|
| Dummy | 0.1662 | 1.8629 |
| GLM Poisson / GLM Gamma | 0.1584 | 1.8471 |
| **GBM compound (LGBM freq, GLM sev)** | **0.1515** (−4.4%) | 1.8471 |

**Optuna tuning** (Phase 7): 15 trials × 3-fold CV. Best tuned val
deviance 0.1517 vs default 0.1515 = **+0.12% worse**. Decision: keep
defaults. Tuning does not help on this dataset.

## Fairness analysis

Descriptive only. No legal claim made. From
`reports/phase9_final_evaluation.txt`:

| Group | n | Mean residual (EUR) | AE ratio |
|-------|---:|--------------------:|---------:|
| DrivAge bucket 0 (18-25) | 5,790 | −71.25 | 0.59 |
| DrivAge bucket 2 (35-55) | 49,826 | +4.74 | 1.06 |
| DrivAge bucket 4 (75+) | 2,781 | **+202.82** | **3.17** |
| Area A | 15,577 | −10.50 | 0.89 |
| Area B | 11,374 | −29.89 | 0.64 |
| Area D | 22,823 | +15.89 | 1.18 |
| Region R93 | 11,977 | +68.64 | 1.87 |

**Largest gaps**:
- **DrivAge bucket 4 (75+)**: model under-predicts by 217% on this
  segment (n=2,781, 2.7% of test). Real finding, not noise.
- **Region R93**: under-predicts by 87% (n=11,977, 11.8% of test).

**Cause**: these subgroups have elevated claim frequency or severity
that the model under-learns. Investigating in production requires
stratified retraining or per-group calibration factors.

**Responsible-usage note**: age and region can act as protected-class
proxies. This model should NOT be used for individual pricing decisions
in France without regulatory review (Article 225-1 of the French
penal code on discrimination, and ACPR guidance on pricing fairness).

## Limitations

1. **Weak ranking (Gini 0.26)**: model is well-calibrated on average
   but cannot sharply order policies. Documented in
   `reports/phase9_final_evaluation.txt`. Root cause: freMTPL2 lacks
   the discriminative features (telematics, credit, claims history)
   that modern motor pricing uses.

2. **Tail-sensitive evaluation**: pure premium AE on val is 1.1257 vs
   0.9909 on test. Difference (13.5%) is consistent with the val-tail
   severity shift (+7.7%, documented in Phase 5) plus cap effects. Not
   a defect. Any single number must be read against its split.

3. **Feature-space boundary at DrivAge = 101**: `engineer_features`
   bins DrivAge with `pd.cut(bins=[15,25,35,55,75,101])`. DrivAge > 100
   produces NaN and crashes the model. Mitigated by Pydantic schema
   (`le=100`) in the API. Documented in `reports/phase12_monitoring.md`.

4. **PSI misses tail shift on mass-at-zero targets**: `pure_premium` is
   ~95% zero. PSI misses tail shift (0.0168 on synthetic drift), but
   KS catches it (p=6.7e-84). Monitoring uses BOTH metrics. Documented
   in `reports/phase12_monitoring.md`.

5. **Single time comparison**: monitoring compares one train / one test
   split. No sliding-window history. Real production would use daily
   or weekly windows.

6. **Categorical drift not directly measured**: Region, VehBrand are
   excluded from PSI (high cardinality). Their shift is captured only
   via prediction drift.

7. **API is single-worker**: FastAPI + uvicorn single process. For
   production, deploy with `--workers 4` or behind a load balancer.

8. **MLflow file store**: local tracking only. Registry UI requires a
   server backend (one-line config change, not implemented).

9. **No authentication**: `/predict` has no API key. Demo only. Not
   production-ready.

10. **Model artifact not in git**: `models/bundle_v1.pkl` is gitignored
    (11.7 MB). Reproduce with `make mlflow && uv run python -m scripts.extend_bundle`.

## Ethical considerations

- **Fairness**: age and region show prediction gaps (Section "Fairness
  analysis"). This model is a portfolio demonstration, NOT a production
  pricing model. Deployment in a regulated context requires ACPR
  review, discrimination testing, and probably group-calibration
  factors.
- **Transparency**: SHAP feature importance is available in
  `reports/phase9_shap_summary.png` (top: `BonusMalus_log` 0.28,
  `VehAge` 0.17, `DrivAge` 0.13). Model is interpretable at the
  feature level, less so at the individual-policy level for deep trees.
- **Uncertainty**: CQR intervals are conditional on a claim occurring.
  For a policy with `P(claim) < 1`, the marginal interval includes the
  zero-claim mass. End-users must understand this distinction.
- **Data provenance**: freMTPL2 is a public academic dataset (Noll,
  Salzmann, Wüthrich 2016). No PII. License: OpenML terms.
- **No dark patterns**: this repo contains no scrapers, no A/B tests on
  humans, no automatic enforcement. Educational only.

## Recommendations for production

1. **Add group calibration** for DrivAge bucket 4 and Region R93 before
   any deployment touching pricing.
2. **Deploy with `uvicorn --workers 4`** or behind a container
   orchestration layer. Current API is single-worker.
3. **Wire `phase12_monitoring.json` to an alert system**: fire on
   `PSI > 0.25` OR `KS p < 1e-6` on any feature.
4. **Add API authentication** (OAuth2 / API key) before exposing
   publicly.
5. **Extend MLflow to server backend** for model registry + line-age
   tracking.
6. **Add external features** (telematics, credit history) to improve
   Gini above 0.30. Current freMTPL2 Gini of 0.26 is at the ceiling of
   what this dataset supports.
7. **Periodic recalibration** (every quarter) using the most recent
   actual/expected ratio per decile.

## Contact

**Author**: Boutouile Abdelkrim ([@boutouileabdelkrim-pixel](https://github.com/boutouileabdelkrim-pixel))
**Repo**: https://github.com/boutouileabdelkrim-pixel/claims-severity
**License**: Code MIT. Dataset: OpenML terms (see repo `LICENSE`).

## Version history

| Version | Date | Change |
|---------|------|--------|
| 0.1.0 | 2026-10-21 | Initial scaffold |
| 0.2.0 | 2026-10-26 | Full model + API + monitoring |
