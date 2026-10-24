# Phase 6 — GBM (XGBoost + LightGBM)

## Scope

Six models on the same features and splits as Phase 5. Test split never
loaded — only train for fitting, val for early stopping + evaluation.

| Family | XGBoost | LightGBM |
|--------|---------|----------|
| Frequency | `count:poisson`, rate target, sample_weight=Exposure | `poisson`, same |
| Severity | `reg:gamma`, RAW per-claim target, weight=n_claims | `gamma`, same |
| Tweedie direct | `reg:tweedie` (p=1.5), raw per-policy target | `tweedie` (p=1.5), same |

## Results

### Frequency — GBM wins

| Model | Val deviance | Val AE |
|-------|--------------|--------|
| GLM Poisson | 0.1584 | 0.9966 |
| XGB Poisson | 0.1518 | 0.9981 |
| **LGBM Poisson** | **0.1515** | **0.9965** |

-4.40% deviance vs GLM. In the expected 3-8% range from freMTPL2
literature. GBM earns its place on frequency modeling.

### Severity — GBM does NOT earn its place

| Model | Val deviance | Val AE |
|-------|--------------|--------|
| GLM Gamma (raw) | 1.8471 | 1.0902 |
| XGB Gamma (raw) | 1.8622 | 1.0946 |
| **LGBM Gamma (raw)** | **1.7965** | **1.2818** |

LGBM has the best deviance (1.7965) but the WORST calibration
(AE = 1.2818 — underpredicts total severity by 24%). Classic tail
overfitting: LGBM finds patterns near the 4M EUR outlier that don't
generalize to val, so rare high-severity rows get predicted too low
while average deviance still improves.

**Conclusion**: on freMTPL2, GBM severity is unstable. GLM wins on
calibration and is nearly as good on deviance.

### Pure premium — best-of-both

| Model | Val MAE | Val AE |
|-------|---------|--------|
| GLM compound (GLM freq × GLM sev) | 170.51 | 1.1228 |
| **GBM compound (LGBM freq × GLM sev)** | **168.64** | **1.1257** |
| XGB Tweedie (raw target) | 156.72 | 1.4048 |
| LGBM Tweedie (raw target) | 154.18 | 1.4605 |

- **Compound GBM ≈ GLM on AE** (1.1257 vs 1.1228), slightly better MAE.
- **Tweedie broken**: AE 1.40-1.46 (underpredicts by 40-46%). Documented
  limitation — the 4M outlier dominates the NLL objective.
- **Residual AE ≈ 1.12** vs raw target is the val-tail-shift (+7.7%
  severity on val vs train, documented in Phase 5 baselines). Not a bug.

## Discovery: capping experiment (documented for posterity)

Three strategies tested, all failed to recover calibration:

| Strategy | Severity cap | Best PP AE | Note |
|----------|--------------|-----------|------|
| Raw + LGBM sev | none | 1.28 | Unstable tail |
| Capped p99.5 + LGBM sev | 35,096 EUR | 1.46 | Capping kills severity mass |
| Capped p99.9 + LGBM sev | 126,014 EUR | 1.57 | Worse — sensitivity flips |

**Root cause**: Phase 3 EDA shows top 0.1% of policies hold 60.2% of
total severity mass. Any cap removes a large fraction of the true mean.
p99.5 removes ~40% of mass, p99.9 removes ~35% (because it caps exactly
the 17 policies holding most of the mass).

**Winning answer**: raw target + GLM severity. Stable, honest.

## Key senior finding

**"On freMTPL2, GBM does not earn its complexity for severity modeling."**
- Use LGBM for frequency (real 4.4% gain).
- Use GLM for severity (stable calibration, nearly equal deviance).
- Tweedie direct: exploratory failure, documented, not used downstream.

This is what a portfolio project should show: not "GBM everywhere", but
**each model where it earns its place**.

## Design decisions

**Frequency — rate target + sample_weight.** Equivalent to log-link GLM
with offset log(Exposure). Cross-version stable (XGBoost's `base_margin`
API shifted 2.x→3.x; LightGBM's `init_score` not applied on .predict()
by default).

**Severity — RAW per-claim target, weight = ClaimNb_sev.**
`y = ClaimAmount_sum / ClaimNb_sev`, `weight = ClaimNb_sev`.
No cap. Accept training noise from the 4M outlier.

**Tweedie direct — raw per-policy target.** `y = ClaimAmount_sum`
(NaN→0 for no-claim policies). Handles the 9,116-claims-without-severity
problem natively. **Broken on this dataset** due to the 4M outlier.
Documented, not used in the final compound model.

**Compound formula**: `LGBM_freq_rate × Exposure × GLM_sev_per_claim × p_has_sev`
where `p_has_sev = 0.7306` (train only, frozen) corrects for the 9,116
claims without severity records.

**No Optuna in Phase 6.** Sane defaults, early stopping (50 rounds).
Phase 7 will tune.

**One-hot for both GBMs.** Matches Phase 5.

## Phase 7 target

Optuna over LGBM frequency and LGBM Tweedie:
- Frequency: 30-50 trials, 5-fold CV on train. Early stopping removed.
- Severity (GLM): no tuning needed — GLM defaults are honest.
- Tweedie: try tuning to see if it can recover AE < 1.20. If not, document
  and move on (probably won't work due to tail).
- Budget ~30-60 min. If exceeds 30 min locally, shift to Kaggle T4.

Compare best-tuned LGBM freq vs Phase 6 defaults. If tuned doesn't beat
defaults by ≥2% on deviance, tuning not worth its cost — report and move
to Phase 8 (uncertainty).
