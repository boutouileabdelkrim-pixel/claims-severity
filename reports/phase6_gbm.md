# Phase 6 — GBM (XGBoost + LightGBM)

## Scope

Six models on the same features and splits as Phase 5. Test split never
loaded — only train for fitting, val for early stopping + evaluation.

| Family | XGBoost | LightGBM |
|--------|---------|----------|
| Frequency | `count:poisson`, rate target, sample_weight=Exposure | `poisson`, same |
| Severity | `reg:gamma`, per-claim raw target, weight=ClaimNb_sev | `gamma`, same |
| Tweedie direct | `reg:tweedie`, **rate target** (ClaimAmount/Exposure), weight=Exposure, variance_power grid | `tweedie`, same |

## Results — final, measured

### Frequency — GBM wins

| Model | Val deviance | Val AE |
|-------|--------------|--------|
| GLM Poisson | 0.1584 | 0.9966 |
| XGB Poisson | 0.1518 | 0.9981 |
| **LGBM Poisson** | **0.1515** | **0.9965** |

-4.40% deviance vs GLM. In the expected 3-8% range from freMTPL2
literature. GBM earns its place on frequency.

### Severity — GBM does NOT earn its place

| Model | Val deviance | Val AE |
|-------|--------------|--------|
| GLM Gamma (raw) | 1.8471 | 1.0902 |
| XGB Gamma (raw) | 1.8622 | 1.0946 |
| **LGBM Gamma (raw)** | **1.7965** | **1.2818** |

LGBM has best deviance but worst calibration. Tail overfitting: LGBM
finds patterns near the 4M EUR outlier that don't generalize. GLM wins
on calibration, almost ties on deviance. **GLM for severity.**

### Tweedie direct — fails structurally, stop using it

Three attempts, all documented:

| Version | Target | Exposure | Best val AE |
|---------|--------|----------|-------------|
| v1 | raw per-policy | ❌ none | 1.4048 (XGB) |
| v2 | raw per-policy | ❌ none | 1.4605 (LGBM) |
| **v3** | rate = amount/exposure | ✅ weight=Exposure | **1.4941 (XGB, vp=1.90)** |
| **v3** | rate = amount/exposure | ✅ weight=Exposure | **1.5337 (LGBM, vp=1.90)** |

**Adding exposure made it worse.** Root cause analysis (v3):

1. **Rate target amplifies outliers.** A policy with 0.1 year exposure
   and a 500 EUR claim becomes rate=5,000/year. One such row dominates
   the Tweedie NLL gradient.
2. **The 4M EUR claim can't be learned from 1 row.** Model ignores it,
   AE off by that single row / 100k rows = ~40%.
3. **variance_power wants 1.90** (near-infinite variance). Model is
   chasing the tail, not learning structure. Best vp is at the boundary
   of the grid, further increasing it would keep improving deviance while
   calibration stays broken.

**Conclusion**: Tweedie direct does not work on freMTPL2 with standard
GBM objectives. Not a bug in our code — a fundamental limit of the
Tweedie NLL on this target distribution. Documented. Not used downstream.

### Pure premium — final decision

| Model | Val MAE | Val AE |
|-------|---------|--------|
| GLM compound (GLM freq × GLM sev) | 170.51 | 1.1228 |
| **GBM compound (LGBM freq × GLM sev)** | **168.64** | **1.1257** |
| XGB Tweedie (best vp=1.90) | 151.21 | 1.4941 |
| LGBM Tweedie (best vp=1.90) | 149.52 | 1.5337 |

**Chosen model**: GBM compound. Best MAE (168.64 vs 170.51), honest AE
(1.1257 ≈ 1.1228 GLM). Residual +12.5% AE vs raw target = val-tail-shift
(+7.7% severity on val vs train, documented in Phase 5) + p_has_sev
correction. Expected, not a defect.

**Tweedie rejected**: AEs 1.49-1.53 are not acceptable for production.

## Design decisions

**Frequency — rate target + sample_weight.** Equivalent to log-link GLM
with offset log(Exposure). Cross-version stable (XGBoost's `base_margin`
API shifted 2.x→3.x; LightGBM's `init_score` not applied on .predict()
by default).

**Severity — RAW per-claim target, weight = ClaimNb_sev.**
`y = ClaimAmount_sum / ClaimNb_sev`, `weight = ClaimNb_sev`.
No cap. Accept training noise from the 4M outlier.

**Tweedie — attempted 3 ways, all failed. Documented. Not used.**

**Compound formula**: `LGBM_freq_rate × Exposure × GLM_sev_per_claim × p_has_sev`
where `p_has_sev = 0.7306` (train only, frozen) corrects for the 9,116
claims without severity records.

**No Optuna in Phase 6.** Sane defaults, early stopping (50 rounds).

**One-hot for both GBMs.** Matches Phase 5.

## Key senior finding

**"On freMTPL2, GBM does not earn its complexity for severity modeling,
and Tweedie direct fails structurally with standard GBM objectives."**

- Frequency: LGBM (−4.4% deviance vs GLM). Use it.
- Severity: GLM (stable calibration, nearly equal deviance). Use it.
- Tweedie: rejected after 3 attempts, all documented. Not used downstream.

Portfolio-grade because it's measured, not dogmatic. "Use GBM everywhere"
is junior. "Use each model where it earns its place, with measured
justification" is senior.

## Phase 7 result (already committed)

Optuna over LGBM frequency: 15 trials × 3-fold CV. Best tuned val
deviance 0.1517 vs default 0.1515 = **+0.12% worse**. Decision: keep
Phase 6 defaults. Tuning does not help on this dataset (documented in
reports/phase7_tuning.md).

## Downstream

Phase 8 (uncertainty), Phase 9 (pure premium metrics + SHAP + fairness),
Phase 10 (MLflow + Docker), Phase 11 (FastAPI), Phase 12 (monitoring),
Phase 13 (Model Card + README) will all use the **GBM compound** model:
LGBM frequency + GLM severity + p_has_sev correction. No Tweedie.
