# Claims Severity & Cost Prediction (Insurance)

[![CI](https://github.com/boutouileabdelkrim-pixel/claims-severity/actions/workflows/ci.yml/badge.svg)](https://github.com/boutouileabdelkrim-pixel/claims-severity/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/linter-ruff-000000.svg)](https://github.com/astral-sh/ruff)

End-to-end **motor insurance pricing** on freMTPL2 (French third-party
liability, ~678k policies). Predicts pure premium with uncertainty, serves
via FastAPI, tracks with MLflow, and monitors for drift — with measured
numbers and documented limitations, not just a notebook.

## Executive summary

- **Compound model**: LightGBM frequency × GLM Gamma severity ×
  `p_has_sev` correction. **Test AE = 0.9909** (actual/predicted ratio —
  near-perfect calibration) on 101,702 held-out policies.
- **Uncertainty**: Conformalized Quantile Regression (Romano 2019) on
  severity, `q_hat = 0.2714` frozen. **Empirical 80% coverage = 0.7975**
  on test claims.
- **Honest findings** (all documented in `reports/`): weak ranking
  (Gini 0.26), Tweedie direct fails on this dataset (3 attempts,
  documented), feature-space boundary at `DrivAge = 101`, PSI misses
  tail shift on mass-at-zero targets (KS catches it).

## What the model does

Given a policy (vehicle + driver + region), returns:

- **Pure premium** (expected claim cost per policy, EUR)
- **Claim probability** (Poisson: `1 - exp(-rate × exposure)`)
- **80% severity interval** (conditional on a claim occurring)
- **Point estimate median severity**

Example — live endpoint, real prediction:

    curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
      "VehPower": 5, "VehAge": 2, "DrivAge": 45, "BonusMalus": 50,
      "VehBrand": "B12", "VehGas": "Regular", "Area": "D",
      "Density": 1217, "Region": "R82", "Exposure": 0.8
    }'

Response:

    {
      "pure_premium": 69.14,
      "claim_probability": 0.0689,
      "severity_median": 1171.42,
      "severity_lo_80": 147.71,
      "severity_hi_80": 3333.65,
      "interval_source": "CQR severity (Phase 8), q_hat=0.2714, conditional on claim"
    }

## Results (test split, n = 101,702)

| Metric | Value | Comment |
|--------|------:|---------|
| Pure premium MAE | 157.16 EUR | per-policy absolute error |
| **Pure premium AE** | **0.9909** | calibration, 1.0 = perfect |
| Pure premium Gini | 0.2635 | ranking quality, honest weak |
| Frequency Poisson dev. | 0.1507 | −4.4% vs GLM Poisson baseline |
| Severity Gamma dev. | 1.6632 | ≈ GLM (GBM does not help severity) |
| **CQR 80% coverage** | **0.7975** | on test claims, nominal 80% |

Full breakdown including per-decile lift, SHAP, and fairness:
`reports/phase9_final_evaluation.txt`.

## Quickstart

    # 1. Install (Python 3.11 + uv required)
    git clone git@github.com:boutouileabdelkrim-pixel/claims-severity.git
    cd claims-severity
    make install

    # 2. Full pipeline end-to-end (download, clean, features, models)
    make data validate features baselines gbm mlflow
    uv run python -m scripts.extend_bundle   # adds Phase 8 quantile models

    # 3. Serve the API
    uv run uvicorn src.api.main:app --port 8000

    # 4. Test
    make test    # 124 tests
    make ci      # lint + format-check + test

## Live demo

    uv run uvicorn src.api.main:app --port 8000 &

    curl http://localhost:8000/health
    # {"status":"ok"}

    curl http://localhost:8000/info
    # {"version":"0.1.0","model":"compound (LGBM freq × GLM sev × p_has_sev)",
    #  "features":42,"endpoints":[...],"bundle_loaded":true}

    curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
      -d '{"VehPower":5,"VehAge":2,"DrivAge":45,"BonusMalus":50,
           "VehBrand":"B12","VehGas":"Regular","Area":"D","Density":1217,
           "Region":"R82","Exposure":0.8}'

## Architecture

    claims-severity/
    ├── src/
    │   ├── data/          # Download, validate, merge (pandera schemas)
    │   ├── features/      # Engineer, split, EDA
    │   ├── models/        # Baselines, GBM, uncertainty, actuarial, fairness
    │   ├── api/           # FastAPI + Pydantic v2
    │   ├── monitoring/    # PSI + KS drift detection
    │   ├── tracking/      # MLflow wrapper
    │   └── utils/         # Paths, config
    ├── scripts/           # Pipeline CLIs (report, train, tune, log, monitor)
    ├── tests/             # 124 offline tests (pytest)
    ├── reports/           # Phase-by-phase measured results
    ├── docs/MODEL_CARD.md # Full model card: metrics, fairness, limitations
    ├── Dockerfile         # python:3.11-slim, non-root, healthcheck
    ├── docker-compose.yml
    ├── pyproject.toml     # deps pinned via uv.lock
    └── Makefile           # 15 targets

## The 14-phase build

Every phase measured, documented, and committed. Reports in `reports/`.

- [x] **Phase 0** — Environment + repo + CI
- [x] **Phase 1** — Scaffold, Settings, Makefile
- [x] **Phase 2** — freMTPL2 download + Pandera schemas
- [x] **Phase 3** — EDA + outlier policy (Gini 0.6475, top 0.1% = 60% mass)
- [x] **Phase 4** — Features + stratified splits + p99.5 cap
- [x] **Phase 5** — Baselines (Dummy, GLM Poisson, GLM Gamma)
- [x] **Phase 6** — GBM (LGBM freq −4.4% dev, Tweedie rejected after 3 attempts)
- [x] **Phase 7** — Optuna (honest flat result, defaults kept)
- [x] **Phase 8** — CQR uncertainty (72.8% → 80.1% coverage restored)
- [x] **Phase 9** — Final test eval + SHAP + fairness + test frozen
- [x] **Phase 10** — MLflow + Docker (image built, /health verified)
- [x] **Phase 11** — FastAPI /predict + /predict/batch
- [x] **Phase 12** — PSI + KS drift monitoring
- [x] **Phase 13** — Model Card + README + Docker CI

## Model, in one paragraph

Frequency model: LightGBM Poisson on rate target (`ClaimNb / Exposure`)
with `sample_weight=Exposure`, equivalent to a log-link GLM with offset
`log(Exposure)`. Severity model: scikit-learn `GammaRegressor` on
per-claim raw target (`ClaimAmount_sum / ClaimNb_sev`) with
`sample_weight=ClaimNb_sev`. Compound formula:
`pure_premium = rate × exposure × p_has_sev × severity_per_claim`, where
`p_has_sev = 0.7306` corrects for the 9,116 freMTPL2 policies that
report a claim but no severity amount. Uncertainty: 3 LightGBM
quantiles (0.1/0.5/0.9) on `log(severity)` for claims only, plus CQR
conformal calibration on a val split (`q_hat = 0.2714`), giving
finite-sample 80% coverage guarantee.

## Honest limitations

- **Weak ranking (Gini 0.26)**: model is well-calibrated globally but
  cannot sharply order policies. freMTPL2 lacks the discriminative
  features (telematics, credit, claims history) that modern motor
  pricing uses.
- **Tweedie direct fails structurally**: 3 attempts (raw, p99.5 cap,
  rate target + exposure), all worse than compound. Documented in
  `reports/phase6_gbm.md`.
- **Fairness gaps**: DrivAge bucket 4 (75+) AE = 3.17, Region R93
  AE = 1.87. Descriptive only, not a legal claim.
- **Feature boundary**: `DrivAge > 100` crashes `engineer_features`.
  API schema enforces ≤ 100. Documented in `reports/phase12_monitoring.md`.
- **PSI misses tail shift on mass-at-zero**: `pure_premium` is 95% zero.
  PSI says stable, KS says significant (p = 6.7e-84). Use both.
  Documented.

Full details in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Stack

Python 3.11 · uv · numpy 1.26 · pandas 2.3 · scikit-learn 1.9 ·
LightGBM 4.7 · XGBoost 3.2 · Optuna 5.0 · SHAP 0.49 · MLflow 2.22 ·
FastAPI 0.159 · Pydantic 2.14 · Pandera 0.34 · Ruff 0.17 · pytest 9.1 ·
pre-commit 4.7 · Docker · GitHub Actions

## Reports (all measured, no invented numbers)

- `reports/phase2_summary.md` — data cleaning decisions
- `reports/phase3_eda.txt` — Gini, quantiles, concentration
- `reports/phase3_outliers.md` — cap policy decision
- `reports/phase5_baselines.txt` — dummy + GLM results
- `reports/phase6_gbm.md` — GBM results + Tweedie history
- `reports/phase7_tuning.txt` — Optuna trials
- `reports/phase8_uncertainty.txt` — CQR coverage verification
- `reports/phase9_final_evaluation.txt` — test metrics + lift + SHAP
- `reports/phase9_shap_summary.png` — feature importance
- `reports/phase9_lift_chart.png` — per-decile calibration
- `reports/phase10_mlflow.md` — MLflow run details + Docker build
- `reports/phase11_api.md` — endpoint contracts
- `reports/phase12_monitoring.md` — PSI/KS + combined verdict principle

## License

Code: MIT. Dataset: OpenML terms. See
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) for usage restrictions and
ethical considerations.

## Author

Boutouile Abdelkrim — Data Scientist / ML Engineer.
[GitHub](https://github.com/boutouileabdelkrim-pixel)
