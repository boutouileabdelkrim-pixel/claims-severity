# Claims Severity & Cost Prediction (Insurance)

[![CI](https://github.com/boutouileabdelkrim-pixel/claims-severity/actions/workflows/ci.yml/badge.svg)](https://github.com/boutouileabdelkrim-pixel/claims-severity/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/linter-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Portfolio project: predict the **cost of an insurance claim** and its **uncertainty**, then derive a **pure premium** (`frequency × severity`) using the French motor third-party liability dataset `freMTPL2`.

## Why this project

Differentiator vs. previous portfolio projects:

- **Regression on a heavy-tailed target** (severity, not a balanced classification).
- **Actuarial logic**: frequency model with exposure offset, severity model, pure premium, actual/expected ratios.
- **Uncertainty estimation**: quantile regression + conformal prediction with empirical coverage checks.
- **Monetary calibration**: sum of predicted premiums must match sum of observed claims.

## Stack

- **Data / ML**: Python 3.11, NumPy, Pandas, scikit-learn, XGBoost, LightGBM, Optuna, SciPy
- **Explainability**: SHAP
- **Tracking**: MLflow 2.x (file store)
- **Serving**: FastAPI + Uvicorn, Pydantic v2
- **Validation**: Pandera (raw + processed schemas)
- **Monitoring**: Evidently + PSI/KS on features and predictions
- **Dev**: uv, ruff, pytest, pre-commit, GitHub Actions
- **Packaging**: Docker (minimal image, non-root user, healthcheck)

## Quickstart

    curl -LsSf https://astral.sh/uv/install.sh | sh
    git clone git@github.com:boutouileabdelkrim-pixel/claims-severity.git
    cd claims-severity
    make install
    make ci

## Project status

- [x] Phase 0 — Environment setup, repo, CI skeleton
- [x] Phase 1 — Package scaffold, settings, Makefile, README
- [ ] Phase 2 — Data acquisition (OpenML freMTPL2) + schema validation
- [ ] Phase 3 — EDA, cleaning, outlier policy
- [ ] Phase 4 — Actuarial feature engineering
- [ ] Phase 5 — Baselines (Dummy, GLM Poisson/Gamma)
- [ ] Phase 6 — GBM frequency + severity + Tweedie direct
- [ ] Phase 7 — Optuna tuning with grouped CV
- [ ] Phase 8 — Uncertainty (quantile + conformal)
- [ ] Phase 9 — Pure premium, actuarial metrics, SHAP, fairness
- [ ] Phase 10 — MLflow + Docker
- [ ] Phase 11 — FastAPI + batch inference
- [ ] Phase 12 — Monitoring (PSI/KS)
- [ ] Phase 13 — Model Card, README polish, sensitivity analysis

## Project structure

    claims-severity/
    ├── src/
    │   ├── data/          # Loading, validation, cleaning
    │   ├── features/      # Feature engineering
    │   ├── models/        # Frequency, severity, Tweedie, tuning
    │   ├── api/           # FastAPI endpoints
    │   ├── monitoring/    # Drift detection
    │   ├── tracking/      # MLflow helpers
    │   └── utils/         # Paths, config, shared utilities
    ├── tests/             # pytest (offline, synthetic fixtures)
    ├── notebooks/         # EDA and analyses
    ├── reports/           # Generated reports, summaries
    ├── docker/            # Dockerfile, compose
    ├── data/              # Raw, processed, features (gitignored)
    ├── models/            # Serialized models (gitignored)
    ├── mlruns/            # MLflow tracking store (gitignored)
    ├── .github/workflows/ # CI
    ├── pyproject.toml
    ├── uv.lock
    ├── Makefile
    └── README.md

## Dataset

`freMTPL2` from OpenML (French motor third-party liability, ~678k policies):

- `freMTPL2freq` (OpenML 41214): `IDpol`, `ClaimNb`, `Exposure`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Area`, `Density`, `Region`
- `freMTPL2sev` (OpenML 41215): `IDpol`, `ClaimAmount`

**Note**: severities are aggregated per `IDpol` **before** joining, to avoid row duplication.

## License

Code: MIT. Dataset: OpenML terms (to be verified in Phase 2).

## Author

BOUTOUILE Abdelkrim — Data Scientist / ML Engineer. [GitHub profile](https://github.com/boutouileabdelkrim-pixel).
