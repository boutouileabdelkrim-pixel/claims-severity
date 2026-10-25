"""FastAPI endpoints for claims-severity (Phase 11).

Endpoints:
- GET  /health         liveness probe (no bundle needed)
- GET  /info           model metadata
- POST /predict        single policy → point + claim probability + interval
- POST /predict/batch  batch (1..1000 policies)

The model bundle is loaded once at startup via the FastAPI lifespan
context manager. If missing (fresh clone, CI without models/), /predict
returns 503 with a clear message. /health still returns 200 so
orchestrators can distinguish "app up, model missing" from "app down".

Bundle contents (models/bundle_v1.pkl):
- lgbm_freq_model: LGBM Poisson (rate target)
- glm_sev_model:   sklearn GammaRegressor (per-claim mean)
- quantile_models: dict {0.1, 0.5, 0.9} LightGBM quantiles (Phase 8)
- p_has_sev, q_hat, scaler, dummy_cols, feature_names, numerics,
  categoricals, version
"""

from __future__ import annotations

import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import StandardScaler

from src.api.schemas import (
    BatchInput,
    BatchOutput,
    InfoOutput,
    PolicyInput,
    PredictionOutput,
)
from src.features.build import engineer_features
from src.models.uncertainty import predict_interval
from src.utils.paths import MODELS_DIR

logger = logging.getLogger(__name__)
BUNDLE_PATH = MODELS_DIR / "bundle_v1.pkl"


def _load_bundle(path: Path) -> dict[str, Any]:
    """Load and validate the model bundle. Raises on missing keys."""
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")
    with path.open("rb") as f:
        bundle = pickle.load(f)
    required = {
        "lgbm_freq_model",
        "glm_sev_model",
        "quantile_models",
        "p_has_sev",
        "q_hat",
        "scaler",
        "dummy_cols",
        "numerics",
        "categoricals",
        "version",
    }
    missing = required - set(bundle.keys())
    if missing:
        raise KeyError(
            f"Bundle missing keys: {sorted(missing)}. "
            "Run `uv run python -m scripts.extend_bundle` to complete it."
        )
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model bundle at startup. Sets app.state.bundle or app.state.bundle=None."""
    try:
        app.state.bundle = _load_bundle(BUNDLE_PATH)
        logger.info("Bundle loaded: %s", BUNDLE_PATH)
    except Exception as e:
        logger.warning("Bundle NOT loaded: %s", e)
        app.state.bundle = None
    yield


app = FastAPI(
    title="claims-severity",
    version="0.2.0",
    description="Motor claims severity + pure premium prediction",
    lifespan=lifespan,
)


def _require_bundle() -> dict[str, Any]:
    """Return app.state.bundle or raise 503."""
    if app.state.bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model bundle not loaded. Check /health or server logs.",
        )
    return app.state.bundle


def _encode_policies(inputs: list[PolicyInput], bundle: dict[str, Any]) -> np.ndarray:
    """Encode PolicyInput list into the exact GLM/GBM feature matrix.

    Uses the training-time functions (engineer_features, scaler, one-hot)
    to guarantee feature parity with Phase 6.
    """
    df = pd.DataFrame([i.model_dump() for i in inputs])
    df = engineer_features(df)
    scaler: StandardScaler = bundle["scaler"]
    dummy_cols: list[str] = bundle["dummy_cols"]
    numerics: list[str] = bundle["numerics"]
    categoricals: list[str] = bundle["categoricals"]

    num = scaler.transform(df[numerics].to_numpy(dtype=float))
    dummies = pd.get_dummies(df[categoricals], drop_first=True, dtype=np.int8)
    dummies = dummies.reindex(columns=dummy_cols, fill_value=0)
    return np.hstack([num, dummies.to_numpy(dtype=float)])


def _predict_batch(inputs: list[PolicyInput], bundle: dict[str, Any]) -> list[PredictionOutput]:
    """Core prediction routine, shared by /predict and /predict/batch."""
    X = _encode_policies(inputs, bundle)

    lgbm_f = bundle["lgbm_freq_model"]
    glm_s = bundle["glm_sev_model"]
    q_models = bundle["quantile_models"]
    p_has_sev: float = float(bundle["p_has_sev"])
    q_hat: float = float(bundle["q_hat"])

    rate = np.clip(lgbm_f.predict(X), 0.0, None)
    exposure = np.array([i.Exposure for i in inputs], dtype=float)
    expected_claims = rate * exposure
    claim_probability = 1.0 - np.exp(-expected_claims)

    sev_mean = np.clip(glm_s.predict(X), 1e-3, None)

    lo_log, med_log, hi_log = predict_interval(q_models, X, q_hat=q_hat)
    sev_lo = np.clip(np.exp(lo_log), 1e-3, None)
    sev_med = np.clip(np.exp(med_log), 1e-3, None)
    sev_hi = np.clip(np.exp(hi_log), 1e-3, None)

    pure_premium = rate * exposure * p_has_sev * sev_mean

    out: list[PredictionOutput] = []
    for i in range(len(inputs)):
        out.append(
            PredictionOutput(
                pure_premium=float(pure_premium[i]),
                claim_probability=float(claim_probability[i]),
                severity_median=float(sev_med[i]),
                severity_lo_80=float(sev_lo[i]),
                severity_hi_80=float(sev_hi[i]),
            )
        )
    return out


@app.get("/health")
def health() -> dict:
    """Liveness probe. Returns 200 always if the app is up."""
    return {"status": "ok"}


@app.get("/info", response_model=InfoOutput)
def info() -> InfoOutput:
    """Model metadata. Works even if bundle is missing (reports nulls)."""
    bundle = app.state.bundle
    if bundle is None:
        return InfoOutput(
            version="unknown",
            model="not-loaded",
            features=0,
            endpoints=["/health", "/info", "/predict", "/predict/batch"],
            bundle_loaded=False,
        )
    return InfoOutput(
        version=str(bundle["version"]),
        model="compound (LGBM freq × GLM sev × p_has_sev)",
        features=len(bundle["feature_names"]),
        endpoints=["/health", "/info", "/predict", "/predict/batch"],
        bundle_loaded=True,
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(policy: PolicyInput) -> PredictionOutput:
    """Predict one policy: pure premium + claim probability + severity interval."""
    bundle = _require_bundle()
    return _predict_batch([policy], bundle)[0]


@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(payload: BatchInput) -> BatchOutput:
    """Predict up to 1000 policies in a single call."""
    bundle = _require_bundle()
    preds = _predict_batch(payload.policies, bundle)
    return BatchOutput(predictions=preds, n=len(preds))
