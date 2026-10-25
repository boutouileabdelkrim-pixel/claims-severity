"""Extend the Phase 10 bundle with Phase 8 quantile models (Phase 11).

Phase 10's log_to_mlflow.py saved lgbm_freq_model + glm_sev_model + q_hat,
but NOT the Phase 8 quantile models. /predict needs them to return proper
80% severity intervals (contract requirement: "prédiction + intervalle").

This script:
- loads models/bundle_v1.pkl
- fits 3 LightGBM quantile models (0.1, 0.5, 0.9) on TRAIN claims
- adds them under "quantile_models" key
- saves back to the same path (overwrites in place)

Idempotent: if "quantile_models" already in bundle, exits without
refitting. Safe to run multiple times.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.uncertainty import fit_quantile_grid
from src.utils.paths import FEATURES_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

BUNDLE_PATH = MODELS_DIR / "bundle_v1.pkl"
CATEGORICALS = ["VehBrand", "Region"]
NUMERICS = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus_log",
    "Density_log",
    "Area_ord",
    "VehGas_diesel",
    "VehAge_zero",
    "VehAge_over_20",
    "BonusMalus_gt100",
    "DrivAge_bucket",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"{BUNDLE_PATH} not found. Run `make mlflow` first.")

    with BUNDLE_PATH.open("rb") as f:
        bundle = pickle.load(f)

    if "quantile_models" in bundle:
        logger.info("Bundle already contains quantile_models. Nothing to do.")
        return

    logger.info("Loading train.parquet...")
    train = pd.read_parquet(FEATURES_DIR / "train.parquet")

    scaler: StandardScaler = bundle["scaler"]
    dummy_cols: list[str] = bundle["dummy_cols"]

    dummies_tr = pd.get_dummies(train[CATEGORICALS], drop_first=True, dtype=np.int8)
    dummies_tr = dummies_tr.reindex(columns=dummy_cols, fill_value=0)
    num_tr = scaler.transform(train[NUMERICS].to_numpy(dtype=float))
    X_tr = np.hstack([num_tr, dummies_tr.to_numpy(dtype=float)])

    mask_tr = (train["ClaimNb_sev"] > 0).to_numpy()
    X_tr_sev = X_tr[mask_tr]
    y_sev_tr = train.loc[mask_tr, "ClaimAmount_sum"].to_numpy(dtype=float)
    y_log = np.log(y_sev_tr)

    logger.info("Fitting 3 quantile models on %d claims (log severity)...", len(y_log))
    quantile_models = fit_quantile_grid(X_tr_sev, y_log, quantiles=(0.1, 0.5, 0.9))
    logger.info("Quantile models fitted.")

    bundle["quantile_models"] = quantile_models
    with BUNDLE_PATH.open("wb") as f:
        pickle.dump(bundle, f)

    size_mb = BUNDLE_PATH.stat().st_size / (1024 * 1024)
    logger.info("Updated %s (%.2f MB)", BUNDLE_PATH, size_mb)


if __name__ == "__main__":
    main()
