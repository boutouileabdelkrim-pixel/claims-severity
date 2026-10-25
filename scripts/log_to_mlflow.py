"""Log the frozen compound model + artifacts to MLflow (Phase 10).

- Loads train + val only (test is frozen after Phase 9).
- Refits LGBM frequency + GLM Gamma severity.
- Computes val metrics (Poisson dev, AE, MAE, Gamma dev).
- Saves bundle to models/bundle_v1.pkl for Phase 11 FastAPI to load.
- Logs params, metrics, artifacts, and the pickle bundle to MLflow.

Test metrics are NOT recomputed here. The Phase 9 report
(reports/phase9_final_evaluation.txt) is logged as an artifact and is
the authoritative source for test performance.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.baseline import (
    fit_gamma_sev,
)
from src.models.baseline import (
    predict_severity_per_claim as predict_glm_sev,
)
from src.models.gbm import fit_lgbm_freq, predict_freq_count
from src.models.metrics import (
    actual_expected_ratio,
    gamma_deviance,
    mae,
    poisson_deviance,
)
from src.tracking.mlflow_client import log_run
from src.utils.config import SETTINGS
from src.utils.paths import FEATURES_DIR, MODELS_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

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

Q_HAT_PHASE8 = 0.2714
BUNDLE_PATH = MODELS_DIR / "bundle_v1.pkl"

ARTIFACTS = [
    REPORTS_DIR / "phase3_eda.txt",
    REPORTS_DIR / "phase5_baselines.txt",
    REPORTS_DIR / "phase6_gbm.txt",
    REPORTS_DIR / "phase8_uncertainty.txt",
    REPORTS_DIR / "phase9_final_evaluation.txt",
    REPORTS_DIR / "phase9_shap_summary.png",
    REPORTS_DIR / "phase9_lift_chart.png",
]


def build_matrix(train: pd.DataFrame, val: pd.DataFrame):
    dummies_tr = pd.get_dummies(train[CATEGORICALS], drop_first=True, dtype=np.int8)
    cols = list(dummies_tr.columns)
    dummies_va = pd.get_dummies(val[CATEGORICALS], drop_first=True, dtype=np.int8).reindex(
        columns=cols, fill_value=0
    )
    scaler = StandardScaler()
    num_tr = scaler.fit_transform(train[NUMERICS].to_numpy(dtype=float))
    num_va = scaler.transform(val[NUMERICS].to_numpy(dtype=float))
    X_tr = np.hstack([num_tr, dummies_tr.to_numpy(dtype=float)])
    X_va = np.hstack([num_va, dummies_va.to_numpy(dtype=float)])
    feature_names = NUMERICS + cols
    return X_tr, X_va, scaler, cols, feature_names


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    logger.info("Loaded train %s, val %s", train.shape, val.shape)

    X_tr, X_va, scaler, dummy_cols, feature_names = build_matrix(train, val)

    exp_tr = train["Exposure"].to_numpy(dtype=float)
    exp_va = val["Exposure"].to_numpy(dtype=float)
    y_freq_tr = train["ClaimNb"].to_numpy(dtype=float)
    y_freq_va = val["ClaimNb"].to_numpy(dtype=float)

    logger.info("Fitting LGBM frequency...")
    lgbm_f = fit_lgbm_freq(X_tr, y_freq_tr, exp_tr, X_va, y_freq_va, exp_va)

    m_tr = (train["ClaimNb_sev"] > 0).to_numpy()
    raw_sev_tr = train.loc[m_tr, "ClaimAmount_sum"].to_numpy(dtype=float)
    n_tr = train.loc[m_tr, "ClaimNb_sev"].to_numpy(dtype=float)

    logger.info("Fitting GLM Gamma severity...")
    glm_s = fit_gamma_sev(X_tr[m_tr], raw_sev_tr, n_tr)

    p_has_sev = float(train["ClaimNb_sev"].sum()) / float(train["ClaimNb"].sum())

    # Val metrics only (test is frozen)
    freq_rate_va = predict_freq_count(lgbm_f, X_va, exp_va) / np.clip(exp_va, 1e-9, None)
    sev_va = predict_glm_sev(glm_s, X_va)
    pp_va = freq_rate_va * exp_va * sev_va * p_has_sev
    y_pp_va = val["ClaimAmount_sum"].fillna(0.0).to_numpy(dtype=float)

    m_va = (val["ClaimNb_sev"] > 0).to_numpy()
    raw_sev_va = val.loc[m_va, "ClaimAmount_sum"].to_numpy(dtype=float)
    n_va = val.loc[m_va, "ClaimNb_sev"].to_numpy(dtype=float)

    metrics = {
        "val_poisson_deviance": poisson_deviance(y_freq_va, freq_rate_va * exp_va),
        "val_freq_ae": actual_expected_ratio(y_freq_va, freq_rate_va * exp_va),
        "val_sev_gamma_deviance": gamma_deviance(
            raw_sev_va / n_va, sev_va[m_va], sample_weight=n_va
        ),
        "val_sev_ae": actual_expected_ratio(raw_sev_va / n_va, sev_va[m_va]),
        "val_pp_mae": mae(y_pp_va, pp_va),
        "val_pp_ae": actual_expected_ratio(y_pp_va, pp_va),
    }

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": SETTINGS.seed,
        "p_has_sev": p_has_sev,
        "q_hat_phase8": Q_HAT_PHASE8,
        "n_train_rows": int(len(train)),
        "n_val_rows": int(len(val)),
        "n_train_claims": int(m_tr.sum()),
    }

    bundle = {
        "lgbm_freq_model": lgbm_f,
        "glm_sev_model": glm_s,
        "p_has_sev": p_has_sev,
        "q_hat": Q_HAT_PHASE8,
        "scaler": scaler,
        "dummy_cols": dummy_cols,
        "feature_names": feature_names,
        "categoricals": CATEGORICALS,
        "numerics": NUMERICS,
        "version": "0.1.0",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with BUNDLE_PATH.open("wb") as f:
        pickle.dump(bundle, f)
    logger.info("Saved bundle to %s", BUNDLE_PATH)

    run_id = log_run(
        tracking_uri=SETTINGS.mlflow_tracking_uri,
        experiment_name=SETTINGS.mlflow_experiment,
        run_name="phase10-compound-lgbm-freq-glm-sev",
        params=params,
        metrics=metrics,
        artifact_paths=ARTIFACTS,
        model_bundle=bundle,
        tags={"phase": "10", "model": "compound", "version": "0.1.0"},
    )

    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    p("=" * 80)
    p("PHASE 10 — MLFLOW LOGGING")
    p("=" * 80)
    p()
    p(f"Tracking URI: {SETTINGS.mlflow_tracking_uri}")
    p(f"Experiment:   {SETTINGS.mlflow_experiment}")
    p(f"Run ID:       {run_id}")
    p(f"Bundle:       {BUNDLE_PATH}")
    p()
    p("=== Val metrics logged ===")
    for k, v in metrics.items():
        p(f"  {k:<28s} {v:.4f}")
    p()
    p("=== Params logged ===")
    for k, v in params.items():
        if isinstance(v, float):
            p(f"  {k:<24s} {v:.6f}")
        else:
            p(f"  {k:<24s} {v}")
    p()
    p("=== Artifacts logged (if present) ===")
    for a in ARTIFACTS:
        status = "OK" if a.exists() else "MISSING"
        p(f"  {status:<8s} {a.name}")

    OUT = REPORTS_DIR / "phase10_mlflow.txt"
    OUT.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
