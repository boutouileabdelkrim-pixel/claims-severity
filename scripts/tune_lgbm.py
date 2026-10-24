"""End-to-end Optuna tuning for LGBM frequency (Phase 7).

Trains only on train. Uses 3-fold CV. Val touched once at end.
Test never loaded.

Budget: 15 trials, 3-fold, timeout=1500s (25 min).
If improvement < 2% over Phase 6 default, tuning is dropped.

Writes:
- reports/phase7_tuning.txt (all trials + best)
- reports/phase7_best_params.json (only if tuning wins)
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.metrics import actual_expected_ratio, poisson_deviance
from src.models.tuning import (
    build_lgbm_freq_params,
    cv_lgbm_freq,
    fit_final_lgbm_freq,
)
from src.utils.paths import FEATURES_DIR, REPORTS_DIR

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

N_TRIALS = 15
CV_FOLDS = 3
TIMEOUT_SEC = 1500


def build_matrix(train: pd.DataFrame, val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Same matrix as Phase 6 (one-hot cats + scaled numerics)."""
    dummies_tr = pd.get_dummies(train[CATEGORICALS], drop_first=True, dtype=np.int8)
    cols = list(dummies_tr.columns)
    dummies_va = pd.get_dummies(val[CATEGORICALS], drop_first=True, dtype=np.int8)
    dummies_va = dummies_va.reindex(columns=cols, fill_value=0)
    scaler = StandardScaler()
    num_tr = scaler.fit_transform(train[NUMERICS].to_numpy(dtype=float))
    num_va = scaler.transform(val[NUMERICS].to_numpy(dtype=float))
    X_tr = np.hstack([num_tr, dummies_tr.to_numpy(dtype=float)])
    X_va = np.hstack([num_va, dummies_va.to_numpy(dtype=float)])
    return X_tr, X_va


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    logger.info("Loaded train %s, val %s", train.shape, val.shape)

    X_tr, X_va = build_matrix(train, val)
    y_tr = train["ClaimNb"].to_numpy(dtype=float)
    y_va = val["ClaimNb"].to_numpy(dtype=float)
    e_tr = train["Exposure"].to_numpy(dtype=float)
    e_va = val["Exposure"].to_numpy(dtype=float)

    def objective(trial: optuna.Trial) -> float:
        params = build_lgbm_freq_params(trial)
        mean_dev, _ = cv_lgbm_freq(X_tr, y_tr, e_tr, params, n_splits=CV_FOLDS)
        return mean_dev

    t0 = time.time()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC)
    elapsed = time.time() - t0
    logger.info(
        "Optuna done: %d trials in %.1fs (%.2f min)",
        len(study.trials),
        elapsed,
        elapsed / 60,
    )

    best_params = dict(study.best_params)
    best_cv_dev = float(study.best_value)
    best_params["objective"] = "poisson"
    best_params["random_state"] = 42
    best_params["verbose"] = -1
    best_params["n_jobs"] = -1

    # Final eval on val with best params (single shot)
    tuned_model = fit_final_lgbm_freq(X_tr, y_tr, e_tr, best_params)
    val_rate = np.clip(tuned_model.predict(X_va), 0.0, None)
    val_count = val_rate * e_va
    tuned_val_dev = float(poisson_deviance(y_va, val_count))
    tuned_val_ae = float(actual_expected_ratio(y_va, val_count))

    # Phase 6 default LGBM freq (val deviance from reports/phase6_gbm.txt)
    default_val_dev = 0.1515
    default_val_ae = 0.9965
    improvement_pct = (tuned_val_dev - default_val_dev) / default_val_dev * 100

    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    p("=" * 90)
    p("PHASE 7 — OPTUNA TUNING (LightGBM frequency)")
    p("=" * 90)
    p()
    p(f"Trials: {len(study.trials)} (limit {N_TRIALS}, timeout {TIMEOUT_SEC}s)")
    p(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    p(f"CV folds: {CV_FOLDS}")
    p()
    p(f"Best CV Poisson deviance: {best_cv_dev:.4f}")
    p("Best params:")
    for k, v in study.best_params.items():
        p(f"  {k}: {v}")
    p()
    p(f"Val Poisson deviance (tuned):   {tuned_val_dev:.4f}")
    p(f"Val Poisson deviance (default): {default_val_dev:.4f}")
    p(f"Val AE (tuned):                 {tuned_val_ae:.4f}")
    p(f"Val AE (default):               {default_val_ae:.4f}")
    p(f"Delta vs default (val):         {improvement_pct:+.2f}%")
    p()

    save_params = False
    if improvement_pct <= -2.0:
        p("DECISION: tuned model earns its place (>=2% deviance gain).")
        p("Best params saved for downstream use.")
        save_params = True
    elif improvement_pct < 0.0:
        p("DECISION: tuned model is marginally better (<2%).")
        p("Tuning not worth its cost. Keep Phase 6 defaults.")
    else:
        p("DECISION: tuned model is not better than default.")
        p("Tuning does not help. Keep Phase 6 defaults.")

    p()
    p("=== Trial history ===")
    p(f"{'trial':>6s} {'state':>10s} {'value':>10s}  params")
    for t in study.trials:
        state = t.state.name
        val_s = f"{t.value:.4f}" if t.value is not None else "-"
        params_short = json.dumps(t.params)[:70]
        p(f"{t.number:>6d} {state:>10s} {val_s:>10s}  {params_short}")

    OUT = REPORTS_DIR / "phase7_tuning.txt"
    OUT.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", OUT)

    if save_params:
        params_path = REPORTS_DIR / "phase7_best_params.json"
        params_path.write_text(json.dumps(best_params, indent=2))
        logger.info("Wrote %s", params_path)


if __name__ == "__main__":
    main()
