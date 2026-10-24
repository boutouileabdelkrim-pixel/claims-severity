"""Optuna tuning for LightGBM frequency model (Phase 7).

Only frequency is tuned. Severity (GLM) and Tweedie were deemed not
tunable in Phase 6:
- Severity: GLM default is stable (AE 1.09). No gain expected.
- Tweedie: broken on the 4M EUR outlier (AE 1.40). No tuning will fix it.

CV is 3-fold KFold on train only. Val is touched ONLY once at the end,
for final evaluation of the best params. Test never loaded.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import KFold

from src.models.metrics import poisson_deviance


def build_lgbm_freq_params(trial: optuna.Trial) -> dict:
    """Search space for LGBM frequency (Poisson).

    Deliberately narrow: 15 trials with a wide space is noise, not signal.
    """
    return {
        "objective": "poisson",
        "n_estimators": trial.suggest_categorical("n_estimators", [150, 300]),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.10, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [31, 63]),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [50, 100]),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }


def cv_lgbm_freq(
    X: np.ndarray,
    y: np.ndarray,
    exposure: np.ndarray,
    params: dict,
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[float, float]:
    """K-fold CV on rate target, returns (mean, std) Poisson deviance."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    devs: list[float] = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        e_tr, e_va = exposure[tr_idx], exposure[va_idx]
        mask_tr = e_tr > 1e-9
        mask_va = e_va > 1e-9
        rate_tr = y_tr[mask_tr] / e_tr[mask_tr]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr[mask_tr], rate_tr, sample_weight=e_tr[mask_tr])
        pred_rate = np.clip(model.predict(X_va[mask_va]), 0.0, None)
        pred_count = pred_rate * e_va[mask_va]
        dev = poisson_deviance(y_va[mask_va], pred_count)
        devs.append(float(dev))
    return float(np.mean(devs)), float(np.std(devs))


def fit_final_lgbm_freq(
    X: np.ndarray,
    y: np.ndarray,
    exposure: np.ndarray,
    params: dict,
) -> lgb.LGBMRegressor:
    """Fit final LGBM on full train with best params (no CV)."""
    mask = exposure > 1e-9
    rate = y[mask] / exposure[mask]
    model = lgb.LGBMRegressor(**params)
    model.fit(X[mask], rate, sample_weight=exposure[mask])
    return model
