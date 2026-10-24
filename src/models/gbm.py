"""GBM models for Phase 6: XGBoost + LightGBM.

Three families:
- Frequency: rate target (ClaimNb / Exposure) + sample_weight=Exposure.
  Equivalent to log-link GLM with offset log(Exposure).
- Severity: per-claim raw target = ClaimAmount_sum / ClaimNb_sev,
  sample_weight = ClaimNb_sev. Gamma objective. Raw per-claim evaluated
  by multiplying back by ClaimNb_sev.
- Tweedie direct: rate per exposure unit = ClaimAmount_sum / Exposure,
  sample_weight = Exposure, variance_power tunable. Predict = rate ×
  exposure per policy. This properly accounts for exposure variability,
  unlike a naive per-policy target.

No Optuna in gbm.py (Phase 7 does that separately). Sane defaults.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import xgboost as xgb

RANDOM_STATE = 42


# =========================================================================
# Frequency (rate target, sample_weight=Exposure)
# =========================================================================


def fit_xgb_freq(
    X_train: np.ndarray,
    y_train: np.ndarray,
    exposure_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    exposure_val: np.ndarray,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: float = 10.0,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
) -> xgb.XGBRegressor:
    """XGBoost Poisson on rate target with exposure weight."""
    mask_tr = exposure_train > 1e-9
    mask_va = exposure_val > 1e-9
    rate_tr = y_train[mask_tr] / exposure_train[mask_tr]
    rate_va = y_val[mask_va] / exposure_val[mask_va]

    model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="poisson-nloglik",
    )
    model.fit(
        X_train[mask_tr],
        rate_tr,
        sample_weight=exposure_train[mask_tr],
        eval_set=[(X_val[mask_va], rate_va)],
        sample_weight_eval_set=[exposure_val[mask_va]],
        verbose=False,
    )
    return model


def fit_lgbm_freq(
    X_train: np.ndarray,
    y_train: np.ndarray,
    exposure_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    exposure_val: np.ndarray,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_child_samples: int = 50,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
) -> lgb.LGBMRegressor:
    """LightGBM Poisson on rate target with exposure weight."""
    mask_tr = exposure_train > 1e-9
    mask_va = exposure_val > 1e-9
    rate_tr = y_train[mask_tr] / exposure_train[mask_tr]
    rate_va = y_val[mask_va] / exposure_val[mask_va]

    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_train[mask_tr],
        rate_tr,
        sample_weight=exposure_train[mask_tr],
        eval_set=[(X_val[mask_va], rate_va)],
        eval_sample_weight=[exposure_val[mask_va]],
        eval_metric="poisson",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def predict_freq_count(model, X: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Rate prediction × exposure = expected count per policy."""
    return np.clip(model.predict(X), 0.0, None) * exposure


# =========================================================================
# Severity (per-claim raw target, sample_weight=ClaimNb_sev)
# =========================================================================


def fit_xgb_sev(
    X_train: np.ndarray,
    y_total: np.ndarray,
    n_claims_train: np.ndarray,
    X_val: np.ndarray,
    y_total_val: np.ndarray,
    n_claims_val: np.ndarray,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: float = 10.0,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
) -> xgb.XGBRegressor:
    """XGBoost Gamma on per-claim raw severity with claim weight."""
    y_per_claim = y_total / n_claims_train
    y_per_claim_va = y_total_val / n_claims_val

    model = xgb.XGBRegressor(
        objective="reg:gamma",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="gamma-nloglik",
    )
    model.fit(
        X_train,
        y_per_claim,
        sample_weight=n_claims_train,
        eval_set=[(X_val, y_per_claim_va)],
        sample_weight_eval_set=[n_claims_val],
        verbose=False,
    )
    return model


def fit_lgbm_sev(
    X_train: np.ndarray,
    y_total: np.ndarray,
    n_claims_train: np.ndarray,
    X_val: np.ndarray,
    y_total_val: np.ndarray,
    n_claims_val: np.ndarray,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 50,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
) -> lgb.LGBMRegressor:
    """LightGBM Gamma on per-claim raw severity with claim weight."""
    y_per_claim = y_total / n_claims_train
    y_per_claim_va = y_total_val / n_claims_val

    model = lgb.LGBMRegressor(
        objective="gamma",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_per_claim,
        sample_weight=n_claims_train,
        eval_set=[(X_val, y_per_claim_va)],
        eval_sample_weight=[n_claims_val],
        eval_metric="gamma",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def predict_sev_per_claim(model, X: np.ndarray) -> np.ndarray:
    """Per-claim severity (clipped positive)."""
    return np.clip(model.predict(X), 1e-3, None)


# =========================================================================
# Tweedie direct (RATE per exposure unit, sample_weight=Exposure)
# =========================================================================
#
# Bug fix (post Phase 6): the original version fit on raw per-policy
# ClaimAmount_sum with no exposure information. The model literally could
# not learn to scale by exposure — leading to AE 1.40-1.46. Now:
#   - target = ClaimAmount / Exposure  (rate per unit exposure)
#   - sample_weight = Exposure         (each policy-year counts equally)
#   - predict returns rate, caller multiplies by exposure
#   - variance_power is tunable: [1.1, 1.3, 1.5, 1.7, 1.9] typical
#     for insurance compound Poisson-Gamma; higher = heavier tail fit.
#


def fit_xgb_tweedie(
    X_train: np.ndarray,
    y_train: np.ndarray,
    exposure_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    exposure_val: np.ndarray,
    variance_power: float = 1.5,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: float = 10.0,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
) -> xgb.XGBRegressor:
    """XGBoost Tweedie on rate target with exposure weight."""
    mask_tr = exposure_train > 1e-9
    mask_va = exposure_val > 1e-9
    rate_tr = y_train[mask_tr] / exposure_train[mask_tr]
    rate_va = y_val[mask_va] / exposure_val[mask_va]

    model = xgb.XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=variance_power,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric=f"tweedie-nloglik@{variance_power}",
    )
    model.fit(
        X_train[mask_tr],
        rate_tr,
        sample_weight=exposure_train[mask_tr],
        eval_set=[(X_val[mask_va], rate_va)],
        sample_weight_eval_set=[exposure_val[mask_va]],
        verbose=False,
    )
    return model


def fit_lgbm_tweedie(
    X_train: np.ndarray,
    y_train: np.ndarray,
    exposure_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    exposure_val: np.ndarray,
    variance_power: float = 1.5,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_child_samples: int = 50,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
) -> lgb.LGBMRegressor:
    """LightGBM Tweedie on rate target with exposure weight."""
    mask_tr = exposure_train > 1e-9
    mask_va = exposure_val > 1e-9
    rate_tr = y_train[mask_tr] / exposure_train[mask_tr]
    rate_va = y_val[mask_va] / exposure_val[mask_va]

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=variance_power,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_train[mask_tr],
        rate_tr,
        sample_weight=exposure_train[mask_tr],
        eval_set=[(X_val[mask_va], rate_va)],
        eval_sample_weight=[exposure_val[mask_va]],
        eval_metric="tweedie",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def predict_tweedie(model, X: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Per-policy pure premium = rate prediction × exposure."""
    return np.clip(model.predict(X), 0.0, None) * exposure
