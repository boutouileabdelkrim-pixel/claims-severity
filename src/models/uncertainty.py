"""Uncertainty quantification for pure premium (Phase 8).

Two-stage pipeline:
1. Quantile regression: 3 LightGBM models on log1p(pure_premium) for
   q in {0.1, 0.5, 0.9}. Target log1p handles the heavy tail and the
   mass-at-zero.
2. CQR (Conformalized Quantile Regression, Romano et al. 2019):
   finite-sample correction using a calibration set. Guarantees
   marginal coverage >= 1 - alpha.

Why conformal: raw quantile regression is only asymptotically valid.
Finite-sample coverage is typically below nominal, especially with
mixture distributions (95% of policies have zero pure premium). CQR
adds a non-negative `q_hat` to widen the interval, restoring the
guarantee.

CQR conformity score for interval [q_lo(x), q_hi(x)] and true y:
    s(x, y) = max(q_lo(x) - y, y - q_hi(x))
    q_hat = quantile(s, ceil((n+1)(1-alpha))/n)   on calibration set
Adjusted interval: [q_lo(x) - q_hat, q_hi(x) + q_hat]
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np


def fit_quantile_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantile: float,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_child_samples: int = 50,
    random_state: int = 42,
) -> lgb.LGBMRegressor:
    """Fit a single LightGBM quantile model."""
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def fit_quantile_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    **fit_kwargs,
) -> dict[float, lgb.LGBMRegressor]:
    """Fit one model per quantile level. Returns {q: model}."""
    return {q: fit_quantile_lgbm(X_train, y_train, q, **fit_kwargs) for q in quantiles}


def predict_quantiles(
    models: dict[float, lgb.LGBMRegressor], X: np.ndarray
) -> dict[float, np.ndarray]:
    """Predict each quantile on X. Returns {q: predictions}."""
    return {q: model.predict(X) for q, model in models.items()}


def cqr_conformity_scores(
    q_lo_pred: np.ndarray,
    q_hi_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """CQR conformity scores: max(q_lo - y, y - q_hi).

    Positive when y is outside [q_lo, q_hi]; negative when inside
    (with magnitude equal to the slack).
    """
    return np.maximum(q_lo_pred - y_true, y_true - q_hi_pred)


def cqr_adjustment(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected quantile of conformity scores.

    Uses ceil((n+1)(1-alpha))/n as the quantile level, the standard
    Romano 2019 correction ensuring coverage >= 1-alpha with
    probability >= 1-alpha (exchangeable data).
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Empty conformity score array")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    level = float(np.ceil((n + 1) * (1 - alpha)) / n)
    level = min(level, 1.0)
    return float(np.quantile(scores, level))


def calibrate_cqr(
    models: dict[float, lgb.LGBMRegressor],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float = 0.2,
) -> float:
    """Convenience: predict lo/hi on calibration set, return q_hat."""
    preds = predict_quantiles(models, X_cal)
    q_lo = preds[0.1]
    q_hi = preds[0.9]
    scores = cqr_conformity_scores(q_lo, q_hi, y_cal)
    return cqr_adjustment(scores, alpha)


def predict_interval(
    models: dict[float, lgb.LGBMRegressor],
    X: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lo_adjusted, median, hi_adjusted) in log1p space."""
    preds = predict_quantiles(models, X)
    lo = preds[0.1] - q_hat
    med = preds[0.5]
    hi = preds[0.9] + q_hat
    return lo, med, hi


def empirical_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of y_true in [lo, hi]."""
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def interval_width(lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """(mean_width, median_width) in the same units as lo/hi."""
    widths = hi - lo
    return float(np.mean(widths)), float(np.median(widths))
