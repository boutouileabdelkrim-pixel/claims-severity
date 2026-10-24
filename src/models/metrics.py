"""Evaluation metrics for Phase 5-6 baselines and GBM models.

Standard regression metrics plus actuarial ones: actual/expected ratio,
Poisson and Gamma deviance, Tweedie deviance. No RMSE as primary metric
— the tail makes it useless (see docs/CONTRACT.md).
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (secondary metric, tail-dominated)."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def poisson_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Mean Poisson deviance. Ignores zero-pred-with-nonzero-true cases."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 1e-10, None)
    dev = np.where(y > 0, y * np.log(np.maximum(y, 1e-10) / p), 0.0) - (y - p)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        return float(np.sum(w * dev) / np.sum(w))
    return float(np.mean(dev))


def gamma_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Mean Gamma deviance (assumes strictly positive target)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 1e-10, None)
    dev = 2.0 * ((y - p) / p - np.log(np.maximum(y, 1e-10) / p))
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        return float(np.sum(w * dev) / np.sum(w))
    return float(np.mean(dev))


def tweedie_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p: float = 1.5,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Mean Tweedie deviance for power p in (1, 2).

    Standard closed-form (per scikit-learn docs):
      dev_i = 2 * (y_i^(2-p)/((1-p)(2-p)) - y_i * mu_i^(1-p)/(1-p)
                   + mu_i^(2-p)/(2-p))
    with y_i=0 handled cleanly. For p=1.5 this is:
      2 * (y_i^0.5 / (0.5 * -0.5) - y_i * mu_i^-0.5 / -0.5
           + mu_i^0.5 / 0.5)
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(y_pred, dtype=float), 1e-10, None)
    if not (1.0 < p < 2.0):
        raise ValueError(f"p must be in (1, 2), got {p}")

    y_term = np.where(y > 0, np.power(np.maximum(y, 1e-10), 2.0 - p), 0.0)
    dev = 2.0 * (
        y_term / ((1.0 - p) * (2.0 - p))
        - y * np.power(mu, 1.0 - p) / (1.0 - p)
        + np.power(mu, 2.0 - p) / (2.0 - p)
    )

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        return float(np.sum(w * dev) / np.sum(w))
    return float(np.mean(dev))


def actual_expected_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Total actual / total predicted. 1.0 = perfectly calibrated overall."""
    s_pred = float(np.sum(y_pred))
    if s_pred == 0:
        return float("inf") if np.sum(y_true) > 0 else 0.0
    return float(np.sum(y_true) / s_pred)
