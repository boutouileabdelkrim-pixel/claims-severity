"""Baseline models: Dummy + GLM Poisson (frequency) + GLM Gamma (severity).

Design choices (documented for interview defense):
- sklearn PoissonRegressor has NO offset. Fitting on RATE target
  (ClaimNb / Exposure) with sample_weight = Exposure is equivalent to
  a log-link GLM with offset log(Exposure). Predictions are rates;
  count = rate * exposure.
- Severity target: RAW ClaimAmount_sum per claim (ClaimAmount_sum /
  ClaimNb_sev). NOT capped — capping is a Phase 6 training concern for
  Gamma GBM stability. Baseline must be the simplest correct model.
- Pure premium composition must correct for the 9,116 freMTPL2 claims
  with no sev record:
      pure_premium = rate * p_has_sev * exposure * sev_per_claim
  where p_has_sev = sum(ClaimNb_sev) / sum(ClaimNb) on train (~0.73).
  Without this correction, pure premium over-predicts by 1/p_has_sev.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor

TARGET_FREQ: str = "ClaimNb"
TARGET_SEV_TOTAL: str = "ClaimAmount_sum"
TARGET_SEV_PER_CLAIM: str = "_sev_per_claim"
EXPOSURE: str = "Exposure"
N_CLAIMS_COL: str = "ClaimNb_sev"


def compute_p_has_sev(df: pd.DataFrame) -> float:
    """Fraction of claims that have a sev record: sum(ClaimNb_sev)/sum(ClaimNb)."""
    total_claims = float(df[TARGET_FREQ].sum())
    if total_claims == 0:
        return 0.0
    return float(df[N_CLAIMS_COL].sum()) / total_claims


class DummyFrequency:
    """Constant rate baseline. Predicts rate * exposure."""

    def __init__(self) -> None:
        self.rate_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "DummyFrequency":
        self.rate_ = float(df[TARGET_FREQ].sum() / df[EXPOSURE].sum())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.rate_ * df[EXPOSURE].to_numpy(dtype=float)


class DummySeverity:
    """Constant per-claim mean baseline (weights by ClaimNb_sev)."""

    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "DummySeverity":
        mask = df[N_CLAIMS_COL] > 0
        sub = df.loc[mask]
        weights = sub[N_CLAIMS_COL].to_numpy(dtype=float)
        values = sub[TARGET_SEV_TOTAL].to_numpy(dtype=float)
        self.mean_ = float(np.sum(weights * values / weights) / np.sum(weights))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean_, dtype=float)


def fit_poisson_freq(
    X: np.ndarray, y_total: np.ndarray, exposure: np.ndarray, alpha: float = 1e-4
) -> PoissonRegressor:
    """Fit PoissonRegressor on rate target (ClaimNb / Exposure).

    sample_weight = Exposure is equivalent to a log-link GLM with offset
    log(Exposure). Predict returns rate; multiply by exposure for count.
    Policies with exposure <= 0 are excluded (should not occur per schema).
    """
    mask = exposure > 1e-9
    rate = y_total[mask] / exposure[mask]
    model = PoissonRegressor(alpha=alpha, max_iter=2000, tol=1e-6, solver="lbfgs")
    model.fit(X[mask], rate, sample_weight=exposure[mask])
    return model


def predict_freq_count(model: PoissonRegressor, X: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Convert rate prediction to expected count per policy."""
    return model.predict(X) * exposure


def fit_gamma_sev(
    X: np.ndarray,
    y_total: np.ndarray,
    n_claims: np.ndarray,
    alpha: float = 1e-4,
) -> GammaRegressor:
    """Fit GammaRegressor on per-claim raw severity with weight=n_claims."""
    y_per_claim = y_total / n_claims
    model = GammaRegressor(alpha=alpha, max_iter=2000, tol=1e-6, solver="lbfgs")
    model.fit(X, y_per_claim, sample_weight=n_claims)
    return model


def predict_severity_per_claim(model, X: np.ndarray) -> np.ndarray:
    """GammaRegressor.predict returns per-claim mean (clipped to positive)."""
    return np.clip(model.predict(X), 1e-3, None)


def compose_pure_premium(
    freq_rate: np.ndarray,
    exposure: np.ndarray,
    severity_per_claim: np.ndarray,
    p_has_sev: float = 1.0,
) -> np.ndarray:
    """Pure premium per policy = rate * p_has_sev * exposure * per-claim sev.

    p_has_sev corrects for claims without a sev record (9,116 in freMTPL2).
    Set p_has_sev=1.0 if the severity model was fit on all claims.
    """
    return freq_rate * p_has_sev * exposure * severity_per_claim
