"""Unit tests for src.models.tuning (Phase 7).

Tiny fixtures, small n_estimators to keep fast.
"""

from __future__ import annotations

import numpy as np
import optuna

from src.models.tuning import (
    build_lgbm_freq_params,
    cv_lgbm_freq,
    fit_final_lgbm_freq,
)


def _xy(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    exposure = rng.uniform(0.2, 1.5, size=n)
    rate_true = np.exp(-2.0 + 0.3 * X[:, 0])
    y = rng.poisson(rate_true * exposure).astype(float)
    return X, y, exposure


def test_build_lgbm_freq_params_returns_dict() -> None:
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    params = build_lgbm_freq_params(trial)
    assert isinstance(params, dict)
    assert params["objective"] == "poisson"
    assert "n_estimators" in params
    assert "learning_rate" in params
    assert "num_leaves" in params


def test_cv_lgbm_freq_returns_tuple() -> None:
    X, y, exposure = _xy(200)
    params = {
        "objective": "poisson",
        "n_estimators": 20,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": 1,
    }
    mean_dev, std_dev = cv_lgbm_freq(X, y, exposure, params, n_splits=2)
    assert isinstance(mean_dev, float)
    assert isinstance(std_dev, float)
    assert mean_dev >= 0.0
    assert std_dev >= 0.0


def test_fit_final_lgbm_freq_returns_model() -> None:
    X, y, exposure = _xy(200)
    params = {
        "objective": "poisson",
        "n_estimators": 10,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": 1,
    }
    model = fit_final_lgbm_freq(X, y, exposure, params)
    pred = model.predict(X)
    assert pred.shape == (200,)
    assert (pred >= 0).all()
