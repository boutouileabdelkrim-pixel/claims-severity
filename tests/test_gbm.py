"""Unit tests for src.models.gbm (Phase 6).

Small synthetic fixtures. Real training tested in scripts/train_gbm.py.
"""

from __future__ import annotations

import numpy as np

from src.models.gbm import (
    fit_lgbm_freq,
    fit_lgbm_sev,
    fit_lgbm_tweedie,
    fit_xgb_freq,
    fit_xgb_sev,
    fit_xgb_tweedie,
    predict_freq_count,
    predict_sev_per_claim,
    predict_tweedie,
)


def _xy(n: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    exposure = rng.uniform(0.1, 1.5, size=n)
    # Claim rate increases with x0
    rate_true = np.exp(-2.0 + 0.3 * X[:, 0])
    claim_nb = rng.poisson(rate_true * exposure).astype(float)
    return X, claim_nb, exposure


def test_xgb_freq_predicts_non_negative_counts() -> None:
    X_tr, y_tr, e_tr = _xy(500)
    X_va, y_va, e_va = _xy(200, seed=99)
    model = fit_xgb_freq(X_tr, y_tr, e_tr, X_va, y_va, e_va, n_estimators=20)
    preds = predict_freq_count(model, X_va, e_va)
    assert (preds >= 0).all()
    assert preds.shape == (200,)


def test_lgbm_freq_predicts_non_negative_counts() -> None:
    X_tr, y_tr, e_tr = _xy(500)
    X_va, y_va, e_va = _xy(200, seed=99)
    model = fit_lgbm_freq(X_tr, y_tr, e_tr, X_va, y_va, e_va, n_estimators=20)
    preds = predict_freq_count(model, X_va, e_va)
    assert (preds >= 0).all()
    assert preds.shape == (200,)


def test_xgb_sev_predicts_positive_per_claim() -> None:
    rng = np.random.default_rng(42)
    n = 300
    X_tr = rng.normal(size=(n, 5))
    n_claims_tr = rng.integers(1, 4, size=n).astype(float)
    y_capped_tr = n_claims_tr * rng.uniform(100, 5000, size=n)
    X_va = rng.normal(size=(100, 5))
    n_claims_va = rng.integers(1, 4, size=100).astype(float)
    y_capped_va = n_claims_va * rng.uniform(100, 5000, size=100)

    model = fit_xgb_sev(
        X_tr,
        y_capped_tr,
        n_claims_tr,
        X_va,
        y_capped_va,
        n_claims_va,
        n_estimators=20,
    )
    preds = predict_sev_per_claim(model, X_va)
    assert (preds > 0).all()
    assert preds.shape == (100,)


def test_lgbm_sev_predicts_positive_per_claim() -> None:
    rng = np.random.default_rng(42)
    n = 300
    X_tr = rng.normal(size=(n, 5))
    n_claims_tr = rng.integers(1, 4, size=n).astype(float)
    y_capped_tr = n_claims_tr * rng.uniform(100, 5000, size=n)
    X_va = rng.normal(size=(100, 5))
    n_claims_va = rng.integers(1, 4, size=100).astype(float)
    y_capped_va = n_claims_va * rng.uniform(100, 5000, size=100)

    model = fit_lgbm_sev(
        X_tr,
        y_capped_tr,
        n_claims_tr,
        X_va,
        y_capped_va,
        n_claims_va,
        n_estimators=20,
    )
    preds = predict_sev_per_claim(model, X_va)
    assert (preds > 0).all()


def test_xgb_tweedie_predicts_non_negative_pp() -> None:
    rng = np.random.default_rng(42)
    n = 500
    X_tr = rng.normal(size=(n, 5))
    y_tr = rng.gamma(0.5, 2000, size=n) * (rng.random(n) < 0.3)
    X_va = rng.normal(size=(200, 5))
    y_va = rng.gamma(0.5, 2000, size=200) * (rng.random(200) < 0.3)
    model = fit_xgb_tweedie(X_tr, y_tr, X_va, y_va, n_estimators=20)
    preds = predict_tweedie(model, X_va)
    assert (preds >= 0).all()


def test_lgbm_tweedie_predicts_non_negative_pp() -> None:
    rng = np.random.default_rng(42)
    n = 500
    X_tr = rng.normal(size=(n, 5))
    y_tr = rng.gamma(0.5, 2000, size=n) * (rng.random(n) < 0.3)
    X_va = rng.normal(size=(200, 5))
    y_va = rng.gamma(0.5, 2000, size=200) * (rng.random(200) < 0.3)
    model = fit_lgbm_tweedie(X_tr, y_tr, X_va, y_va, n_estimators=20)
    preds = predict_tweedie(model, X_va)
    assert (preds >= 0).all()


def test_freq_predict_scales_with_exposure() -> None:
    """Doubling exposure should roughly double predicted count."""
    X_tr, y_tr, e_tr = _xy(500)
    X_va, y_va, e_va = _xy(200, seed=99)
    model = fit_lgbm_freq(X_tr, y_tr, e_tr, X_va, y_va, e_va, n_estimators=20)
    p1 = predict_freq_count(model, X_va, e_va)
    p2 = predict_freq_count(model, X_va, 2.0 * e_va)
    np.testing.assert_allclose(p2, 2.0 * p1, rtol=1e-6)


def test_sev_predict_shape() -> None:
    rng = np.random.default_rng(42)
    n = 300
    X_tr = rng.normal(size=(n, 5))
    n_claims_tr = np.ones(n)
    y_capped_tr = rng.uniform(100, 5000, size=n)
    X_va = rng.normal(size=(100, 5))
    n_claims_va = np.ones(100)
    y_capped_va = rng.uniform(100, 5000, size=100)
    model = fit_xgb_sev(
        X_tr,
        y_capped_tr,
        n_claims_tr,
        X_va,
        y_capped_va,
        n_claims_va,
        n_estimators=10,
    )
    assert predict_sev_per_claim(model, X_va).shape == (100,)
