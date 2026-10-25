"""Unit tests for src.models.uncertainty (Phase 8)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.uncertainty import (
    calibrate_cqr,
    cqr_adjustment,
    cqr_conformity_scores,
    empirical_coverage,
    fit_quantile_grid,
    fit_quantile_lgbm,
    interval_width,
    predict_interval,
    predict_quantiles,
)


def _xy(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    # Heavy-tailed target with zeros (like pure premium)
    y = np.where(rng.random(n) < 0.7, 0.0, rng.gamma(0.5, 3.0, size=n))
    return X, y


def test_fit_quantile_lgbm_returns_model() -> None:
    X, y = _xy(200)
    model = fit_quantile_lgbm(X, y, quantile=0.5, n_estimators=20)
    pred = model.predict(X)
    assert pred.shape == (200,)


def test_fit_quantile_lgbm_rejects_bad_quantile() -> None:
    X, y = _xy(50)
    with pytest.raises(ValueError, match="quantile must be"):
        fit_quantile_lgbm(X, y, quantile=1.5, n_estimators=5)


def test_fit_quantile_grid_returns_dict() -> None:
    X, y = _xy(200)
    models = fit_quantile_grid(X, y, quantiles=(0.1, 0.9), n_estimators=10)
    assert set(models.keys()) == {0.1, 0.9}
    preds = predict_quantiles(models, X)
    assert set(preds.keys()) == {0.1, 0.9}
    assert preds[0.1].shape == (200,)


def test_cqr_conformity_scores_formula() -> None:
    q_lo = np.array([1.0, 2.0, 3.0])
    q_hi = np.array([1.5, 2.5, 3.5])
    y = np.array([1.2, 3.0, 2.8])
    scores = cqr_conformity_scores(q_lo, q_hi, y)
    # row 0: max(1.0-1.2, 1.2-1.5) = max(-0.2, -0.3) = -0.2
    # row 1: max(2.0-3.0, 3.0-2.5) = max(-1.0, 0.5) = 0.5
    # row 2: max(3.0-2.8, 2.8-3.5) = max(0.2, -0.7) = 0.2
    np.testing.assert_allclose(scores, [-0.2, 0.5, 0.2])


def test_cqr_adjustment_finite_sample() -> None:
    scores = np.linspace(-1.0, 5.0, 100)  # 100 values
    # n=100, alpha=0.2 -> level = ceil(101*0.8)/100 = 81/100 = 0.81
    q_hat = cqr_adjustment(scores, alpha=0.2)
    expected = float(np.quantile(scores, 0.81))
    assert q_hat == pytest.approx(expected)


def test_cqr_adjustment_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        cqr_adjustment(np.array([1.0, 2.0]), alpha=1.5)


def test_empirical_coverage_perfect() -> None:
    y = np.array([0.5, 1.0, 1.5])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.0, 2.0])
    assert empirical_coverage(y, lo, hi) == 1.0


def test_empirical_coverage_partial() -> None:
    y = np.array([0.5, 1.0, 1.5, 5.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.0, 2.0, 2.0])
    assert empirical_coverage(y, lo, hi) == pytest.approx(0.75)


def test_interval_width_computation() -> None:
    lo = np.array([0.0, 1.0, 2.0])
    hi = np.array([1.0, 3.0, 5.0])
    mean_w, median_w = interval_width(lo, hi)
    assert mean_w == pytest.approx(2.0)
    assert median_w == pytest.approx(2.0)


def test_calibrate_cqr_returns_float() -> None:
    X, y = _xy(400)
    models = fit_quantile_grid(X, y, quantiles=(0.1, 0.9), n_estimators=10)
    q_hat = calibrate_cqr(models, X, y, alpha=0.2)
    assert isinstance(q_hat, float)


def test_predict_interval_ordering() -> None:
    X, y = _xy(200)
    models = fit_quantile_grid(X, y, quantiles=(0.1, 0.5, 0.9), n_estimators=10)
    lo, med, hi = predict_interval(models, X, q_hat=0.5)
    assert (lo <= hi).all()
    # Median is between lo and hi on average (may not hold per-row)
    assert lo.shape == (200,) and med.shape == (200,) and hi.shape == (200,)
