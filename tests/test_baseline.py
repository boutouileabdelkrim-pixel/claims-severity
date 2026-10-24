"""Unit tests for src.models.baseline (Phase 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import (
    DummyFrequency,
    DummySeverity,
    compose_pure_premium,
    compute_p_has_sev,
    fit_gamma_sev,
    fit_poisson_freq,
    predict_freq_count,
    predict_severity_per_claim,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Exposure": [1.0, 1.0, 0.5, 1.0, 1.0],
            "ClaimNb": [0, 1, 0, 2, 0],
            "ClaimNb_sev": [0, 1, 0, 1, 0],
            "ClaimAmount_sum": [np.nan, 500.0, np.nan, 1000.0, np.nan],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_compute_p_has_sev() -> None:
    df = pd.DataFrame(
        {
            "ClaimNb": [1, 1, 2, 0, 0],
            "ClaimNb_sev": [1, 0, 1, 0, 0],
        }
    )
    # sum(ClaimNb_sev) = 2, sum(ClaimNb) = 4 -> 0.5
    assert compute_p_has_sev(df) == pytest.approx(0.5)


def test_dummy_frequency_rate_is_total_claims_over_total_exposure() -> None:
    df = _df()
    model = DummyFrequency().fit(df)
    expected = (0 + 1 + 0 + 2 + 0) / (1 + 1 + 0.5 + 1 + 1)
    assert model.rate_ == expected


def test_dummy_frequency_predict_scales_with_exposure() -> None:
    df = _df()
    model = DummyFrequency().fit(df)
    preds = model.predict(df)
    np.testing.assert_allclose(preds, model.rate_ * df["Exposure"].to_numpy())


def test_dummy_severity_mean_uses_per_claim_weighting() -> None:
    df = _df()
    model = DummySeverity().fit(df)
    # Only rows 1 and 3 have claims. Per-claim: 500 and 1000. Equal weight.
    assert model.mean_ == 750.0


def test_dummy_severity_predict_is_constant() -> None:
    df = _df()
    model = DummySeverity().fit(df)
    preds = model.predict(df)
    assert len(set(preds)) == 1


def test_fit_poisson_freq_shapes() -> None:
    df = _df()
    X = df[["x1"]].to_numpy(dtype=float)
    y = df["ClaimNb"].to_numpy(dtype=float)
    exposure = df["Exposure"].to_numpy(dtype=float)
    model = fit_poisson_freq(X, y, exposure)
    pred_count = predict_freq_count(model, X, exposure)
    assert pred_count.shape == (5,)


def test_fit_poisson_freq_predicts_non_negative() -> None:
    df = _df()
    X = df[["x1"]].to_numpy(dtype=float)
    y = df["ClaimNb"].to_numpy(dtype=float)
    exposure = df["Exposure"].to_numpy(dtype=float)
    model = fit_poisson_freq(X, y, exposure)
    assert (predict_freq_count(model, X, exposure) >= 0).all()


def test_fit_gamma_sev_shapes() -> None:
    df = _df()
    mask = df["ClaimNb_sev"] > 0
    sub = df.loc[mask].reset_index(drop=True)
    X = sub[["x1"]].to_numpy(dtype=float)
    y_total = sub["ClaimAmount_sum"].to_numpy(dtype=float)
    n_claims = sub["ClaimNb_sev"].to_numpy(dtype=float)
    model = fit_gamma_sev(X, y_total, n_claims)
    assert model.predict(X).shape == (2,)


def test_fit_gamma_sev_predicts_positive() -> None:
    df = _df()
    mask = df["ClaimNb_sev"] > 0
    sub = df.loc[mask].reset_index(drop=True)
    X = sub[["x1"]].to_numpy(dtype=float)
    y_total = sub["ClaimAmount_sum"].to_numpy(dtype=float)
    n_claims = sub["ClaimNb_sev"].to_numpy(dtype=float)
    model = fit_gamma_sev(X, y_total, n_claims)
    preds = predict_severity_per_claim(model, X)
    assert (preds > 0).all()


def test_compose_pure_premium_is_product() -> None:
    out = compose_pure_premium(
        np.array([0.05, 0.10]),
        np.array([1.0, 0.5]),
        np.array([1000.0, 2000.0]),
    )
    np.testing.assert_allclose(out, np.array([50.0, 100.0]))


def test_compose_pure_premium_with_p_has_sev() -> None:
    out = compose_pure_premium(
        np.array([0.10]),
        np.array([1.0]),
        np.array([1000.0]),
        p_has_sev=0.75,
    )
    np.testing.assert_allclose(out, np.array([75.0]))


def test_compose_pure_premium_handles_zero_rate() -> None:
    out = compose_pure_premium(
        np.array([0.0, 0.10]),
        np.array([1.0, 1.0]),
        np.array([1000.0, 1000.0]),
    )
    assert out[0] == 0.0
    assert out[1] == pytest.approx(100.0)
