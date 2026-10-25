"""API tests for src.api.main (Phase 11).

Uses FakeBundle so tests run without the real models/bundle_v1.pkl
(gitignored). Tests do NOT touch test.parquet — inputs are synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from src.api.main import app

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


class _FakeLGBM:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 0.10, dtype=float)


class _FakeGLM:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 2000.0, dtype=float)


class _FakeQuantile:
    def __init__(self, log_offset: float):
        self.log_offset = log_offset

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), np.log(1500.0) + self.log_offset, dtype=float)


def _fake_scaler() -> StandardScaler:
    s = StandardScaler()
    s.fit(np.zeros((2, len(NUMERICS))))
    return s


def _fake_bundle() -> dict:
    return {
        "lgbm_freq_model": _FakeLGBM(),
        "glm_sev_model": _FakeGLM(),
        "quantile_models": {
            0.1: _FakeQuantile(-0.3),
            0.5: _FakeQuantile(0.0),
            0.9: _FakeQuantile(0.3),
        },
        "p_has_sev": 0.7306,
        "q_hat": 0.2714,
        "scaler": _fake_scaler(),
        "dummy_cols": ["VehBrand_B1", "Region_R11"],
        "feature_names": NUMERICS + ["VehBrand_B1", "Region_R11"],
        "numerics": NUMERICS,
        "categoricals": CATEGORICALS,
        "version": "test",
    }


def _valid_policy() -> dict:
    return {
        "VehPower": 5,
        "VehAge": 2,
        "DrivAge": 45,
        "BonusMalus": 50,
        "VehBrand": "B12",
        "VehGas": "Regular",
        "Area": "D",
        "Density": 1217,
        "Region": "R82",
        "Exposure": 0.8,
    }


@pytest.fixture
def client_with_bundle():
    """TestClient with a fake bundle pre-installed."""
    with TestClient(app) as client:
        client.app.state.bundle = _fake_bundle()
        yield client


def test_health_ok() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_info_with_bundle(client_with_bundle) -> None:
    r = client_with_bundle.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["bundle_loaded"] is True
    assert body["model"].startswith("compound")
    assert body["features"] == len(NUMERICS) + 2
    assert "/predict" in body["endpoints"]


def test_predict_returns_predictions(client_with_bundle) -> None:
    r = client_with_bundle.post("/predict", json=_valid_policy())
    assert r.status_code == 200
    body = r.json()
    assert body["pure_premium"] > 0
    assert 0 <= body["claim_probability"] <= 1
    assert body["severity_lo_80"] <= body["severity_median"] <= body["severity_hi_80"]


def test_predict_rejects_invalid_bonus_malus(client_with_bundle) -> None:
    bad = _valid_policy()
    bad["BonusMalus"] = 10
    r = client_with_bundle.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_invalid_gas(client_with_bundle) -> None:
    bad = _valid_policy()
    bad["VehGas"] = "Electric"
    r = client_with_bundle.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_extra_fields(client_with_bundle) -> None:
    bad = _valid_policy()
    bad["unknown_field"] = "x"
    r = client_with_bundle.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_5_policies(client_with_bundle) -> None:
    payload = {"policies": [_valid_policy() for _ in range(5)]}
    r = client_with_bundle.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 5
    assert len(body["predictions"]) == 5


def test_batch_rejects_empty(client_with_bundle) -> None:
    r = client_with_bundle.post("/predict/batch", json={"policies": []})
    assert r.status_code == 422


def test_predict_503_when_bundle_missing() -> None:
    with TestClient(app) as client:
        client.app.state.bundle = None
        r = client.post("/predict", json=_valid_policy())
        assert r.status_code == 503
        assert "bundle" in r.json()["detail"].lower()
