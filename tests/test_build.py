"""Unit tests for src.features.build (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    CAPPED_COL,
    apply_cap,
    compute_cap_threshold,
    engineer_features,
)


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [1, 2, 3, 4, 5],
            "ClaimNb": [0, 1, 0, 2, 0],
            "ClaimNb_sev": [0, 1, 0, 1, 0],
            "Exposure": [0.5, 0.7, 0.3, 1.8, 0.6],
            "Area": ["A", "B", "C", "D", "E"],
            "VehPower": [5, 6, 7, 5, 6],
            "VehAge": [0, 2, 25, 1, 3],
            "DrivAge": [55, 22, 40, 30, 70],
            "BonusMalus": [50, 50, 120, 70, 55],
            "VehBrand": ["B12", "B12", "B1", "B2", "B3"],
            "VehGas": ["Regular", "Diesel", "Regular", "Diesel", "Regular"],
            "Density": [1217, 1217, 54, 100, 500],
            "Region": ["R82", "R82", "R22", "R11", "R31"],
            "ClaimAmount_sum": [np.nan, 500.0, np.nan, 1500.0, np.nan],
        }
    )


def test_engineer_features_adds_expected_columns() -> None:
    out = engineer_features(_make_df())
    expected = {
        "VehGas_diesel",
        "Area_ord",
        "BonusMalus_log",
        "Density_log",
        "VehAge_zero",
        "BonusMalus_gt100",
        "DrivAge_bucket",
        "VehAge_over_20",
    }
    assert expected.issubset(set(out.columns))


def test_engineer_features_vehgas_diesel() -> None:
    out = engineer_features(_make_df())
    assert out["VehGas_diesel"].tolist() == [0, 1, 0, 1, 0]


def test_engineer_features_area_ord() -> None:
    out = engineer_features(_make_df())
    assert out["Area_ord"].tolist() == [0, 1, 2, 3, 4]


def test_engineer_features_bonusmalus_log_matches_log() -> None:
    out = engineer_features(_make_df())
    expected = np.log([50, 50, 120, 70, 55])
    np.testing.assert_allclose(out["BonusMalus_log"], expected, rtol=1e-5)


def test_engineer_features_vehage_zero_and_over_20() -> None:
    out = engineer_features(_make_df())
    assert out["VehAge_zero"].tolist() == [1, 0, 0, 0, 0]
    assert out["VehAge_over_20"].tolist() == [0, 0, 1, 0, 0]


def test_engineer_features_drivage_bucket_range() -> None:
    out = engineer_features(_make_df())
    assert out["DrivAge_bucket"].between(0, 4).all()


def test_engineer_features_raises_on_missing_columns() -> None:
    df = _make_df().drop(columns=["VehGas"])
    with pytest.raises(KeyError, match="VehGas"):
        engineer_features(df)


def test_compute_cap_threshold_p995() -> None:
    s = pd.Series(np.arange(1, 1001, dtype=float))
    cap = compute_cap_threshold(s, quantile=0.995)
    assert cap == pytest.approx(995.0, rel=0.01)


def test_compute_cap_threshold_rejects_empty() -> None:
    with pytest.raises(ValueError, match="No severity values"):
        compute_cap_threshold(pd.Series([np.nan, np.nan]))


def test_apply_cap_respects_threshold_and_preserves_nan() -> None:
    df = _make_df()
    cap = 1000.0
    out = apply_cap(df, cap)
    assert CAPPED_COL in out.columns
    assert out.loc[1, CAPPED_COL] == 500.0
    assert out.loc[3, CAPPED_COL] == 1000.0
    assert pd.isna(out.loc[0, CAPPED_COL])
    assert pd.isna(out.loc[2, CAPPED_COL])
    assert pd.isna(out.loc[4, CAPPED_COL])


def test_apply_cap_raises_on_missing_target() -> None:
    df = _make_df().drop(columns=["ClaimAmount_sum"])
    with pytest.raises(KeyError, match="ClaimAmount_sum"):
        apply_cap(df, 1000.0)
