"""Unit tests for src.data.validate (Phase 2).

6 offline tests with synthetic fixtures. No file I/O, no network.
"""

from __future__ import annotations

import pandas as pd

from src.data.validate import (
    clean_freq,
    clean_sev,
    merge_freq_sev,
    validate_clean_freq,
    validate_clean_sev,
    validate_merged,
)


def _freq_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [1.0, 3.0, 5.0],
            "ClaimNb": [0, 1, 0],
            "Exposure": [0.5, 0.7, 0.3],
            "Area": ["A", "B", "C"],
            "VehPower": [5, 6, 7],
            "VehAge": [0, 2, 5],
            "DrivAge": [55, 52, 40],
            "BonusMalus": [50, 50, 60],
            "VehBrand": ["B12", "B12", "B1"],
            "VehGas": ["'Regular'", "'Diesel'", "'Regular'"],
            "Density": [1217, 1217, 54],
            "Region": ["R82", "R82", "R22"],
        }
    )


def _sev_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [1, 1, 3],
            "ClaimAmount": [500.0, 495.2, 1128.12],
        }
    )


def test_clean_freq_strips_vehgas_quotes() -> None:
    out = clean_freq(_freq_raw())
    assert out["VehGas"].tolist() == ["Regular", "Diesel", "Regular"]


def test_clean_freq_casts_idpol_to_int64() -> None:
    out = clean_freq(_freq_raw())
    assert str(out["IDpol"].dtype) == "int64"


def test_clean_sev_aggregates_per_idpol() -> None:
    out = clean_sev(_sev_raw())
    assert len(out) == 2
    row1 = out.loc[out["IDpol"] == 1].iloc[0]
    assert row1["ClaimAmount_sum"] == 995.2
    assert row1["ClaimNb_sev"] == 2


def test_merge_freq_sev_left_join_fills_zeros() -> None:
    freq = clean_freq(_freq_raw())
    sev_agg = clean_sev(_sev_raw())
    merged = merge_freq_sev(freq, sev_agg)
    assert len(merged) == 3
    row5 = merged.loc[merged["IDpol"] == 5].iloc[0]
    assert row5["ClaimNb_sev"] == 0
    assert pd.isna(row5["ClaimAmount_sum"])


def test_clean_outputs_pass_schemas() -> None:
    freq = clean_freq(_freq_raw())
    sev_agg = clean_sev(_sev_raw())
    validate_clean_freq(freq)
    validate_clean_sev(sev_agg)


def test_merged_passes_schema() -> None:
    freq = clean_freq(_freq_raw())
    sev_agg = clean_sev(_sev_raw())
    merged = merge_freq_sev(freq, sev_agg)
    validate_merged(merged)
