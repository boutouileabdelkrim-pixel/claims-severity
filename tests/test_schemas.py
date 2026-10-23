"""Unit tests for src.data.schemas (Phase 2).

8 offline tests with synthetic fixtures. No file I/O, no network.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.data.schemas import (
    FREQ_CLEAN_SCHEMA,
    FREQ_RAW_SCHEMA,
    MERGED_SCHEMA,
    SEV_AGG_SCHEMA,
    SEV_RAW_SCHEMA,
)


def _make_freq_raw() -> pd.DataFrame:
    """Minimal valid raw freq, matches OpenML dtypes (synthetic IDs)."""
    return pd.DataFrame(
        {
            "IDpol": [1.0, 3.0, 5.0, 7.0, 9.0],
            "ClaimNb": [0, 1, 0, 2, 0],
            "Exposure": [0.5, 0.7, 0.3, 1.8, 0.6],
            "Area": ["A", "B", "C", "D", "E"],
            "VehPower": [5, 6, 7, 5, 6],
            "VehAge": [0, 2, 5, 1, 3],
            "DrivAge": [55, 52, 40, 30, 60],
            "BonusMalus": [50, 50, 60, 70, 55],
            "VehBrand": ["B12", "B12", "B1", "B2", "B3"],
            "VehGas": ["'Regular'", "'Diesel'", "'Regular'", "'Diesel'", "'Regular'"],
            "Density": [1217, 1217, 54, 100, 500],
            "Region": ["R82", "R82", "R22", "R11", "R31"],
        }
    )


def _make_freq_clean() -> pd.DataFrame:
    """Clean version (IDpol int64, VehGas unquoted)."""
    df = _make_freq_raw().copy()
    df["IDpol"] = df["IDpol"].astype("int64")
    df["VehGas"] = df["VehGas"].str.strip("'")
    return df


def _make_sev_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [1552, 1010996, 4024277],
            "ClaimAmount": [995.20, 1128.12, 1851.11],
        }
    )


def test_freq_raw_schema_accepts_valid() -> None:
    validated = FREQ_RAW_SCHEMA.validate(_make_freq_raw())
    assert len(validated) == 5


def test_freq_raw_schema_accepts_exposure_above_one() -> None:
    """Exposure = 2.01 is valid (multi-year contracts exist in freMTPL2)."""
    df = _make_freq_raw()
    df.loc[0, "Exposure"] = 2.01
    FREQ_RAW_SCHEMA.validate(df)


def test_freq_raw_schema_rejects_negative_exposure() -> None:
    df = _make_freq_raw()
    df.loc[0, "Exposure"] = -0.1
    with pytest.raises(SchemaError):
        FREQ_RAW_SCHEMA.validate(df)


def test_freq_raw_schema_rejects_invalid_area() -> None:
    df = _make_freq_raw()
    df.loc[0, "Area"] = "Z"
    with pytest.raises(SchemaError):
        FREQ_RAW_SCHEMA.validate(df)


def test_sev_raw_schema_rejects_negative_amount() -> None:
    df = _make_sev_raw()
    df.loc[0, "ClaimAmount"] = -10.0
    with pytest.raises(SchemaError):
        SEV_RAW_SCHEMA.validate(df)


def test_sev_agg_schema_accepts_valid() -> None:
    df = pd.DataFrame(
        {
            "IDpol": [1552, 1010996],
            "ClaimAmount_sum": [995.20, 1128.12],
            "ClaimNb_sev": [1, 1],
        }
    )
    validated = SEV_AGG_SCHEMA.validate(df)
    assert len(validated) == 2


def test_freq_clean_schema_requires_int_idpol() -> None:
    df = _make_freq_clean()
    df["IDpol"] = df["IDpol"].astype("float64")
    with pytest.raises(SchemaError):
        FREQ_CLEAN_SCHEMA.validate(df)


def test_merged_schema_accepts_valid_with_nan_severity() -> None:
    df = _make_freq_clean()
    df["ClaimAmount_sum"] = [995.20, None, None, 1128.12, None]
    df["ClaimNb_sev"] = [1, 0, 0, 1, 0]
    validated = MERGED_SCHEMA.validate(df)
    assert len(validated) == 5
