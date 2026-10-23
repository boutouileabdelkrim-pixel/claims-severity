"""Pandera schemas for freMTPL2 (Phase 2).

Five schemas cover the pipeline stages:
  - FREQ_RAW_SCHEMA / SEV_RAW_SCHEMA      : as delivered by OpenML
  - FREQ_CLEAN_SCHEMA / SEV_AGG_SCHEMA    : after cleaning + aggregation
  - MERGED_SCHEMA                         : after LEFT JOIN on IDpol

Design choices:
  - Uses pandera.pandas (top-level pandera import is deprecated in 0.34+).
  - coerce=True on raw schemas: tolerate dtype drift at the ingest
    boundary (OpenML returns category for some string columns).
  - coerce=False on clean/merged schemas: dtype drift becomes a bug.
  - Exposure is checked for > 0 only. It is NOT bounded by 1: freMTPL2
    Exposure is in years and multi-year contracts exist (real values up
    to 2.01 observed). Bounding it at 1 rejects valid data.
  - Multiple checks on one column are passed as a list: checks=[...].
    The `&` operator between Check objects is NOT supported in pandera
    0.34.1. Check.in_range's include_left kwarg was also removed.
  - VehGas raw values carry embedded single quotes ("'Regular'",
    "'Diesel'"). FREQ_RAW_SCHEMA checks them WITH quotes. clean_freq
    strips the quotes; FREQ_CLEAN_SCHEMA checks WITHOUT quotes.
  - ClaimAmount_sum in merged is nullable (policies with no sev record
    get NaN). ClaimNb_sev is filled with 0 so it stays non-nullable.
"""

from __future__ import annotations

import pandera.pandas as pa

# =========================================================================
# RAW — as delivered by OpenML (recon-confirmed shapes: 678k x 12, 26k x 2)
# =========================================================================

FREQ_RAW_SCHEMA = pa.DataFrameSchema(
    columns={
        "IDpol": pa.Column(float, pa.Check.ge(0), nullable=False),
        "ClaimNb": pa.Column(int, pa.Check.ge(0), nullable=False),
        "Exposure": pa.Column(float, pa.Check.gt(0), nullable=False),
        "Area": pa.Column(str, pa.Check.isin(list("ABCDEF")), nullable=False),
        "VehPower": pa.Column(int, pa.Check.ge(1), nullable=False),
        "VehAge": pa.Column(int, pa.Check.ge(0), nullable=False),
        "DrivAge": pa.Column(int, pa.Check.ge(15), nullable=False),
        "BonusMalus": pa.Column(int, pa.Check.ge(50), nullable=False),
        "VehBrand": pa.Column(str, nullable=False),
        "VehGas": pa.Column(str, pa.Check.isin(["'Regular'", "'Diesel'"]), nullable=False),
        "Density": pa.Column(int, pa.Check.ge(0), nullable=False),
        "Region": pa.Column(str, nullable=False),
    },
    strict=True,
    ordered=False,
    coerce=True,
)

SEV_RAW_SCHEMA = pa.DataFrameSchema(
    columns={
        "IDpol": pa.Column(int, pa.Check.ge(0), nullable=False),
        "ClaimAmount": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    strict=True,
    ordered=False,
    coerce=True,
)

# =========================================================================
# CLEAN — post-clean, pre-merge
# =========================================================================

FREQ_CLEAN_SCHEMA = pa.DataFrameSchema(
    columns={
        "IDpol": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "ClaimNb": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "Exposure": pa.Column("float64", pa.Check.gt(0), nullable=False),
        "Area": pa.Column(str, pa.Check.isin(list("ABCDEF")), nullable=False),
        "VehPower": pa.Column("int64", pa.Check.ge(1), nullable=False),
        "VehAge": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "DrivAge": pa.Column("int64", pa.Check.ge(15), nullable=False),
        "BonusMalus": pa.Column("int64", pa.Check.ge(50), nullable=False),
        "VehBrand": pa.Column(str, nullable=False),
        "VehGas": pa.Column(str, pa.Check.isin(["Regular", "Diesel"]), nullable=False),
        "Density": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "Region": pa.Column(str, nullable=False),
    },
    strict=True,
    ordered=False,
    coerce=False,
)

SEV_AGG_SCHEMA = pa.DataFrameSchema(
    columns={
        "IDpol": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "ClaimAmount_sum": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "ClaimNb_sev": pa.Column("int64", pa.Check.ge(1), nullable=False),
    },
    strict=True,
    ordered=False,
    coerce=False,
)

# =========================================================================
# MERGED — freq LEFT JOIN sev_agg on IDpol
# =========================================================================

MERGED_SCHEMA = pa.DataFrameSchema(
    columns={
        "IDpol": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "ClaimNb": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "Exposure": pa.Column("float64", pa.Check.gt(0), nullable=False),
        "Area": pa.Column(str, nullable=False),
        "VehPower": pa.Column("int64", pa.Check.ge(1), nullable=False),
        "VehAge": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "DrivAge": pa.Column("int64", pa.Check.ge(15), nullable=False),
        "BonusMalus": pa.Column("int64", pa.Check.ge(50), nullable=False),
        "VehBrand": pa.Column(str, nullable=False),
        "VehGas": pa.Column(str, nullable=False),
        "Density": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "Region": pa.Column(str, nullable=False),
        "ClaimAmount_sum": pa.Column("float64", nullable=True),
        "ClaimNb_sev": pa.Column("int64", pa.Check.ge(0), nullable=False),
    },
    strict=True,
    ordered=False,
    coerce=False,
)
