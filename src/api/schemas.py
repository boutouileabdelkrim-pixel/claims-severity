"""Pydantic v2 schemas for the claims-severity API (Phase 11).

Input validation is strict: extra fields forbidden, ranges enforced.
This is deliberate — a permissive schema on a pricing API hides bugs
downstream. Contract requires Pydantic v2 with validation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PolicyInput(BaseModel):
    """Single policy input. All fields required, extra forbidden."""

    model_config = ConfigDict(extra="forbid")

    VehPower: int = Field(..., ge=1, le=20, description="Vehicle power")
    VehAge: int = Field(..., ge=0, le=100, description="Vehicle age in years")
    DrivAge: int = Field(..., ge=18, le=100, description="Driver age in years")
    BonusMalus: int = Field(..., ge=50, le=300, description="Bonus-malus coefficient")
    VehBrand: str = Field(..., min_length=1, max_length=20)
    VehGas: Literal["Regular", "Diesel"]
    Area: Literal["A", "B", "C", "D", "E", "F"]
    Density: int = Field(..., ge=0, le=50_000, description="Population density")
    Region: str = Field(..., min_length=1, max_length=10)
    Exposure: float = Field(..., gt=0, le=2.5, description="Exposure in years")


class PredictionOutput(BaseModel):
    """Single prediction with point estimate, claim probability, and interval."""

    pure_premium: float = Field(..., ge=0, description="Expected claim cost per policy (EUR)")
    claim_probability: float = Field(..., ge=0, le=1, description="P(at least one claim)")
    severity_median: float = Field(..., ge=0, description="Median per-claim cost (EUR)")
    severity_lo_80: float = Field(..., ge=0, description="80% lower bound per-claim (EUR)")
    severity_hi_80: float = Field(..., ge=0, description="80% upper bound per-claim (EUR)")
    interval_source: str = Field(
        default="CQR severity (Phase 8), q_hat=0.2714, conditional on claim"
    )


class BatchInput(BaseModel):
    """Batch input: 1 to 1000 policies."""

    model_config = ConfigDict(extra="forbid")
    policies: list[PolicyInput] = Field(..., min_length=1, max_length=1000)


class BatchOutput(BaseModel):
    """Batch output: list of predictions."""

    predictions: list[PredictionOutput]
    n: int


class InfoOutput(BaseModel):
    """Metadata about the loaded model."""

    version: str
    model: str
    features: int
    endpoints: list[str]
    bundle_loaded: bool
