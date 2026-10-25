"""Unit tests for src.models.fairness (Phase 9)."""

from __future__ import annotations

import pandas as pd

from src.models.fairness import coverage_by_group, residuals_by_group


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": [10.0, 20.0, 30.0, 40.0, 50.0],
            "y_pred": [8.0, 22.0, 28.0, 42.0, 48.0],
            "lo": [5.0, 15.0, 25.0, 35.0, 45.0],
            "hi": [15.0, 25.0, 35.0, 45.0, 55.0],
            "group": ["A", "A", "B", "B", "C"],
        }
    )


def test_residuals_by_group_columns() -> None:
    out = residuals_by_group(_df(), "y_true", "y_pred", "group")
    for col in (
        "group",
        "n",
        "mean_true",
        "mean_pred",
        "mean_residual",
        "median_residual",
        "ae_ratio",
    ):
        assert col in out.columns


def test_residuals_by_group_values() -> None:
    out = residuals_by_group(_df(), "y_true", "y_pred", "group")
    row_a = out[out["group"] == "A"].iloc[0]
    assert row_a["n"] == 2
    assert abs(row_a["mean_residual"] - 0.0) < 1e-9


def test_residuals_top_k_limits_groups() -> None:
    out = residuals_by_group(_df(), "y_true", "y_pred", "group", top_k=1)
    assert len(out) == 1


def test_coverage_by_group_columns() -> None:
    out = coverage_by_group(_df(), "y_true", "lo", "hi", "group")
    for col in ("group", "n", "coverage", "mean_width"):
        assert col in out.columns


def test_coverage_by_group_perfect() -> None:
    out = coverage_by_group(_df(), "y_true", "lo", "hi", "group")
    # All y_true in [lo, hi] by construction
    assert (out["coverage"] == 1.0).all()


def test_coverage_by_group_partial() -> None:
    df = _df()
    df.loc[0, "hi"] = 5.0  # now y_true=10 is outside [5,5]
    out = coverage_by_group(df, "y_true", "lo", "hi", "group")
    row_a = out[out["group"] == "A"].iloc[0]
    # A has 2 rows, 1 in, 1 out => coverage 0.5
    assert abs(row_a["coverage"] - 0.5) < 1e-9
