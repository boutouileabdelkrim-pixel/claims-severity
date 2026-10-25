"""Fairness analysis: coverage and residuals by group (Phase 9).

Group-level diagnosis:
- residuals_by_group: mean/median residual, AE, count per group
- coverage_by_group: fraction of y_true in [lo, hi] per group

Groups checked: DrivAge_bucket (0-4 ordinal), Region (top-K), Area (A-F).
Descriptive, not a normative fairness verdict. Report only.

Silences pandas FutureWarning: we drop the group column from the
aggregated DataFrame BEFORE groupby, and pass the grouping Series
explicitly. This avoids the "apply operated on grouping columns"
warning without relying on include_groups (which has version-specific
semantics).
"""

from __future__ import annotations

import pandas as pd


def residuals_by_group(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Mean/median residual, AE, count per group.

    residual = y_true - y_pred. Positive = model under-predicts (actual
    higher than predicted).
    """
    sub = df[[y_true_col, y_pred_col, group_col]].copy()
    if top_k is not None:
        top = sub[group_col].value_counts().nlargest(top_k).index
        sub = sub[sub[group_col].isin(top)]

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n": len(g),
                "mean_true": float(g[y_true_col].mean()),
                "mean_pred": float(g[y_pred_col].mean()),
                "mean_residual": float((g[y_true_col] - g[y_pred_col]).mean()),
                "median_residual": float((g[y_true_col] - g[y_pred_col]).median()),
                "ae_ratio": float(g[y_true_col].sum() / g[y_pred_col].sum())
                if g[y_pred_col].sum() > 0
                else 0.0,
            }
        )

    return (
        sub.drop(columns=[group_col])
        .groupby(sub[group_col], observed=True)
        .apply(_agg)
        .reset_index()
    )


def coverage_by_group(
    df: pd.DataFrame,
    y_true_col: str,
    lo_col: str,
    hi_col: str,
    group_col: str,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Fraction of y_true in [lo, hi] per group, plus mean width."""
    sub = df[[y_true_col, lo_col, hi_col, group_col]].copy()
    if top_k is not None:
        top = sub[group_col].value_counts().nlargest(top_k).index
        sub = sub[sub[group_col].isin(top)]

    def _agg(g: pd.DataFrame) -> pd.Series:
        in_interval = (g[y_true_col] >= g[lo_col]) & (g[y_true_col] <= g[hi_col])
        width = g[hi_col] - g[lo_col]
        return pd.Series(
            {
                "n": len(g),
                "coverage": float(in_interval.mean()),
                "mean_width": float(width.mean()),
            }
        )

    return (
        sub.drop(columns=[group_col])
        .groupby(sub[group_col], observed=True)
        .apply(_agg)
        .reset_index()
    )
