"""Distribution-shift detection (Phase 12).

Two classic metrics:
- PSI (Population Stability Index): binned comparison. Industry
  standard in credit/insurance for "has the population shifted?"
  Thresholds: <0.10 stable, 0.10-0.25 moderate, >0.25 significant.
- KS (Kolmogorov-Smirnov): continuous, non-parametric. Tests H0 = same
  distribution. Report both statistic (max CDF gap) and p-value.

Design choice: PSI bins are computed from the REFERENCE (train) data via
quantiles, then applied to the CURRENT (test) data. This is the correct
approach — using test quantiles would mask the shift we're trying to
detect.

Categorical features (Region with 22 values, VehBrand with 11) are
excluded from PSI. Their cardinality makes 10-bin quantization arbitrary,
and the drift would be dominated by rare categories. Categorical shift
is captured implicitly via tree-model feature importance drift (Phase 13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_STABLE: float = 0.10
PSI_MODERATE: float = 0.25


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two samples.

    Bins come from reference quantiles (10 uniform quantile bins).
    Constant features return 0.0 (no possible drift).
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    eps = 1e-4
    ref_pct = np.clip(ref_counts / len(ref), eps, None)
    cur_pct = np.clip(cur_counts / len(cur), eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0, 1.0
    stat, p = ks_2samp(ref, cur)
    return float(stat), float(p)


def _verdict(p: float) -> str:
    if p < PSI_STABLE:
        return "stable"
    if p < PSI_MODERATE:
        return "moderate"
    return "significant"


def drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-feature PSI + KS table with verdict.

    Skips columns missing from either frame (with a warning row).
    """
    rows: list[dict] = []
    for col in features:
        if col not in reference_df.columns or col not in current_df.columns:
            rows.append(
                {
                    "feature": col,
                    "psi": np.nan,
                    "ks_stat": np.nan,
                    "ks_pvalue": np.nan,
                    "verdict": "missing",
                }
            )
            continue
        ref = reference_df[col].to_numpy(dtype=float)
        cur = current_df[col].to_numpy(dtype=float)
        p = psi(ref, cur, n_bins=n_bins)
        ks_stat, ks_p = ks_statistic(ref, cur)
        rows.append(
            {
                "feature": col,
                "psi": p,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
                "verdict": _verdict(p),
            }
        )
    return pd.DataFrame(rows)
