"""Actuarial EDA helpers (Phase 3).

Pure functions for distribution analysis of a heavy-tailed target:
Lorenz curve, Gini coefficient, quantile summary, concentration metrics.

Used in scripts/eda_report.py (console report) and intended to be reused
in Phase 6 (model evaluation via lift charts) and Phase 9 (actuarial
metrics: actual/expected by decile, Lorenz curve of predictions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lorenz_curve(
    values: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz curve of a non-negative 1D distribution.

    Returns (pop_share, value_share), each of length n+1, starting at
    (0, 0) and ending at (1, 1). Values are sorted ascending internally.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    if (x < 0).any():
        raise ValueError("Lorenz curve requires non-negative values")

    x = np.sort(x)
    total = x.sum()
    if total == 0:
        return np.linspace(0, 1, len(x) + 1), np.linspace(0, 1, len(x) + 1)

    cum = np.cumsum(x) / total
    pop_share = np.linspace(0, 1, len(x) + 1)
    value_share = np.concatenate([[0.0], cum])
    return pop_share, value_share


def gini_normalized(values: pd.Series | np.ndarray) -> float:
    """Gini coefficient in [0, 1]. 0 = perfectly equal, 1 = all in one value.

    For freMTPL2 severity, high values (0.7+) are expected (heavy tail).
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    if (x < 0).any():
        raise ValueError("Gini requires non-negative values")

    x = np.sort(x)
    n = len(x)
    gini = (2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x))) - (n + 1) / n
    return float(max(gini, 0.0))


def quantile_summary(
    values: pd.Series | np.ndarray,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999),
) -> dict:
    """Return count, mean, std, min, max, and the requested quantiles."""
    x = pd.Series(values, dtype=float).dropna()
    out = {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }
    for q in quantiles:
        out[f"p{q * 100:g}"] = float(x.quantile(q))
    return out


def concentration_stats(
    values: pd.Series | np.ndarray, top_fracs: tuple[float, ...] = (0.10, 0.01, 0.001)
) -> dict:
    """Share of total value held by the top X% of policies.

    Example: top 1% of claims may hold 30% of total severity. Used to
    justify the outlier policy decision.
    """
    x = pd.Series(values, dtype=float).dropna()
    x = x[x > 0]
    if len(x) == 0:
        return {f"top_{f}": 0.0 for f in top_fracs}
    total = x.sum()
    out = {}
    for frac in top_fracs:
        n_top = max(1, int(len(x) * frac))
        top_sum = x.nlargest(n_top).sum()
        out[f"top_{frac}"] = float(top_sum / total)
    return out


def exposure_weighted_frequency(df: pd.DataFrame) -> float:
    """Total claims / total exposure. The baseline frequency rate."""
    if "ClaimNb" not in df.columns or "Exposure" not in df.columns:
        raise KeyError("Need ClaimNb and Exposure columns")
    total_exposure = df["Exposure"].sum()
    if total_exposure == 0:
        return 0.0
    return float(df["ClaimNb"].sum() / total_exposure)
