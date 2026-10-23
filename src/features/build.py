"""Feature engineering for freMTPL2 (Phase 4).

Adds derived features from raw columns. NO target encoding, NO frequency
encoding of categoricals (both risk leaking the target). Trees will
handle categoricals natively (LightGBM) or via one-hot (sklearn GLM).

Also provides the cap logic: compute the p99.5 threshold on TRAIN
severity only, then apply as a constant to all splits. The raw
ClaimAmount_sum is preserved; ClaimAmount_capped is the model target
for Gamma-objective GBMs. Tweedie-direct models use the raw target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AREA_ORDER: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
DRIVAGE_BINS: list[int] = [15, 25, 35, 55, 75, 101]
CAPPED_COL: str = "ClaimAmount_capped"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 8 derived columns. Preserves all input columns."""
    required = {"VehGas", "Area", "BonusMalus", "Density", "VehAge", "DrivAge"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    df = df.copy()

    df["VehGas_diesel"] = (df["VehGas"] == "Diesel").astype("int8")
    df["Area_ord"] = df["Area"].map(AREA_ORDER).fillna(-1).astype("int8")
    df["BonusMalus_log"] = np.log(df["BonusMalus"]).astype("float32")
    df["Density_log"] = np.log1p(df["Density"]).astype("float32")
    df["VehAge_zero"] = (df["VehAge"] == 0).astype("int8")
    df["BonusMalus_gt100"] = (df["BonusMalus"] > 100).astype("int8")
    df["DrivAge_bucket"] = pd.cut(
        df["DrivAge"], bins=DRIVAGE_BINS, labels=False, include_lowest=True
    ).astype("int8")
    df["VehAge_over_20"] = (df["VehAge"] > 20).astype("int8")

    return df


def compute_cap_threshold(sev_train: pd.Series, quantile: float = 0.995) -> float:
    """p99.5 threshold for ClaimAmount_sum, computed on TRAIN severity only.

    Train severity = rows where split=='train' AND ClaimNb_sev > 0.
    Do NOT compute this on val/test or full data: that leaks the target.
    """
    values = sev_train.dropna()
    if len(values) == 0:
        raise ValueError("No severity values provided")
    return float(values.quantile(quantile))


def apply_cap(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add ClaimAmount_capped = min(ClaimAmount_sum, threshold). NaN preserved."""
    if "ClaimAmount_sum" not in df.columns:
        raise KeyError("Missing ClaimAmount_sum column")
    df = df.copy()
    df[CAPPED_COL] = df["ClaimAmount_sum"].clip(upper=threshold)
    return df
