"""Stratified train/val/test split (Phase 4).

One assignment per IDpol, shared across frequency and severity models,
so no policy appears in two different splits. Stratified on ClaimNb > 0
(the frequency target's positive class) to keep the ~5% claim-positive
rate stable across train/val/test.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

STRATIFY_COL = "ClaimNb"


def assign_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a copy of df with a 'split' column (train/val/test).

    Stratifies on (ClaimNb > 0). Indices are disjoint and cover all rows.
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Sizes must sum to 1.0, got {total}")
    if STRATIFY_COL not in df.columns:
        raise KeyError(f"Missing stratify column: {STRATIFY_COL}")

    df = df.copy()
    strat = (df[STRATIFY_COL] > 0).astype("int8")

    trainval_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=strat, random_state=seed
    )

    val_ratio = val_size / (train_size + val_size)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_ratio,
        stratify=strat.loc[trainval_idx],
        random_state=seed,
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        int((df["split"] == "train").sum()),
        int((df["split"] == "val").sum()),
        int((df["split"] == "test").sum()),
    )
    return df
