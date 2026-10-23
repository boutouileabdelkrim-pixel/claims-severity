"""Build the features parquets (Phase 4).

Pipeline:
    data/processed/merged.parquet
      -> assign_split (stratified on ClaimNb > 0)
      -> engineer_features (8 derived columns)
      -> cap threshold (p99.5 on TRAIN severity only)
      -> apply_cap
      -> data/features/{train,val,test}.parquet
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.split import assign_split
from src.features.build import apply_cap, compute_cap_threshold, engineer_features
from src.utils.config import SETTINGS
from src.utils.paths import FEATURES_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

MERGED = PROCESSED_DIR / "merged.parquet"
CAP_QUANTILE = 0.995


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_parquet(MERGED)
    logger.info("Loaded merged: %s", df.shape)

    df = assign_split(
        df,
        train_size=SETTINGS.train_size,
        val_size=SETTINGS.val_size,
        test_size=SETTINGS.test_size,
        seed=SETTINGS.seed,
    )
    df = engineer_features(df)

    sev_train = df.loc[(df["split"] == "train") & (df["ClaimNb_sev"] > 0), "ClaimAmount_sum"]
    cap = compute_cap_threshold(sev_train, quantile=CAP_QUANTILE)
    logger.info("Cap threshold (p99.5 of train severity): %.2f EUR", cap)

    df = apply_cap(df, cap)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        sub = df[df["split"] == split_name].drop(columns=["split"]).reset_index(drop=True)
        path = FEATURES_DIR / f"{split_name}.parquet"
        sub.to_parquet(path, index=False, compression="snappy")
        logger.info("Wrote %s: %s", path, sub.shape)

    print()
    print("=== Feature build summary ===")
    print(f"Cap threshold:           {cap:,.2f} EUR")
    for split_name in ("train", "val", "test"):
        sub = df[df["split"] == split_name]
        n_freq = int((sub["ClaimNb"] > 0).sum())
        n_sev = int((sub["ClaimNb_sev"] > 0).sum())
        print(f"{split_name:6s}: rows={len(sub):>7,}  freq_pos={n_freq:>6,}  sev_pos={n_sev:>6,}")


if __name__ == "__main__":
    main()
