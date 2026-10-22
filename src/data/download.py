"""Download freMTPL2 datasets from OpenML (Phase 2, Step 2).

Pure acquisition: fetch freq (41214) and sev (41215), save raw parquet
to data/raw/. No transformation, no dtype casting, no aggregation.
Those belong to validate.py (Steps 3-4).

Fallback policy: if OpenML is unreachable, raise loudly with Kaggle
instructions. No silent retry, no silent fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from src.utils.paths import RAW_DIR

logger = logging.getLogger(__name__)

# OpenML dataset identifiers (verified in Step 1 recon)
FREQ_DATA_ID: int = 41214
SEV_DATA_ID: int = 41215
FREQ_NAME: str = "freMTPL2freq"
SEV_NAME: str = "freMTPL2sev"

# Output paths (anchored in data/raw/, gitignored via /data/)
FREQ_PARQUET: Path = RAW_DIR / "freq.parquet"
SEV_PARQUET: Path = RAW_DIR / "sev.parquet"


def _fetch(data_id: int, expected_name: str) -> pd.DataFrame:
    """Fetch one OpenML dataset by data_id and verify its name.

    Raises RuntimeError on network failure or name mismatch. No fallback.
    """
    try:
        ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch OpenML data_id={data_id} ({expected_name}): {e}. "
            "Check network connectivity. Manual Kaggle fallback: download the "
            "equivalent CSV files and place them under data/raw/ before "
            "rerunning validate."
        ) from e

    actual_name = ds.details.get("name")
    if actual_name != expected_name:
        raise RuntimeError(
            f"OpenML data_id={data_id} returned name={actual_name!r}, "
            f"expected {expected_name!r}. Dataset IDs may have changed upstream."
        )

    logger.info("Fetched %s: %d rows x %d cols", expected_name, *ds.frame.shape)
    return ds.frame


def fetch_freq() -> pd.DataFrame:
    """Return raw freMTPL2freq frame (678,013 x 12)."""
    return _fetch(FREQ_DATA_ID, FREQ_NAME)


def fetch_sev() -> pd.DataFrame:
    """Return raw freMTPL2sev frame (26,639 x 2)."""
    return _fetch(SEV_DATA_ID, SEV_NAME)


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write df as snappy-compressed parquet, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MB)", path, size_mb)
    return path


def download_all(force: bool = False) -> tuple[Path, Path]:
    """Download both datasets to data/raw/. Skip if present and not force."""
    if not force and FREQ_PARQUET.exists() and SEV_PARQUET.exists():
        logger.info("Both parquets already exist. Use force=True to re-download.")
        return FREQ_PARQUET, SEV_PARQUET

    freq = fetch_freq()
    sev = fetch_sev()

    save_parquet(freq, FREQ_PARQUET)
    save_parquet(sev, SEV_PARQUET)

    return FREQ_PARQUET, SEV_PARQUET


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    freq_path, sev_path = download_all()
    print(f"freq: {freq_path}")
    print(f"sev:  {sev_path}")


if __name__ == "__main__":
    main()
