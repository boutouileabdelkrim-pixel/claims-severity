"""Clean, aggregate, merge freMTPL2 (Phase 2).

Pipeline:
    data/raw/{freq,sev}.parquet
      -> validate_raw (schemas)
      -> clean_freq (cast IDpol, strip VehGas quotes)
      -> clean_sev (aggregate per IDpol: sum amount, count claims)
      -> validate_clean (schemas)
      -> merge_freq_sev (LEFT JOIN on IDpol, fillna 0 on ClaimNb_sev)
      -> validate_merged (schema)
      -> data/processed/{freq,sev_agg,merged}.parquet

Critical: aggregating sev per IDpol BEFORE the join. Without it, the
join explodes to 18B rows (see reports/phase2_recon.txt).

Policies with ClaimNb > 0 but no sev record (9,110 in freMTPL2) get
ClaimNb_sev=0 and ClaimAmount_sum=NaN. This is a documented dataset
quirk, not a bug. Phase 3 decides the modeling treatment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.download import FREQ_PARQUET, SEV_PARQUET, download_all
from src.data.schemas import (
    FREQ_CLEAN_SCHEMA,
    FREQ_RAW_SCHEMA,
    MERGED_SCHEMA,
    SEV_AGG_SCHEMA,
    SEV_RAW_SCHEMA,
)
from src.utils.paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

FREQ_PROCESSED: Path = PROCESSED_DIR / "freq.parquet"
SEV_AGG_PROCESSED: Path = PROCESSED_DIR / "sev_agg.parquet"
MERGED_PROCESSED: Path = PROCESSED_DIR / "merged.parquet"


def validate_raw_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw freq against FREQ_RAW_SCHEMA."""
    return FREQ_RAW_SCHEMA.validate(df)


def validate_raw_sev(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw sev against SEV_RAW_SCHEMA."""
    return SEV_RAW_SCHEMA.validate(df)


def clean_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Cast IDpol to int64, strip embedded quotes from VehGas."""
    df = df.copy()
    df["IDpol"] = df["IDpol"].astype("int64")
    df["VehGas"] = df["VehGas"].str.strip("'").astype(str)
    return df


def clean_sev(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sev per IDpol: total claim amount, number of claims.

    Output: one row per unique IDpol, columns IDpol, ClaimAmount_sum,
    ClaimNb_sev. This is the critical step that prevents the join
    explosion (18B rows naive vs 678k target).
    """
    agg = df.groupby("IDpol", as_index=False).agg(
        ClaimAmount_sum=("ClaimAmount", "sum"),
        ClaimNb_sev=("ClaimAmount", "count"),
    )
    agg = agg.astype({"ClaimNb_sev": "int64"})
    logger.info("Aggregated sev: %d rows -> %d unique IDpol", len(df), len(agg))
    return agg


def validate_clean_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Validate cleaned freq against FREQ_CLEAN_SCHEMA."""
    return FREQ_CLEAN_SCHEMA.validate(df)


def validate_clean_sev(df: pd.DataFrame) -> pd.DataFrame:
    """Validate aggregated sev against SEV_AGG_SCHEMA."""
    return SEV_AGG_SCHEMA.validate(df)


def merge_freq_sev(freq: pd.DataFrame, sev_agg: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN freq with aggregated sev on IDpol.

    Policies in freq but not in sev get ClaimAmount_sum=NaN, ClaimNb_sev=0.
    """
    merged = freq.merge(sev_agg, on="IDpol", how="left")
    merged["ClaimNb_sev"] = merged["ClaimNb_sev"].fillna(0).astype("int64")

    missing = set(sev_agg["IDpol"]) - set(freq["IDpol"])
    if missing:
        logger.warning(
            "%d sev IDpol(s) absent from freq. First 5: %s",
            len(missing),
            sorted(missing)[:5],
        )

    return merged


def validate_merged(df: pd.DataFrame) -> pd.DataFrame:
    """Validate merged frame against MERGED_SCHEMA."""
    return MERGED_SCHEMA.validate(df)


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write df as snappy-compressed parquet, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MB)", path, size_mb)
    return path


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end: ensure raw -> validate -> clean -> merge -> save."""
    download_all()

    freq_raw = pd.read_parquet(FREQ_PARQUET)
    sev_raw = pd.read_parquet(SEV_PARQUET)
    logger.info("Loaded raw: freq %s, sev %s", freq_raw.shape, sev_raw.shape)

    freq = clean_freq(validate_raw_freq(freq_raw))
    sev_agg = clean_sev(validate_raw_sev(sev_raw))

    validate_clean_freq(freq)
    validate_clean_sev(sev_agg)

    merged = merge_freq_sev(freq, sev_agg)
    validate_merged(merged)

    save_parquet(freq, FREQ_PROCESSED)
    save_parquet(sev_agg, SEV_AGG_PROCESSED)
    save_parquet(merged, MERGED_PROCESSED)

    return freq, sev_agg, merged


def summary(freq: pd.DataFrame, sev_agg: pd.DataFrame, merged: pd.DataFrame) -> None:
    """Print a compact pipeline summary."""
    print("=== Pipeline summary ===")
    print(f"freq clean:     {freq.shape}   IDpol: {freq['IDpol'].dtype}")
    print(f"sev aggregated: {sev_agg.shape}   IDpol: {sev_agg['IDpol'].dtype}")
    print(f"merged:         {merged.shape}")
    print()
    n_with_sev = int(merged["ClaimNb_sev"].gt(0).sum())
    n_claims_no_sev = int(((merged["ClaimNb"] > 0) & (merged["ClaimNb_sev"] == 0)).sum())
    print(f"Policies with sev:       {n_with_sev:,}  ({n_with_sev / len(merged) * 100:.2f}%)")
    print(f"Claim reported, no sev:  {n_claims_no_sev:,}")
    print(f"ClaimAmount_sum total:   {merged['ClaimAmount_sum'].sum():,.2f}")
    print(f"Exposure total:          {merged['Exposure'].sum():,.2f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    freq, sev_agg, merged = run_pipeline()
    summary(freq, sev_agg, merged)
    print()
    print(f"freq:   {FREQ_PROCESSED}")
    print(f"sev:    {SEV_AGG_PROCESSED}")
    print(f"merged: {MERGED_PROCESSED}")


if __name__ == "__main__":
    main()
