"""Unit tests for src.data.split (Phase 4)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import assign_split


def _make_df(n: int = 1000, claim_rate: float = 0.05) -> pd.DataFrame:
    """Synthetic frame with controlled claim-positive rate."""
    n_pos = int(n * claim_rate)
    return pd.DataFrame(
        {
            "IDpol": range(n),
            "ClaimNb": [1] * n_pos + [0] * (n - n_pos),
            "Exposure": [1.0] * n,
        }
    )


def test_split_sizes_sum_to_total() -> None:
    df = _make_df(1000)
    out = assign_split(df, 0.70, 0.15, 0.15, seed=42)
    assert len(out) == 1000
    assert set(out["split"].unique()) == {"train", "val", "test"}


def test_split_ratios_within_tolerance() -> None:
    df = _make_df(1000)
    out = assign_split(df, 0.70, 0.15, 0.15, seed=42)
    n_train = (out["split"] == "train").sum()
    n_val = (out["split"] == "val").sum()
    n_test = (out["split"] == "test").sum()
    assert 0.68 <= n_train / 1000 <= 0.72
    assert 0.13 <= n_val / 1000 <= 0.17
    assert 0.13 <= n_test / 1000 <= 0.17


def test_split_is_stratified() -> None:
    """Claim-positive rate should be preserved in every split."""
    df = _make_df(2000, claim_rate=0.05)
    out = assign_split(df, 0.70, 0.15, 0.15, seed=42)
    for split in ("train", "val", "test"):
        sub = out[out["split"] == split]
        rate = (sub["ClaimNb"] > 0).mean()
        assert 0.03 <= rate <= 0.07, (split, rate)


def test_split_is_reproducible() -> None:
    df = _make_df(500)
    a = assign_split(df, 0.70, 0.15, 0.15, seed=42)
    b = assign_split(df, 0.70, 0.15, 0.15, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_split_different_seeds_differ() -> None:
    df = _make_df(500)
    a = assign_split(df, 0.70, 0.15, 0.15, seed=42)
    b = assign_split(df, 0.70, 0.15, 0.15, seed=99)
    assert not a["split"].equals(b["split"])


def test_split_rejects_bad_sizes() -> None:
    df = _make_df(100)
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_split(df, 0.50, 0.20, 0.20, seed=42)
