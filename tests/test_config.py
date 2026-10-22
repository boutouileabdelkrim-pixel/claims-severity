"""Unit tests for src.utils.config.Settings (offline)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.utils.config import Settings


def test_default_settings_are_valid() -> None:
    """Default Settings must pass __post_init__ validation."""
    s = Settings()
    assert s.seed == 42
    assert 0 < s.train_size < 1
    assert s.n_splits >= 2
    assert s.mlflow_tracking_uri.startswith("file:")


def test_split_sizes_must_sum_to_one() -> None:
    """Invalid splits must raise ValueError at construction."""
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        Settings(train_size=0.5, val_size=0.2, test_size=0.2)


def test_settings_is_frozen() -> None:
    """Settings must be immutable (frozen dataclass)."""
    s = Settings()
    with pytest.raises(FrozenInstanceError):
        s.seed = 999  # type: ignore[misc]
