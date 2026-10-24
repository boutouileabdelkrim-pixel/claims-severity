"""Unit tests for src.models.metrics (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.metrics import (
    actual_expected_ratio,
    gamma_deviance,
    mae,
    poisson_deviance,
    rmse,
)


def test_mae_zero_for_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mae_known_values() -> None:
    assert mae(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.5)


def test_rmse_zero_for_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_poisson_deviance_zero_for_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-9)


def test_poisson_deviance_positive_for_mismatch() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert poisson_deviance(y, np.array([2.0, 2.0, 2.0])) > 0.0


def test_gamma_deviance_zero_for_perfect() -> None:
    y = np.array([100.0, 500.0, 1000.0])
    assert gamma_deviance(y, y) == pytest.approx(0.0, abs=1e-6)


def test_actual_expected_ratio_unit_for_calibrated() -> None:
    y = np.array([10.0, 20.0, 30.0])
    assert actual_expected_ratio(y, y) == pytest.approx(1.0)
