"""Unit tests for src.models.actuarial (Phase 9)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.actuarial import (
    gini_normalized_predictions,
    lift_by_decile,
    lorenz_curve_predictions,
)


def test_lift_by_decile_shape() -> None:
    n = 100
    y_true = np.random.default_rng(42).uniform(0, 10, n)
    y_pred = y_true + np.random.default_rng(7).normal(0, 1, n)
    out = lift_by_decile(y_true, y_pred, n_deciles=5)
    assert len(out) == 5
    for row in out:
        assert "decile" in row
        assert "n" in row
        assert "ae_ratio" in row


def test_lift_by_decile_deciles_partition_population() -> None:
    n = 100
    y_true = np.ones(n)
    y_pred = np.arange(n, dtype=float)
    out = lift_by_decile(y_true, y_pred, n_deciles=10)
    total_n = sum(row["n"] for row in out)
    assert total_n == n


def test_lift_by_decile_rejects_too_few_rows() -> None:
    with pytest.raises(ValueError, match="at least"):
        lift_by_decile(np.ones(5), np.ones(5), n_deciles=10)


def test_gini_perfect_ranking() -> None:
    # Highest y_true = highest y_pred
    y_true = np.array([10.0, 5.0, 1.0, 0.0, 0.0])
    y_pred = np.array([9.0, 4.0, 2.0, 1.0, 0.0])
    gini = gini_normalized_predictions(y_true, y_pred)
    assert gini > 0.5  # strong positive ranking


def test_gini_random_ranking() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 10, 1000)
    y_pred = rng.uniform(0, 10, 1000)  # independent
    gini = gini_normalized_predictions(y_true, y_pred)
    assert abs(gini) < 0.1  # close to 0 for random


def test_gini_anti_ranking_is_negative() -> None:
    y_true = np.array([10.0, 5.0, 1.0, 0.0, 0.0])
    y_pred = np.array([0.0, 1.0, 2.0, 4.0, 9.0])  # reverse order
    gini = gini_normalized_predictions(y_true, y_pred)
    assert gini < 0.0


def test_lorenz_curve_ends_at_one() -> None:
    y_true = np.array([10.0, 5.0, 1.0, 0.0, 0.0])
    y_pred = np.array([9.0, 4.0, 2.0, 1.0, 0.0])
    pop, loss = lorenz_curve_predictions(y_true, y_pred)
    assert pop[0] == 0.0 and pop[-1] == 1.0
    assert loss[0] == 0.0
    assert abs(loss[-1] - 1.0) < 1e-9
