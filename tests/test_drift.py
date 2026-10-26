"""Unit tests for src.monitoring.drift (Phase 12)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import drift_report, ks_statistic, psi


def test_psi_zero_for_identical_samples() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10_000)
    assert psi(x, x) == 0.0


def test_psi_small_for_near_identical_samples() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10_000)
    y = rng.normal(0.05, 1, 10_000)  # tiny mean shift
    p = psi(x, y)
    assert p < 0.10, p


def test_psi_large_for_shifted_samples() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10_000)
    y = rng.normal(3, 1, 10_000)  # 3-sigma shift
    p = psi(x, y)
    assert p > 0.25, p


def test_psi_zero_for_constant_feature() -> None:
    x = np.ones(100)
    y = np.ones(100)
    assert psi(x, y) == 0.0


def test_ks_identical_samples() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10_000)
    stat, p = ks_statistic(x, x)
    assert stat == 0.0
    assert p >= 0.99


def test_ks_shifted_samples() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10_000)
    y = rng.normal(2, 1, 10_000)  # 2-sigma shift
    stat, p = ks_statistic(x, y)
    assert stat > 0.5, stat
    assert p < 0.001, p


def test_drift_report_shape_and_columns() -> None:
    rng = np.random.default_rng(42)
    n = 1000
    ref = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
            "c": rng.uniform(0, 10, n),
        }
    )
    cur = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
            "c": rng.uniform(0, 10, n),
        }
    )
    rep = drift_report(ref, cur, features=["a", "b", "c"])
    assert len(rep) == 3
    for col in ("feature", "psi", "ks_stat", "ks_pvalue", "verdict"):
        assert col in rep.columns
    assert (rep["verdict"] == "stable").all()


def test_drift_report_marks_significant_shift() -> None:
    rng = np.random.default_rng(42)
    n = 5000
    ref = pd.DataFrame({"x": rng.normal(0, 1, n)})
    cur = pd.DataFrame({"x": rng.normal(5, 1, n)})
    rep = drift_report(ref, cur, features=["x"])
    assert rep.iloc[0]["verdict"] == "significant"
