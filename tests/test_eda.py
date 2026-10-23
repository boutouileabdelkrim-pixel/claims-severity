"""Unit tests for src.features.eda (Phase 3).

9 offline tests with synthetic fixtures. No file I/O, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.eda import (
    concentration_stats,
    exposure_weighted_frequency,
    gini_normalized,
    lorenz_curve,
    quantile_summary,
)


def test_lorenz_curve_starts_and_ends_at_corners() -> None:
    pop, val = lorenz_curve([1, 2, 3, 4, 5])
    assert pop[0] == 0.0 and pop[-1] == 1.0
    assert val[0] == 0.0 and val[-1] == 1.0


def test_lorenz_curve_is_monotonic() -> None:
    pop, val = lorenz_curve([1, 2, 3, 4, 5])
    assert np.all(np.diff(pop) >= 0)
    assert np.all(np.diff(val) >= 0)


def test_lorenz_curve_rejects_negative() -> None:
    with pytest.raises(ValueError):
        lorenz_curve([-1, 0, 1])


def test_gini_zero_for_uniform() -> None:
    assert gini_normalized([1, 1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-9)


def test_gini_max_for_one_value() -> None:
    # For n=5, max Gini = (n-1)/n = 0.8
    assert gini_normalized([0, 0, 0, 0, 1]) == pytest.approx(0.8, abs=1e-9)


def test_gini_high_for_heavy_tail() -> None:
    values = [0.01] * 99 + [100.0]
    assert gini_normalized(values) > 0.9


def test_quantile_summary_keys() -> None:
    out = quantile_summary([1, 2, 3, 4, 5])
    assert out["count"] == 5
    assert out["min"] == 1.0
    assert out["max"] == 5.0
    assert "p50" in out and "p99" in out


def test_concentration_stats_full_share_at_100pct() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    out = concentration_stats(values, top_fracs=(1.0,))
    assert out["top_1.0"] == pytest.approx(1.0)


def test_exposure_weighted_frequency() -> None:
    df = pd.DataFrame({"ClaimNb": [0, 1, 0, 2], "Exposure": [1.0, 1.0, 1.0, 1.0]})
    assert exposure_weighted_frequency(df) == pytest.approx(0.75)
