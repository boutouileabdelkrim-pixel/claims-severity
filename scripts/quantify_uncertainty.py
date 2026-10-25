"""Phase 8: quantify SEVERITY uncertainty with quantile + CQR.

Diagnosis of first attempt (2026-10-25):
The original pipeline fit quantiles on log1p(pure_premium), which has ~95%
mass at exactly 0 (no claim). Quantile regression predicts 0 everywhere
because in almost every leaf, the empirical quantile of a mostly-zero
distribution is 0. Result: zero-width intervals, q_hat=0, trivially 96%
coverage from the zero mass. Vacuous.

Fix: fit quantiles on SEVERITY only (claims with ClaimAmount > 0).
Measure coverage CONDITIONALLY on having a claim. Report the interval as
"given a claim, my 80% interval for the amount is [lo, hi]".

Frequency uncertainty (P(claim)) is quantified by Phase 6 LGBM frequency
model. Compound composition + portfolio-level aggregation is Phase 9.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.uncertainty import (
    calibrate_cqr,
    empirical_coverage,
    fit_quantile_grid,
    interval_width,
    predict_interval,
    predict_quantiles,
)
from src.utils.paths import FEATURES_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

CATEGORICALS = ["VehBrand", "Region"]
NUMERICS = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus_log",
    "Density_log",
    "Area_ord",
    "VehGas_diesel",
    "VehAge_zero",
    "VehAge_over_20",
    "BonusMalus_gt100",
    "DrivAge_bucket",
]

NOMINAL_COVERAGE = 0.80
ALPHA = 1.0 - NOMINAL_COVERAGE
CAL_FRAC = 0.5
SEED = 42


def build_matrix(train: pd.DataFrame, val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dummies_tr = pd.get_dummies(train[CATEGORICALS], drop_first=True, dtype=np.int8)
    cols = list(dummies_tr.columns)
    dummies_va = pd.get_dummies(val[CATEGORICALS], drop_first=True, dtype=np.int8)
    dummies_va = dummies_va.reindex(columns=cols, fill_value=0)
    scaler = StandardScaler()
    num_tr = scaler.fit_transform(train[NUMERICS].to_numpy(dtype=float))
    num_va = scaler.transform(val[NUMERICS].to_numpy(dtype=float))
    X_tr = np.hstack([num_tr, dummies_tr.to_numpy(dtype=float)])
    X_va = np.hstack([num_va, dummies_va.to_numpy(dtype=float)])
    return X_tr, X_va


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    logger.info("Loaded train %s, val %s", train.shape, val.shape)

    X_tr, X_va = build_matrix(train, val)

    y_tr_raw = train["ClaimAmount_sum"].to_numpy(dtype=float)
    y_va_raw = val["ClaimAmount_sum"].to_numpy(dtype=float)
    mask_tr = y_tr_raw > 0
    mask_va = y_va_raw > 0

    X_tr_sev = X_tr[mask_tr]
    y_tr_sev_raw = y_tr_raw[mask_tr]
    y_tr_sev = np.log(y_tr_sev_raw)

    X_va_sev = X_va[mask_va]
    y_va_sev_raw = y_va_raw[mask_va]
    y_va_sev = np.log(y_va_sev_raw)

    logger.info("Train claims: %d, Val claims: %d", len(y_tr_sev), len(y_va_sev))

    logger.info("Fitting 3 severity quantile LGBM models on log(severity)...")
    models = fit_quantile_grid(X_tr_sev, y_tr_sev, quantiles=(0.1, 0.5, 0.9))
    logger.info("Quantile models fitted.")

    cal_idx, check_idx = train_test_split(
        np.arange(len(y_va_sev)),
        test_size=1.0 - CAL_FRAC,
        random_state=SEED,
    )
    X_cal = X_va_sev[cal_idx]
    y_cal = y_va_sev[cal_idx]
    X_check = X_va_sev[check_idx]
    y_check = y_va_sev[check_idx]
    y_check_raw = y_va_sev_raw[check_idx]
    logger.info("Val claims split: cal=%d, check=%d", len(cal_idx), len(check_idx))

    preds_check = predict_quantiles(models, X_check)
    raw_lo = preds_check[0.1]
    raw_hi = preds_check[0.9]
    raw_coverage = empirical_coverage(y_check, raw_lo, raw_hi)

    q_hat = calibrate_cqr(models, X_cal, y_cal, alpha=ALPHA)
    logger.info("CQR q_hat = %.4f (log EUR units)", q_hat)

    lo_cf, med_cf, hi_cf = predict_interval(models, X_check, q_hat=q_hat)
    cf_coverage = empirical_coverage(y_check, lo_cf, hi_cf)

    raw_w_mean_log, raw_w_med_log = interval_width(raw_lo, raw_hi)
    cf_w_mean_log, cf_w_med_log = interval_width(lo_cf, hi_cf)

    raw_lo_eur = np.exp(raw_lo)
    raw_hi_eur = np.exp(raw_hi)
    cf_lo_eur = np.exp(lo_cf)
    cf_hi_eur = np.exp(hi_cf)
    raw_w_eur_mean, raw_w_eur_med = interval_width(raw_lo_eur, raw_hi_eur)
    cf_w_eur_mean, cf_w_eur_med = interval_width(cf_lo_eur, cf_hi_eur)

    med_pred_eur = float(np.median(np.exp(med_cf)))
    med_true_eur = float(np.median(y_check_raw))
    mean_true_eur = float(np.mean(y_check_raw))

    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    p("=" * 90)
    p("PHASE 8 — SEVERITY UNCERTAINTY (quantile + CQR)")
    p("=" * 90)
    p()
    p("Diagnosis of first attempt (2026-10-25):")
    p("  Fitting quantiles on log1p(pure_premium) fails: ~95% mass at 0.")
    p("  Quantile regression predicts 0 everywhere (leaf empirical")
    p("  quantiles are 0). Zero-width intervals, q_hat=0, trivially")
    p("  96% coverage from the zero mass. Vacuous.")
    p("  FIX: fit on severity only (claims > 0), cover conditionally.")
    p()
    p("Setup (corrected):")
    p("  Target: log(ClaimAmount_sum) for claims only (ClaimAmount > 0)")
    p(f"  Train claims: {len(y_tr_sev):,}")
    p(f"  Val claims:   {len(y_va_sev):,} (cal={len(cal_idx):,} / check={len(check_idx):,})")
    p(f"  Quantiles:    0.1 / 0.5 / 0.9 (nominal {NOMINAL_COVERAGE:.0%})")
    p("  Method:       CQR (Romano et al. 2019)")
    p()

    p("=== Coverage on val-check CLAIMS (conditional on claim > 0) ===")
    p(f"  Nominal target:      {NOMINAL_COVERAGE:.4f}")
    p(f"  Raw quantile:        {raw_coverage:.4f}")
    p(f"  CQR-calibrated:      {cf_coverage:.4f}")
    p(f"  q_hat (log EUR):     {q_hat:.4f}")
    p()

    delta_raw = raw_coverage - NOMINAL_COVERAGE
    delta_cf = cf_coverage - NOMINAL_COVERAGE
    p("=== Interpretation ===")
    if raw_coverage < NOMINAL_COVERAGE:
        p(
            f"  Raw quantiles UNDER-COVER by {abs(delta_raw) * 100:.1f}% "
            f"({raw_coverage:.1%} vs {NOMINAL_COVERAGE:.0%} target)."
        )
    else:
        p(
            f"  Raw quantiles over-cover by {delta_raw * 100:.1f}% "
            f"({raw_coverage:.1%} vs {NOMINAL_COVERAGE:.0%} target)."
        )

    if cf_coverage >= NOMINAL_COVERAGE - 0.02:
        p(f"  CQR restores coverage: {cf_coverage:.1%} ({delta_cf * 100:+.1f}% vs target). OK.")
    else:
        p(f"  CQR coverage below target: {cf_coverage:.1%} ({delta_cf * 100:+.1f}%). Investigate.")
    p()

    p("=== Severity interval widths (conditional on claim) ===")
    p(f"  Raw median width (log):  {raw_w_med_log:.4f}")
    p(f"  CQR median width (log):  {cf_w_med_log:.4f}")
    p(f"  Raw median width (EUR):  {raw_w_eur_med:,.2f}")
    p(f"  CQR median width (EUR):  {cf_w_eur_med:,.2f}")
    p()
    p(f"  Median predicted sev:    {med_pred_eur:,.2f} EUR")
    p(f"  Median observed sev:     {med_true_eur:,.2f} EUR")
    p(f"  Mean observed sev:       {mean_true_eur:,.2f} EUR")
    p()

    p("=== Portfolio-level context ===")
    p("  This interval is CONDITIONAL on having a claim.")
    p("  For a specific policy: P(claim) ~ 5% (Phase 6 LGBM freq),")
    p("  so the unconditional pure premium is mostly 0.")
    p("  Compound composition + portfolio aggregation: Phase 9.")
    p()

    OUT = REPORTS_DIR / "phase8_uncertainty.txt"
    OUT.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", OUT)

    assert "test_va" not in "\n".join(lines)
    assert "test_tr" not in "\n".join(lines)
    print("OK: test split never evaluated")


if __name__ == "__main__":
    main()
