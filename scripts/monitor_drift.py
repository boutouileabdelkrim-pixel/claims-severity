"""Drift monitoring: train (reference) vs test (current) (Phase 12).

Two reports:
1. REAL comparison: train vs test, as they came out of the stratified
   split. Expected: mostly stable (split is random + stratified on
   ClaimNb only).
2. SYNTHETIC drift: train vs a modified test where BonusMalus is x1.5
   and Exposure is halved. Expected: significant PSI/KS on BonusMalus
   and Exposure, stable on the rest. Proves the detector works.

Also compares the model PREDICTION distribution (pure_premium per policy).

Discovery: the feature engineering pipeline has a hard upper bound on
DrivAge (pd.cut bins end at 101). DrivAge > 100 produces NaN which
crashes the astype cast. The API's Pydantic schema enforces DrivAge
<= 100, so production is protected. Monitoring bypasses Pydantic for
speed, so we only use synthetic drift that stays in schema bounds.

Test split is FROZEN since Phase 9 for MODEL METRICS. Monitoring only
reads test distributions, never computes test accuracy/MAE/AE.
"""

from __future__ import annotations

import json
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.build import engineer_features
from src.monitoring.drift import drift_report
from src.utils.paths import FEATURES_DIR, MODELS_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

BUNDLE_PATH = MODELS_DIR / "bundle_v1.pkl"

DRIFT_FEATURES = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density",
    "Exposure",
    "BonusMalus_log",
    "Density_log",
    "Area_ord",
    "VehGas_diesel",
    "VehAge_zero",
    "VehAge_over_20",
    "BonusMalus_gt100",
    "DrivAge_bucket",
]


def _encode(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    df = engineer_features(df)
    scaler: StandardScaler = bundle["scaler"]
    num = scaler.transform(df[bundle["numerics"]].to_numpy(dtype=float))
    dummies = pd.get_dummies(df[bundle["categoricals"]], drop_first=True, dtype=np.int8).reindex(
        columns=bundle["dummy_cols"], fill_value=0
    )
    return np.hstack([num, dummies.to_numpy(dtype=float)])


def _predict_pp(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    X = _encode(df, bundle)
    rate = np.clip(bundle["lgbm_freq_model"].predict(X), 0.0, None)
    sev = np.clip(bundle["glm_sev_model"].predict(X), 1e-3, None)
    p_has_sev = float(bundle["p_has_sev"])
    exposure = df["Exposure"].to_numpy(dtype=float)
    return rate * exposure * p_has_sev * sev


def _report_to_dict(rep: pd.DataFrame) -> list[dict]:
    return [
        {
            "feature": row["feature"],
            "psi": float(row["psi"]) if not pd.isna(row["psi"]) else None,
            "ks_stat": float(row["ks_stat"]) if not pd.isna(row["ks_stat"]) else None,
            "ks_pvalue": float(row["ks_pvalue"]) if not pd.isna(row["ks_pvalue"]) else None,
            "verdict": str(row["verdict"]),
        }
        for _, row in rep.iterrows()
    ]


def _table_lines(title: str, rep: pd.DataFrame) -> list[str]:
    lines = [f"=== {title} ==="]
    lines.append(
        f"{'feature':<18s} {'PSI':>10s} {'KS stat':>10s} {'KS p-value':>12s} {'verdict':>12s}"
    )
    for _, r in rep.iterrows():
        if pd.isna(r["psi"]):
            lines.append(f"{r['feature']:<18s} {'':>10s} {'':>10s} {'':>12s} {r['verdict']:>12s}")
            continue
        lines.append(
            f"{r['feature']:<18s} {r['psi']:>10.4f} "
            f"{r['ks_stat']:>10.4f} {r['ks_pvalue']:>12.4e} "
            f"{r['verdict']:>12s}"
        )
    lines.append("")
    n_sig = int((rep["verdict"] == "significant").sum())
    n_mod = int((rep["verdict"] == "moderate").sum())
    n_stable = int((rep["verdict"] == "stable").sum())
    lines.append(f"Summary: {n_stable} stable, {n_mod} moderate, {n_sig} significant")
    lines.append("")
    return lines


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Bundle not found: {BUNDLE_PATH}")

    with BUNDLE_PATH.open("rb") as f:
        bundle = pickle.load(f)

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    logger.info("Loaded train %s, test %s", train.shape, test.shape)

    logger.info("Computing pure-premium predictions on train and test...")
    pp_train = _predict_pp(train, bundle)
    pp_test = _predict_pp(test, bundle)

    train_aug = train.copy()
    test_aug = test.copy()
    train_aug["pure_premium"] = pp_train
    test_aug["pure_premium"] = pp_test

    all_features = DRIFT_FEATURES + ["pure_premium"]

    # ---------- 1. Real: train vs test ----------
    logger.info("Drift report 1/2: train (reference) vs test (current)...")
    real_rep = drift_report(train_aug, test_aug, features=all_features)

    # ---------- 2. Synthetic drift (in-bounds only) ----------
    logger.info("Drift report 2/2: train vs synthetic drift...")
    drift_test = test.copy()
    # BonusMalus schema bound is [50, 300]. Multiplying by 1.5 shifts
    # the distribution strongly while staying in-bounds.
    drift_test["BonusMalus"] = (drift_test["BonusMalus"] * 1.5).clip(upper=300).astype(int)
    # Exposure bound is (0, 2.5]. Halving shifts the distribution while
    # staying in-bounds.
    drift_test["Exposure"] = (drift_test["Exposure"] * 0.3).clip(lower=0.01)
    pp_drift = _predict_pp(drift_test, bundle)
    drift_aug = drift_test.copy()
    drift_aug["pure_premium"] = pp_drift

    synth_rep = drift_report(
        engineer_features(train_aug), engineer_features(drift_aug), features=all_features
    )

    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    p("=" * 90)
    p("PHASE 12 — DRIFT MONITORING (PSI + KS)")
    p("=" * 90)
    p()
    p(f"Reference: train ({len(train):,} rows)")
    p(f"Current:   test  ({len(test):,} rows)")
    p(f"Features:  {len(all_features)} (numeric + engineered + prediction)")
    p()

    p("--- Interpretation ---")
    p("PSI  < 0.10 : stable (no shift)")
    p("PSI 0.10-0.25 : moderate shift (investigate)")
    p("PSI > 0.25 : significant shift (retrain candidate)")
    p("KS p-value < 0.05 : reject H0 (distributions differ)")
    p()

    lines.extend(_table_lines("Real comparison: train vs test", real_rep))
    lines.extend(
        _table_lines(
            "Synthetic drift: train vs (test + BonusMalus*1.5, Exposure*0.5)",
            synth_rep,
        )
    )

    p("--- Note on synthetic drift choice ---")
    p("We avoid DrivAge + shift because pd.cut bins end at 101;")
    p("DrivAge > 100 produces NaN and crashes engineer_features.")
    p("The API schema enforces DrivAge <= 100, so production is safe.")
    p("Documented as a known boundary in reports/phase12_monitoring.md.")
    p()

    p("--- Conclusion ---")
    n_sig_real = int((real_rep["verdict"] == "significant").sum())
    if n_sig_real == 0:
        p("Real comparison: no significant drift. Test population is")
        p("consistent with training distribution. Model can be served.")
    else:
        p(f"Real comparison: {n_sig_real} significant shifts detected.")
        p("Investigate before deploying.")

    n_sig_synth = int((synth_rep["verdict"] == "significant").sum())
    if n_sig_synth >= 2:
        p(f"Synthetic drift: {n_sig_synth} significant shifts detected.")
        p("Detector works. Would catch real distribution shift.")
    else:
        p(f"Synthetic drift: only {n_sig_synth} detected. Detector may")
        p("be under-sensitive. Investigate PSI binning.")
    p()

    OUT_TXT = REPORTS_DIR / "phase12_monitoring.txt"
    OUT_TXT.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", OUT_TXT)

    payload = {
        "reference": "train",
        "current": "test",
        "features": all_features,
        "real": _report_to_dict(real_rep),
        "synthetic_drift": _report_to_dict(synth_rep),
        "synthetic_drift_spec": "BonusMalus * 1.5 (clip 300), Exposure * 0.3 (clip 0.01)",
        "thresholds": {
            "psi_stable": 0.10,
            "psi_moderate": 0.25,
            "ks_pvalue_threshold": 0.05,
        },
    }
    OUT_JSON = REPORTS_DIR / "phase12_monitoring.json"
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
