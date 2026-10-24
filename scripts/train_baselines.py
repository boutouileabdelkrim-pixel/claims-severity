"""Train and evaluate Phase 5 baselines.

Loads train + val ONLY (test reserved for Phase 9).
Fits 3 baselines per family:
  - Dummy (constant)
  - GLM Poisson (frequency, rate target + sample_weight=Exposure)
  - GLM Gamma (severity, raw ClaimAmount_sum per claim)

Corrective factors:
  - StandardScaler on numeric features (fixes LBFGS convergence)
  - p_has_sev on train (corrects pure premium for 9,116 claims w/o sev)
  - Severity target uses RAW ClaimAmount_sum, not capped (cap is Phase 6)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.baseline import (
    DummyFrequency,
    DummySeverity,
    compose_pure_premium,
    compute_p_has_sev,
    fit_gamma_sev,
    fit_poisson_freq,
    predict_freq_count,
    predict_severity_per_claim,
)
from src.models.metrics import (
    actual_expected_ratio,
    gamma_deviance,
    mae,
    poisson_deviance,
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


def build_glm_matrix(train: pd.DataFrame, val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-hot cat + scale numerics (fit on train). Returns (X_train, X_val)."""
    dummies_tr = pd.get_dummies(train[CATEGORICALS], drop_first=True, dtype=np.int8)
    dummy_cols = list(dummies_tr.columns)
    dummies_va = pd.get_dummies(val[CATEGORICALS], drop_first=True, dtype=np.int8)
    dummies_va = dummies_va.reindex(columns=dummy_cols, fill_value=0)

    scaler = StandardScaler()
    num_tr = scaler.fit_transform(train[NUMERICS].to_numpy(dtype=float))
    num_va = scaler.transform(val[NUMERICS].to_numpy(dtype=float))

    X_tr = np.hstack([num_tr, dummies_tr.to_numpy(dtype=float)])
    X_va = np.hstack([num_va, dummies_va.to_numpy(dtype=float)])
    return X_tr, X_va


def eval_freq(df, preds, label) -> dict:
    y = df["ClaimNb"].to_numpy(dtype=float)
    return {
        "label": label,
        "n": len(df),
        "mae": mae(y, preds),
        "poisson_deviance": poisson_deviance(y, preds),
        "actual_expected": actual_expected_ratio(y, preds),
        "total_actual": float(y.sum()),
        "total_predicted": float(preds.sum()),
    }


def eval_sev(df, preds_per_claim, label) -> dict:
    mask = df["ClaimNb_sev"] > 0
    sub = df.loc[mask]
    y_total = sub["ClaimAmount_sum"].to_numpy(dtype=float)
    n_claims = sub["ClaimNb_sev"].to_numpy(dtype=float)
    y_per_claim = y_total / n_claims
    pred_subset = preds_per_claim[mask.to_numpy()]
    return {
        "label": label,
        "n_policies": len(sub),
        "n_claims": int(n_claims.sum()),
        "mae_per_claim": mae(y_per_claim, pred_subset),
        "gamma_deviance": gamma_deviance(y_per_claim, pred_subset, sample_weight=n_claims),
        "actual_expected": actual_expected_ratio(y_per_claim, pred_subset),
        "total_actual": float(y_total.sum()),
        "total_predicted": float((pred_subset * n_claims).sum()),
    }


def eval_pp(df, pp_preds, label) -> dict:
    y_total = df["ClaimAmount_sum"].fillna(0.0).to_numpy(dtype=float)
    return {
        "label": label,
        "mae": mae(y_total, pp_preds),
        "actual_expected": actual_expected_ratio(y_total, pp_preds),
        "total_actual": float(y_total.sum()),
        "total_predicted": float(pp_preds.sum()),
    }


def print_block(title: str, rows: list[dict], keys: list[str]) -> list[str]:
    lines = [f"=== {title} ==="]
    header = f"{'model':<12s} " + " ".join(f"{k:>16s}" for k in keys)
    lines.append(header)
    for r in rows:
        line = f"{r['label']:<12s} "
        for k in keys:
            v = r[k]
            if isinstance(v, float):
                line += f" {v:>16.4f}"
            else:
                line += f" {v:>16,}"
        lines.append(line)
    return lines


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    logger.info("Loaded train %s, val %s", train.shape, val.shape)

    p_has_sev = compute_p_has_sev(train)
    logger.info("p_has_sev (train) = %.4f", p_has_sev)

    X_train, X_val = build_glm_matrix(train, val)
    logger.info("GLM matrix: %d features", X_train.shape[1])

    exp_train = train["Exposure"].to_numpy(dtype=float)
    exp_val = val["Exposure"].to_numpy(dtype=float)

    # --- Frequency ---
    y_freq_train = train["ClaimNb"].to_numpy(dtype=float)
    dummy_f = DummyFrequency().fit(train)
    poisson_f = fit_poisson_freq(X_train, y_freq_train, exp_train)

    freq_results = [
        eval_freq(train, dummy_f.predict(train), "dummy_tr"),
        eval_freq(val, dummy_f.predict(val), "dummy_va"),
        eval_freq(train, predict_freq_count(poisson_f, X_train, exp_train), "poisson_tr"),
        eval_freq(val, predict_freq_count(poisson_f, X_val, exp_val), "poisson_va"),
    ]

    # --- Severity (RAW target) ---
    mask_train = (train["ClaimNb_sev"] > 0).to_numpy()
    y_sev_train = train.loc[mask_train, "ClaimAmount_sum"].to_numpy(dtype=float)
    n_claims_train = train.loc[mask_train, "ClaimNb_sev"].to_numpy(dtype=float)

    dummy_s = DummySeverity().fit(train)
    gamma_s = fit_gamma_sev(X_train[mask_train], y_sev_train, n_claims_train)

    sev_results = [
        eval_sev(train, dummy_s.predict(train), "dummy_tr"),
        eval_sev(val, dummy_s.predict(val), "dummy_va"),
        eval_sev(train, predict_severity_per_claim(gamma_s, X_train), "gamma_tr"),
        eval_sev(val, predict_severity_per_claim(gamma_s, X_val), "gamma_va"),
    ]

    # --- Pure premium (with p_has_sev correction) ---
    dummy_rate_tr = np.full(len(train), dummy_f.rate_)
    dummy_rate_va = np.full(len(val), dummy_f.rate_)
    glm_rate_tr = poisson_f.predict(X_train)
    glm_rate_va = poisson_f.predict(X_val)

    sev_dummy_tr = np.full(len(train), dummy_s.mean_)
    sev_dummy_va = np.full(len(val), dummy_s.mean_)
    sev_glm_tr = predict_severity_per_claim(gamma_s, X_train)
    sev_glm_va = predict_severity_per_claim(gamma_s, X_val)

    dummy_pp_tr = compose_pure_premium(dummy_rate_tr, exp_train, sev_dummy_tr, p_has_sev)
    dummy_pp_va = compose_pure_premium(dummy_rate_va, exp_val, sev_dummy_va, p_has_sev)
    glm_pp_tr = compose_pure_premium(glm_rate_tr, exp_train, sev_glm_tr, p_has_sev)
    glm_pp_va = compose_pure_premium(glm_rate_va, exp_val, sev_glm_va, p_has_sev)

    pp_results = [
        eval_pp(train, dummy_pp_tr, "dummy_tr"),
        eval_pp(val, dummy_pp_va, "dummy_va"),
        eval_pp(train, glm_pp_tr, "glm_tr"),
        eval_pp(val, glm_pp_va, "glm_va"),
    ]

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("PHASE 5 — BASELINE RESULTS (with corrections)")
    lines.append("=" * 100)
    lines.append(f"p_has_sev (train) = {p_has_sev:.4f}   (fraction of claims with sev record)")
    lines.append(
        f"GLM features: {X_train.shape[1]}  ({len(NUMERICS)} scaled numeric + one-hot cat)"
    )
    lines.append("")
    lines.extend(
        print_block(
            "Frequency (rate target)",
            freq_results,
            ["n", "mae", "poisson_deviance", "actual_expected", "total_actual", "total_predicted"],
        )
    )
    lines.append("")
    lines.extend(
        print_block(
            "Severity (raw target, per-claim)",
            sev_results,
            [
                "n_policies",
                "n_claims",
                "mae_per_claim",
                "gamma_deviance",
                "actual_expected",
                "total_actual",
                "total_predicted",
            ],
        )
    )
    lines.append("")
    lines.extend(
        print_block(
            "Pure premium (raw target, p_has_sev corrected)",
            pp_results,
            ["mae", "actual_expected", "total_actual", "total_predicted"],
        )
    )

    OUT = REPORTS_DIR / "phase5_baselines.txt"
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    print(text)
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
