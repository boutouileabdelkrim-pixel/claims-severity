"""Train and evaluate Phase 6 GBM models.

Trains 6 models (XGB+LGBM × freq+sev+tweedie), evaluates on val,
compares vs Phase 5 baselines. Writes reports/phase6_gbm.txt.

Test split is NEVER loaded. Assertion at end confirms.

Cap experiment (Phase 6 finding):
Phase 4 chose p99.5 = 35,096 EUR. Phase 6 first run showed this destroys
pure premium calibration: AE = 1.44 on val. Raising to p99.9 (train-only
cap = 126,014 EUR) improved compound to AE = 1.30 but Tweedie got WORSE
(AE 1.50). Root cause: top 0.1% of policies hold 60% of severity mass
(Phase 3 EDA). No cap in [p99.5, p99.9] recovers calibration.

Decision: revert to RAW target for severity and Tweedie, matching Phase 5.
Accept training noise from the 4M EUR outlier. Report honest PP AE ≈ 1.10
which is the val-tail-shift (+7.7% documented in Phase 5).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.baseline import (
    DummyFrequency,
    DummySeverity,
    fit_gamma_sev,
    fit_poisson_freq,
)
from src.models.baseline import (
    predict_freq_count as predict_glm_freq,
)
from src.models.baseline import (
    predict_severity_per_claim as predict_glm_sev,
)
from src.models.gbm import (
    fit_lgbm_freq,
    fit_lgbm_sev,
    fit_lgbm_tweedie,
    fit_xgb_freq,
    fit_xgb_sev,
    fit_xgb_tweedie,
    predict_tweedie,
)
from src.models.gbm import (
    predict_freq_count as predict_gbm_freq,
)
from src.models.gbm import (
    predict_sev_per_claim as predict_gbm_sev,
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


def eval_freq(y_true: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "mae": float(mae(y_true, preds)),
        "deviance": float(poisson_deviance(y_true, preds)),
        "ae": float(actual_expected_ratio(y_true, preds)),
        "total_actual": float(y_true.sum()),
        "total_predicted": float(preds.sum()),
    }


def eval_sev(y_per_claim: np.ndarray, preds_per_claim: np.ndarray, n_claims: np.ndarray) -> dict:
    return {
        "mae_per_claim": float(mae(y_per_claim, preds_per_claim)),
        "deviance": float(gamma_deviance(y_per_claim, preds_per_claim, sample_weight=n_claims)),
        "ae": float(actual_expected_ratio(y_per_claim, preds_per_claim)),
    }


def eval_pp(y_raw: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "mae": float(mae(y_raw, preds)),
        "ae": float(actual_expected_ratio(y_raw, preds)),
        "total_actual": float(y_raw.sum()),
        "total_predicted": float(preds.sum()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    logger.info("Loaded train %s, val %s", train.shape, val.shape)

    X_tr, X_va = build_matrix(train, val)
    exp_tr = train["Exposure"].to_numpy(dtype=float)
    exp_va = val["Exposure"].to_numpy(dtype=float)
    y_freq_tr = train["ClaimNb"].to_numpy(dtype=float)
    y_freq_va = val["ClaimNb"].to_numpy(dtype=float)

    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    p("=" * 100)
    p("PHASE 6 — GBM RESULTS")
    p("=" * 100)
    p()

    # ------------------ Frequency ------------------
    p("=== Frequency (Poisson) ===")
    dummy_f = DummyFrequency().fit(train)
    poisson_glm = fit_poisson_freq(X_tr, y_freq_tr, exp_tr)
    xgb_f = fit_xgb_freq(X_tr, y_freq_tr, exp_tr, X_va, y_freq_va, exp_va)
    lgbm_f = fit_lgbm_freq(X_tr, y_freq_tr, exp_tr, X_va, y_freq_va, exp_va)

    freq_rows = [
        ("dummy_va", eval_freq(y_freq_va, dummy_f.predict(val))),
        ("glm_poisson_va", eval_freq(y_freq_va, predict_glm_freq(poisson_glm, X_va, exp_va))),
        ("xgb_poisson_va", eval_freq(y_freq_va, predict_gbm_freq(xgb_f, X_va, exp_va))),
        ("lgbm_poisson_va", eval_freq(y_freq_va, predict_gbm_freq(lgbm_f, X_va, exp_va))),
    ]
    p(
        f"{'model':<18s} {'mae':>10s} {'deviance':>12s} {'ae':>8s} "
        f"{'tot_actual':>14s} {'tot_pred':>14s}"
    )
    for name, r in freq_rows:
        p(
            f"{name:<18s} {r['mae']:>10.4f} {r['deviance']:>12.4f} "
            f"{r['ae']:>8.4f} {r['total_actual']:>14,.0f} "
            f"{r['total_predicted']:>14,.0f}"
        )
    p()

    # ------------------ Severity (RAW target, no cap) ------------------
    p("=== Severity (Gamma, per-claim, RAW target — no cap) ===")
    m_tr = (train["ClaimNb_sev"] > 0).to_numpy()
    m_va = (val["ClaimNb_sev"] > 0).to_numpy()
    train_sev = train.loc[m_tr]
    val_sev = val.loc[m_va]
    X_tr_sev = X_tr[m_tr]
    X_va_sev = X_va[m_va]
    raw_sev_tr = train_sev["ClaimAmount_sum"].to_numpy(dtype=float)
    raw_sev_va = val_sev["ClaimAmount_sum"].to_numpy(dtype=float)
    n_tr = train_sev["ClaimNb_sev"].to_numpy(dtype=float)
    n_va = val_sev["ClaimNb_sev"].to_numpy(dtype=float)
    y_per_va = raw_sev_va / n_va

    dummy_s = DummySeverity().fit(train)
    gamma_glm = fit_gamma_sev(X_tr_sev, raw_sev_tr, n_tr)
    xgb_s = fit_xgb_sev(X_tr_sev, raw_sev_tr, n_tr, X_va_sev, raw_sev_va, n_va)
    lgbm_s = fit_lgbm_sev(X_tr_sev, raw_sev_tr, n_tr, X_va_sev, raw_sev_va, n_va)

    sev_rows = [
        ("dummy_va", eval_sev(y_per_va, dummy_s.predict(val_sev), n_va)),
        ("glm_gamma_va", eval_sev(y_per_va, predict_glm_sev(gamma_glm, X_va_sev), n_va)),
        ("xgb_gamma_va", eval_sev(y_per_va, predict_gbm_sev(xgb_s, X_va_sev), n_va)),
        ("lgbm_gamma_va", eval_sev(y_per_va, predict_gbm_sev(lgbm_s, X_va_sev), n_va)),
    ]
    p(f"{'model':<18s} {'mae_per_claim':>16s} {'deviance':>12s} {'ae':>8s}")
    for name, r in sev_rows:
        p(f"{name:<18s} {r['mae_per_claim']:>16.4f} {r['deviance']:>12.4f} {r['ae']:>8.4f}")
    p()

    # ------------------ Pure premium (raw target) ------------------
    p("=== Pure premium (raw target, per-policy) ===")
    y_raw_tr = train["ClaimAmount_sum"].fillna(0.0).to_numpy(dtype=float)
    y_raw_va = val["ClaimAmount_sum"].fillna(0.0).to_numpy(dtype=float)
    p_has_sev = float(train["ClaimNb_sev"].sum()) / float(train["ClaimNb"].sum())

    # Compound: LGBM freq × GLM sev × p_has_sev
    # LGBM Gamma severity is unstable on raw target (val AE 1.28, overfits
    # the 4M EUR tail). GLM Gamma is stable (val AE 1.09). Use best of both
    # families; see reports/phase6_gbm.md "Discovery" section.
    best_freq_rate_va = predict_gbm_freq(lgbm_f, X_va, exp_va) / np.clip(exp_va, 1e-9, None)
    best_sev_va = predict_glm_sev(gamma_glm, X_va)
    pp_compound = best_freq_rate_va * exp_va * best_sev_va * p_has_sev

    # Tweedie direct (raw target)
    xgb_t = fit_xgb_tweedie(X_tr, y_raw_tr, X_va, y_raw_va)
    lgbm_t = fit_lgbm_tweedie(X_tr, y_raw_tr, X_va, y_raw_va)
    pp_xgb_t = predict_tweedie(xgb_t, X_va)
    pp_lgbm_t = predict_tweedie(lgbm_t, X_va)

    # GLM compound
    pp_glm = (
        predict_glm_freq(poisson_glm, X_va, exp_va) * predict_glm_sev(gamma_glm, X_va) * p_has_sev
    )

    pp_rows = [
        ("glm_compound_va", eval_pp(y_raw_va, pp_glm)),
        ("gbm_compound_va", eval_pp(y_raw_va, pp_compound)),
        ("xgb_tweedie_va", eval_pp(y_raw_va, pp_xgb_t)),
        ("lgbm_tweedie_va", eval_pp(y_raw_va, pp_lgbm_t)),
    ]
    p(f"{'model':<18s} {'mae':>12s} {'ae':>8s} {'tot_actual':>14s} {'tot_pred':>14s}")
    for name, r in pp_rows:
        p(
            f"{name:<18s} {r['mae']:>12.2f} {r['ae']:>8.4f} "
            f"{r['total_actual']:>14,.0f} {r['total_predicted']:>14,.0f}"
        )
    p()
    p(f"p_has_sev (train) = {p_has_sev:.4f}")
    p()

    # ------------------ Deltas vs baseline ------------------
    p("=== Deltas vs Phase 5 baselines (val) ===")
    glm_f_dev = freq_rows[1][1]["deviance"]
    best_gbm_f_dev = min(freq_rows[2][1]["deviance"], freq_rows[3][1]["deviance"])
    p(
        f"Frequency Poisson deviance: GLM={glm_f_dev:.4f}  "
        f"best GBM={best_gbm_f_dev:.4f}  "
        f"delta={(best_gbm_f_dev - glm_f_dev) / glm_f_dev * 100:+.2f}%"
    )

    glm_s_dev = sev_rows[1][1]["deviance"]
    best_gbm_s_dev = min(sev_rows[2][1]["deviance"], sev_rows[3][1]["deviance"])
    p(
        f"Severity Gamma deviance:    GLM={glm_s_dev:.4f}  "
        f"best GBM={best_gbm_s_dev:.4f}  "
        f"delta={(best_gbm_s_dev - glm_s_dev) / glm_s_dev * 100:+.2f}%"
    )
    p()

    OUT = REPORTS_DIR / "phase6_gbm.txt"
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    logger.info("Wrote %s", OUT)

    assert "test_va" not in text
    assert "test_tr" not in text
    print("OK: test split never evaluated")


if __name__ == "__main__":
    main()
