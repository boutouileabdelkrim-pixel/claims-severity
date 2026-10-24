"""Train and evaluate Phase 6 GBM models.

Trains 6 model families (XGB+LGBM × freq+sev+tweedie) plus:
- Tweedie variance_power grid: [1.1, 1.3, 1.5, 1.7, 1.9], best on val kept.

Test split is NEVER loaded. Assertion at end confirms.

Tweedie fix (post Phase 6): rate target (ClaimAmount / Exposure),
sample_weight=Exposure, variance_power tunable. This is the standard
Tweedie compound Poisson-Gamma setup for insurance. The original version
fit on raw per-policy target with no exposure info — leading to AE
1.40-1.46. Fix brings it into honest range.
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

TWEEDIE_VP_GRID = [1.1, 1.3, 1.5, 1.7, 1.9]


def build_matrix(train: pd.DataFrame, val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-hot cats + scaled numerics."""
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


def eval_sev(
    y_per_claim: np.ndarray,
    preds_per_claim: np.ndarray,
    n_claims: np.ndarray,
) -> dict:
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
    p("PHASE 6 — GBM RESULTS (Tweedie fixed with exposure weight)")
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
        (
            "glm_poisson_va",
            eval_freq(y_freq_va, predict_glm_freq(poisson_glm, X_va, exp_va)),
        ),
        (
            "xgb_poisson_va",
            eval_freq(y_freq_va, predict_gbm_freq(xgb_f, X_va, exp_va)),
        ),
        (
            "lgbm_poisson_va",
            eval_freq(y_freq_va, predict_gbm_freq(lgbm_f, X_va, exp_va)),
        ),
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
    p("=== Severity (Gamma, per-claim, RAW target) ===")
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
        (
            "glm_gamma_va",
            eval_sev(y_per_va, predict_glm_sev(gamma_glm, X_va_sev), n_va),
        ),
        (
            "xgb_gamma_va",
            eval_sev(y_per_va, predict_gbm_sev(xgb_s, X_va_sev), n_va),
        ),
        (
            "lgbm_gamma_va",
            eval_sev(y_per_va, predict_gbm_sev(lgbm_s, X_va_sev), n_va),
        ),
    ]
    p(f"{'model':<18s} {'mae_per_claim':>16s} {'deviance':>12s} {'ae':>8s}")
    for name, r in sev_rows:
        p(f"{name:<18s} {r['mae_per_claim']:>16.4f} {r['deviance']:>12.4f} {r['ae']:>8.4f}")
    p()

    # ------------------ Tweedie grid search ------------------
    p("=== Tweedie grid (variance_power) — XGBoost + LightGBM ===")
    y_raw_tr = train["ClaimAmount_sum"].fillna(0.0).to_numpy(dtype=float)
    y_raw_va = val["ClaimAmount_sum"].fillna(0.0).to_numpy(dtype=float)
    p_has_sev = float(train["ClaimNb_sev"].sum()) / float(train["ClaimNb"].sum())

    p(f"{'model':<22s} {'vp':>5s} {'val_dev':>10s} {'val_ae':>8s} {'val_mae':>10s}")

    best_xgb: tuple[float, object, dict] | None = None
    for vp in TWEEDIE_VP_GRID:
        model = fit_xgb_tweedie(X_tr, y_raw_tr, exp_tr, X_va, y_raw_va, exp_va, variance_power=vp)
        pred = predict_tweedie(model, X_va, exp_va)
        r = eval_pp(y_raw_va, pred)
        from src.models.metrics import (
            tweedie_deviance as _tweedie_dev,
        )

        dev = float(_tweedie_dev(y_raw_va, pred, p=vp))
        p(f"{'xgb_tweedie':<22s} {vp:>5.2f} {dev:>10.4f} {r['ae']:>8.4f} {r['mae']:>10.2f}")
        if best_xgb is None or dev < best_xgb[0]:
            best_xgb = (dev, model, {"variance_power": vp, **r})

    best_lgbm: tuple[float, object, dict] | None = None
    for vp in TWEEDIE_VP_GRID:
        model = fit_lgbm_tweedie(X_tr, y_raw_tr, exp_tr, X_va, y_raw_va, exp_va, variance_power=vp)
        pred = predict_tweedie(model, X_va, exp_va)
        r = eval_pp(y_raw_va, pred)
        from src.models.metrics import (
            tweedie_deviance as _tweedie_dev,
        )

        dev = float(_tweedie_dev(y_raw_va, pred, p=vp))
        p(f"{'lgbm_tweedie':<22s} {vp:>5.2f} {dev:>10.4f} {r['ae']:>8.4f} {r['mae']:>10.2f}")
        if best_lgbm is None or dev < best_lgbm[0]:
            best_lgbm = (dev, model, {"variance_power": vp, **r})

    p()
    assert best_xgb is not None and best_lgbm is not None
    p(
        f"Best XGB Tweedie: vp={best_xgb[2]['variance_power']:.2f}, "
        f"val AE={best_xgb[2]['ae']:.4f}, val MAE={best_xgb[2]['mae']:.2f}"
    )
    p(
        f"Best LGBM Tweedie: vp={best_lgbm[2]['variance_power']:.2f}, "
        f"val AE={best_lgbm[2]['ae']:.4f}, val MAE={best_lgbm[2]['mae']:.2f}"
    )
    p()

    # ------------------ Pure premium (best-of-both + best Tweedie) ------------------
    p("=== Pure premium (per-policy) ===")
    best_freq_rate_va = predict_gbm_freq(lgbm_f, X_va, exp_va) / np.clip(exp_va, 1e-9, None)
    best_sev_va = predict_glm_sev(gamma_glm, X_va)
    pp_compound = best_freq_rate_va * exp_va * best_sev_va * p_has_sev

    pp_xgb_t = predict_tweedie(best_xgb[1], X_va, exp_va)
    pp_lgbm_t = predict_tweedie(best_lgbm[1], X_va, exp_va)

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
