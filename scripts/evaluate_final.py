"""Phase 9: final evaluation on test split (touched once).

Refits the chosen compound model (LGBM freq × GLM sev × p_has_sev) and
evaluates on test. Also:
- Actuarial metrics: MAE, AE, Poisson deviance, Gamma deviance, Gini, lift
- SHAP feature importance on the LGBM frequency model (PNG)
- Fairness: AE and residuals by DrivAge_bucket, Region (top-5), Area
- Conditional CQR coverage on test claims (frozen q_hat from Phase 8)

After success, creates data/test_frozen.flag. Subsequent runs refuse
unless --force passed.
"""

from __future__ import annotations

import logging
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend, safe for CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.models.actuarial import (  # noqa: E402
    gini_normalized_predictions,
    lift_by_decile,
)
from src.models.baseline import (  # noqa: E402
    fit_gamma_sev,
)
from src.models.baseline import (
    predict_severity_per_claim as predict_glm_sev,
)
from src.models.fairness import residuals_by_group  # noqa: E402
from src.models.gbm import fit_lgbm_freq, predict_freq_count  # noqa: E402
from src.models.metrics import (  # noqa: E402
    actual_expected_ratio,
    gamma_deviance,
    mae,
    poisson_deviance,
)
from src.models.uncertainty import (  # noqa: E402
    empirical_coverage,
    fit_quantile_grid,
    predict_interval,
)
from src.utils.paths import DATA_DIR, FEATURES_DIR, REPORTS_DIR  # noqa: E402

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

Q_HAT_PHASE8 = 0.2714  # frozen from reports/phase8_uncertainty.txt
SENTINEL = DATA_DIR / "test_frozen.flag"


def build_matrix(train, val, test):
    dummies_tr = pd.get_dummies(train[CATEGORICALS], drop_first=True, dtype=np.int8)
    cols = list(dummies_tr.columns)
    dummies_va = pd.get_dummies(val[CATEGORICALS], drop_first=True, dtype=np.int8).reindex(
        columns=cols, fill_value=0
    )
    dummies_te = pd.get_dummies(test[CATEGORICALS], drop_first=True, dtype=np.int8).reindex(
        columns=cols, fill_value=0
    )
    scaler = StandardScaler()
    num_tr = scaler.fit_transform(train[NUMERICS].to_numpy(dtype=float))
    num_va = scaler.transform(val[NUMERICS].to_numpy(dtype=float))
    num_te = scaler.transform(test[NUMERICS].to_numpy(dtype=float))
    X_tr = np.hstack([num_tr, dummies_tr.to_numpy(dtype=float)])
    X_va = np.hstack([num_va, dummies_va.to_numpy(dtype=float)])
    X_te = np.hstack([num_te, dummies_te.to_numpy(dtype=float)])
    names = NUMERICS + cols
    return X_tr, X_va, X_te, names


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    force = "--force" in sys.argv
    if SENTINEL.exists() and not force:
        print(f"ERROR: {SENTINEL} exists — test split already touched.")
        print("Pass --force if you really want to re-run (Phase 9+ may not re-touch).")
        sys.exit(1)

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    logger.info("Loaded train %s, val %s, test %s", train.shape, val.shape, test.shape)

    X_tr, X_va, X_te, feature_names = build_matrix(train, val, test)

    exp_tr = train["Exposure"].to_numpy(dtype=float)
    exp_va = val["Exposure"].to_numpy(dtype=float)
    exp_te = test["Exposure"].to_numpy(dtype=float)
    y_freq_tr = train["ClaimNb"].to_numpy(dtype=float)
    y_freq_va = val["ClaimNb"].to_numpy(dtype=float)
    y_freq_te = test["ClaimNb"].to_numpy(dtype=float)

    logger.info("Fitting LGBM frequency on train...")
    lgbm_f = fit_lgbm_freq(X_tr, y_freq_tr, exp_tr, X_va, y_freq_va, exp_va)

    m_tr = (train["ClaimNb_sev"] > 0).to_numpy()
    raw_sev_tr = train.loc[m_tr, "ClaimAmount_sum"].to_numpy(dtype=float)
    n_tr = train.loc[m_tr, "ClaimNb_sev"].to_numpy(dtype=float)

    logger.info("Fitting GLM Gamma severity on train claims...")
    glm_s = fit_gamma_sev(X_tr[m_tr], raw_sev_tr, n_tr)

    p_has_sev = float(train["ClaimNb_sev"].sum()) / float(train["ClaimNb"].sum())

    logger.info("Predicting on test...")
    freq_rate_te = predict_freq_count(lgbm_f, X_te, exp_te) / np.clip(exp_te, 1e-9, None)
    sev_te = predict_glm_sev(glm_s, X_te)
    pp_te = freq_rate_te * exp_te * sev_te * p_has_sev

    y_pp_te = test["ClaimAmount_sum"].fillna(0.0).to_numpy(dtype=float)

    pp_mae = mae(y_pp_te, pp_te)
    pp_ae = actual_expected_ratio(y_pp_te, pp_te)
    freq_dev = poisson_deviance(y_freq_te, freq_rate_te * exp_te)
    freq_ae = actual_expected_ratio(y_freq_te, freq_rate_te * exp_te)

    m_te = (test["ClaimNb_sev"] > 0).to_numpy()
    raw_sev_te = test.loc[m_te, "ClaimAmount_sum"].to_numpy(dtype=float)
    n_te = test.loc[m_te, "ClaimNb_sev"].to_numpy(dtype=float)
    sev_pred_te = sev_te[m_te]
    sev_dev = gamma_deviance(raw_sev_te / n_te, sev_pred_te, sample_weight=n_te)
    sev_ae = actual_expected_ratio(raw_sev_te / n_te, sev_pred_te)

    pp_gini = gini_normalized_predictions(y_pp_te, pp_te)
    lift = lift_by_decile(y_pp_te, pp_te, n_deciles=10)

    logger.info("Computing SHAP on freq model (5000-row sample)...")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_te), size=min(5000, len(X_te)), replace=False)
    X_shap = X_te[idx]
    explainer = shap.TreeExplainer(lgbm_f)
    shap_values = explainer.shap_values(X_shap)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_rank = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda t: -t[1],
    )

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
    plt.tight_layout()
    shap_path = REPORTS_DIR / "phase9_shap_summary.png"
    plt.savefig(shap_path, dpi=100, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    deciles = [r["decile"] for r in lift]
    ae_ratios = [r["ae_ratio"] for r in lift]
    plt.bar(deciles, ae_ratios, color="steelblue")
    plt.axhline(1.0, color="red", linestyle="--", label="perfect calibration")
    plt.xlabel("Decile (1 = highest predicted premium)")
    plt.ylabel("Actual / Expected ratio")
    plt.title("Lift by decile — pure premium")
    plt.legend()
    plt.tight_layout()
    lift_path = REPORTS_DIR / "phase9_lift_chart.png"
    plt.savefig(lift_path, dpi=100, bbox_inches="tight")
    plt.close()

    test_df = test.copy()
    test_df["y_pp"] = y_pp_te
    test_df["pp_pred"] = pp_te
    fair_rows: list[pd.DataFrame] = []
    for group_col in ("DrivAge_bucket", "Area"):
        r = residuals_by_group(test_df, "y_pp", "pp_pred", group_col)
        r = r.rename(columns={group_col: "group"})
        r["group_col"] = group_col
        fair_rows.append(r)
    r = residuals_by_group(test_df, "y_pp", "pp_pred", "Region", top_k=5)
    r = r.rename(columns={"Region": "group"})
    r["group_col"] = "Region"
    fair_rows.append(r)
    fairness = pd.concat(fair_rows, ignore_index=True)

    logger.info("Refitting severity quantile models for test coverage...")
    X_tr_sev = X_tr[m_tr]
    y_tr_sev_log = np.log(raw_sev_tr)
    models_q = fit_quantile_grid(X_tr_sev, y_tr_sev_log, quantiles=(0.1, 0.5, 0.9))
    X_te_sev = X_te[m_te]
    y_te_sev_log = np.log(raw_sev_te)
    lo_sev_log, _, hi_sev_log = predict_interval(models_q, X_te_sev, q_hat=Q_HAT_PHASE8)
    test_cqr_coverage = empirical_coverage(y_te_sev_log, lo_sev_log, hi_sev_log)

    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    p("=" * 90)
    p("PHASE 9 — FINAL EVALUATION ON TEST")
    p("=" * 90)
    p()
    p("Model: LGBM freq × GLM sev × p_has_sev (Phase 6 chosen compound)")
    p(f"Test rows: {len(test):,}")
    p(f"p_has_sev: {p_has_sev:.4f}")
    p(f"q_hat (frozen from Phase 8): {Q_HAT_PHASE8:.4f}")
    p()
    p("=== Global metrics on test ===")
    p(f"Pure premium MAE:          {pp_mae:.2f} EUR")
    p(f"Pure premium AE:           {pp_ae:.4f}")
    p(f"Pure premium Gini:         {pp_gini:.4f}  (0=random, 1=perfect)")
    p(f"Frequency Poisson dev:     {freq_dev:.4f}")
    p(f"Frequency AE:              {freq_ae:.4f}")
    p(f"Severity Gamma dev:        {sev_dev:.4f}")
    p(f"Severity AE:               {sev_ae:.4f}")
    p(f"Conditional CQR coverage:  {test_cqr_coverage:.4f}  (nominal 0.80)")
    p()
    p("=== Lift by decile (pure premium) ===")
    p(f"{'decile':>6s} {'n':>10s} {'actual':>14s} {'predicted':>14s} {'AE':>8s}")
    for r in lift:
        p(
            f"{r['decile']:>6d} {r['n']:>10,} {r['actual']:>14,.0f} "
            f"{r['predicted']:>14,.0f} {r['ae_ratio']:>8.4f}"
        )
    p()
    p("=== SHAP top-10 features (LGBM frequency) ===")
    for i, (name, val) in enumerate(shap_rank[:10], 1):
        p(f"  {i:>2d}. {name:<24s} mean|SHAP| = {val:.4f}")
    p()
    p(f"SHAP summary plot: {shap_path}")
    p(f"Lift chart:        {lift_path}")
    p()
    p("=== Fairness — residuals by group (pure premium) ===")
    p(f"{'group_type':<16s} {'group':>10s} {'n':>10s} {'mean_resid':>14s} {'ae':>8s}")
    for _, row in fairness.iterrows():
        p(
            f"{row['group_col']:<16s} {str(row['group']):>10s} "
            f"{int(row['n']):>10,} {row['mean_residual']:>14.2f} "
            f"{row['ae_ratio']:>8.4f}"
        )
    p()
    p("=== Notes ===")
    p("- Pure premium AE > 1 means model underpredicts on average.")
    p("  At test level, expect similar +10-15% underprediction as val.")
    p("  Documented in Phase 5 (val-tail-shift +7.7%). Not a bug.")
    p("- Gini measures ranking quality (portfolio prioritization),")
    p("  not calibration (measured by AE).")
    p("- Fairness analysis is DESCRIPTIVE, not normative. Groups with")
    p("  high AE or large residuals should be reviewed but are not")
    p("  evidence of discrimination on their own.")
    p()

    OUT = REPORTS_DIR / "phase9_final_evaluation.txt"
    OUT.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", OUT)

    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text("Phase 9 completed. Test split frozen.\n")
    logger.info("Wrote sentinel %s", SENTINEL)


if __name__ == "__main__":
    main()
