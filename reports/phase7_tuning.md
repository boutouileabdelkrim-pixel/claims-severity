# Phase 7 — Optuna tuning (LightGBM frequency)

## Scope

Tune **LightGBM frequency** with Optuna. Nothing else.

- Severity (GLM): no tuning. Default stable (Phase 6 AE 1.09). No gain expected.
- Tweedie: no tuning. Broken on 4M EUR outlier (AE 1.40). No tuning will fix it.
- XGBoost frequency: no tuning. Phase 6 showed LGBM ≈ XGB (0.1515 vs 0.1518).

## Budget

- 15 trials, 3-fold CV
- Hard timeout: 25 min
- Sampler: TPE, 5 random startup trials
- Compute: CPU only, ~500 MB RAM peak. **No Kaggle needed.**

If 15 trials is insufficient (best trial is the last one), we can extend
to 30 trials locally (~40 min) or shift to Kaggle T4 with 50 trials.
Decision after seeing trial history.

## Search space (deliberately narrow)

| Param | Values |
|-------|--------|
| n_estimators | {150, 300} |
| learning_rate | [0.03, 0.10] log |
| num_leaves | {31, 63} |
| min_child_samples | {50, 100} |
| subsample | [0.7, 1.0] |
| colsample_bytree | [0.7, 1.0] |
| reg_alpha | [0.0, 0.5] |
| reg_lambda | [0.0, 0.5] |

A wide space with 15 trials is noise, not signal. Narrow space produces
distinguishable trials.

## CV strategy

- 3-fold KFold on train, shuffle, seed=42
- Rate target, sample_weight=Exposure (Phase 5 convention)
- Poisson deviance on count (rate × exposure), no clip beyond ≥ 0
- Mean across folds is the Optuna objective
- Std across folds is logged but not used in the objective (single metric)

## Final evaluation

- Best params → fit once on full train → eval on val (single shot)
- Compared vs Phase 6 default LGBM freq (val deviance 0.1515)
- Decision rule:
  - Improvement ≥ 2% → save params to `phase7_best_params.json`, use in Phase 8+
  - Improvement < 2% → keep Phase 6 defaults, document

## Risks

| Risk | Mitigation |
|------|-----------|
| CV overfits (TPE chases noise) | 5 random startup trials; check best trial is not the last |
| Val leak | Val touched once, only after best params chosen |
| Time overrun | Hard timeout 25 min; measured elapsed in report |

## Results

Exact numbers in `reports/phase7_tuning.txt`.

If `reports/phase7_best_params.json` exists → tuning won, params used downstream.
If absent → Phase 6 default LGBM params stay. Documented.

## Deferred

- Severity tuning: skipped (GLM stable, no gain expected).
- Tweedie tuning: skipped (objective broken by 4M outlier).
- Native LightGBM categoricals: deferred to Phase 8+ if relevant.
- Cross-version stability of best params: not tested (single run).
