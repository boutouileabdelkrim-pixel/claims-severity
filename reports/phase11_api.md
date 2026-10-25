# Phase 11 — FastAPI /predict and /predict/batch

## Scope

Serving endpoints for the frozen compound model:
- `GET /health` — liveness (works without bundle)
- `GET /info` — model metadata
- `POST /predict` — single policy → point + claim probability + severity interval
- `POST /predict/batch` — 1..1000 policies

Test split remains frozen (Phase 9 sentinel). No model training here.

## Bundle extension (why `extend_bundle.py`)

Phase 10 saved the bundle with `lgbm_freq_model`, `glm_sev_model`,
`p_has_sev`, `q_hat`, `scaler`, `dummy_cols`, `feature_names` — but NOT
the Phase 8 quantile models. `/predict` needs them for the CQR interval
(contract: "prédiction + intervalle").

`scripts/extend_bundle.py` fits 3 LightGBM quantile models (0.1, 0.5, 0.9)
on train claims (17,417 rows, ~15 s), adds them under `quantile_models`,
and saves the bundle in place. Idempotent — running twice is a no-op.

Alternative considered and rejected: retrain quantile models on every
startup. Too slow (~15 s), and tests would need the full training
pipeline. Frozen bundle is the right pattern for serving.

## Response format

    POST /predict
    {
      "VehPower": 5, "VehAge": 2, "DrivAge": 45, "BonusMalus": 50,
      "VehBrand": "B12", "VehGas": "Regular", "Area": "D",
      "Density": 1217, "Region": "R82", "Exposure": 0.8
    }
    →
    {
      "pure_premium": 142.37,
      "claim_probability": 0.0768,
      "severity_median": 1165.04,
      "severity_lo_80": 512.31,
      "severity_hi_80": 3142.87,
      "interval_source": "CQR severity (Phase 8), q_hat=0.2714, conditional on claim"
    }

**Interval semantics**: `severity_lo_80` and `severity_hi_80` are the
80% CQR interval **conditional on a claim occurring**. For a policy with
`claim_probability < 1`, the marginal interval includes the zero-claim
mass and is NOT symmetric around `pure_premium`. Documented in Phase 13
Model Card.

## Validation

- `extra="forbid"` on PolicyInput — unknown fields → 422 (catches typos
  in client code before they silently leak through).
- Ranges: `BonusMalus` in [50, 300], `Exposure` in (0, 2.5],
  `DrivAge` in [18, 100], `VehPower` in [1, 20]. Aligns with training
  data ranges from Phase 3 EDA.
- `Literal` on `VehGas` and `Area` — enforced at Pydantic layer.

## Bundle-missing behavior

If `models/bundle_v1.pkl` is absent (fresh clone before `make mlflow`):
- `/health` → 200 (liveness OK, app is up)
- `/info` → 200 with `bundle_loaded: false`
- `/predict` and `/predict/batch` → 503 with clear message

This is standard orchestration-friendly behavior: k8s readiness probes
can check `/info` for `bundle_loaded`, while liveness checks `/health`.

## Tests (9)

- test_health_ok — health returns 200 `{"status":"ok"}` with or without bundle
- test_info_with_bundle — reports `bundle_loaded: true`
- test_predict_returns_predictions — happy path, intervals ordered
- test_predict_rejects_invalid_bonus_malus — 422
- test_predict_rejects_invalid_gas — 422 (Literal mismatch)
- test_predict_rejects_extra_fields — 422 (extra="forbid")
- test_batch_5_policies — 200, 5 predictions
- test_batch_rejects_empty — 422 (min_length=1)
- test_predict_503_when_bundle_missing — 503 with clear detail

Tests use `FakeBundle` (mock LGBM, GLM, quantile models). No I/O, no
real model load, no test.parquet touch.

## Design decisions

**Why `lifespan` and not startup event?** FastAPI deprecated `@app.on_event`
in favor of `lifespan` context managers. Cleaner teardown, matches the
current FastAPI documentation.

**Why not async endpoints?** The underlying sklearn/lightgbm `predict`
calls are CPU-bound and synchronous. Making handlers `async def` would
require `await asyncio.to_thread(...)` for real concurrency. For a
single-worker deployment, sync is fine and simpler. Documented as a
Phase 13 optimization if throughput matters.

**Why 1000 max on batch?** LGBM predict scales linearly; 1000 rows at
~50k predictions/sec = 20 ms. Larger batches (10k+) risk slow responses
under load. 1000 is a reasonable default; raise in Phase 13 if benchmarks
justify it.

**Why return `claim_probability` and not `claim_rate`?** Users (pricing
systems, underwriters) reason in probabilities. `1 - exp(-rate*exposure)`
is the Poisson probability of at least one claim. For typical rates
< 0.15, this is close to `rate*exposure` anyway.

## Running locally

    make mlflow                                  # ensure bundle_v1.pkl exists
    uv run python -m scripts.extend_bundle       # add quantile models
    uv run uvicorn src.api.main:app --reload     # serve on :8000
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict \
         -H 'Content-Type: application/json' \
         -d '{"VehPower":5,"VehAge":2,"DrivAge":45,"BonusMalus":50,
              "VehBrand":"B12","VehGas":"Regular","Area":"D",
              "Density":1217,"Region":"R82","Exposure":0.8}'

## Limitations

- Batch endpoint is synchronous. Long requests block the worker. For
  production, deploy with `uvicorn --workers 4` or behind a queue. Phase
  13 Model Card documents this.
- Interval is conditional on a claim. Marginal interval requires
  compounding with claim_probability — see Phase 9 lift by decile.
- No authentication. This is a demo API, not a production service. Phase
  13 Model Card documents the gap.
- Model reloaded only at startup. Live retraining is out of scope.
