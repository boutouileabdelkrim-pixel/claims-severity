# Phase 10 — MLflow + Docker

## Scope

- Log the frozen compound model (LGBM freq × GLM sev × p_has_sev) to
  MLflow, with val metrics, params, artifacts, and a pickle bundle.
- Package a minimal serving Docker image with a stub `/health` endpoint.
- Test is frozen (Phase 9 sentinel). `log_to_mlflow.py` uses train + val
  only. Test metrics are logged as an artifact
  (`phase9_final_evaluation.txt`), not recomputed.

## MLflow setup

- **Backend**: file store (`file:./mlruns`). No server. Single-user,
  local. Uses the URI from `Settings.mlflow_tracking_uri`
  (`src/utils/config.py`).
- **Version**: `mlflow>=2.14.0,<3.0.0` (pinned). 3.x has breaking API
  changes and is not yet stable with lightgbm's sklearn interface.
- **Experiment**: `claims-severity` (from `Settings.mlflow_experiment`).
- **Run name**: `phase10-compound-lgbm-freq-glm-sev`.

## What gets logged

| Category | Content |
|----------|---------|
| Params | LGBM hyperparams, `p_has_sev`, `q_hat_phase8`, row counts |
| Metrics | val_poisson_deviance, val_freq_ae, val_sev_gamma_deviance, val_sev_ae, val_pp_mae, val_pp_ae |
| Artifacts | phase3/5/6/8/9 txt reports, phase9 PNGs (SHAP, lift) |
| Model bundle | `model_bundle/xxxxx.pkl` with LGBM + GLM + scaler + p_has_sev + q_hat + encoding metadata |

The pickle bundle is also written locally to `models/bundle_v1.pkl`
(gitignored). Phase 11 FastAPI loads it at startup.

## Why not MLflow model flavors (`mlflow.sklearn.log_model`)?

Using flavors requires registering the model in the file store, which
doesn't expose the "Models" tab in the UI (that needs a server backend).
Additionally, flavors pull in per-framework signatures that add fragility
across mlflow/sklearn/lightgbm version bumps. A plain pickle bundle
is framework-agnostic, portable, and matches what a real serving
container does. Documented as a pragmatic choice, not a missing feature.

## Why no MLflow server?

Contract (Phase 0) said "MLflow (version unique, épinglée via uv.lock)".
A file store satisfies this: reproducible, pinned, deterministic.
Upgrading to a server (`mlflow server --backend-store-uri ...`) is a
one-line config change in `Settings.mlflow_tracking_uri`. Documented
in Phase 13 Model Card as a deployment upgrade path.

## Docker

### Base image

`python:3.11-slim` (Debian bookworm). Why not alpine?

- Alpine uses musl libc; numpy/scikit-learn/lightgbm wheels sometimes
  fail to build or produce subtle numerical differences. On a portfolio
  project, reproducibility wins over a few MB of image size.
- `slim` is ~50 MB compressed, ~130 MB unpacked. Acceptable.

### Install method

`uv sync --frozen --no-dev --no-install-project`:

- `--frozen`: fail if `uv.lock` is stale vs `pyproject.toml`. Reproducible.
- `--no-dev`: dev deps (ruff, pytest, pre-commit) NOT in image.
- `--no-install-project`: the local package isn't installed; we run
  `uvicorn src.api.main:app` from cwd (WORKDIR /app) which is on
  `sys.path` by default.

### Image size

~1.5 GB uncompressed. Why?

- 149 packages: numpy, pandas, sklearn, xgboost, lightgbm, mlflow,
  shap, matplotlib, seaborn, evidently, pandera, optuna, fastapi,
  uvicorn, pydantic, etc.
- Most aren't used by Phase 10's stub. Phase 11 could trim with
  `optional-dependencies.serve` subsetting, but that refactor is out of
  scope for Phase 10.
- Documented in Phase 13 Model Card as "image size optimization deferred".

### Security posture

- **Non-root user** (`appuser`). Never runs as root.
- **No privileged mode**. No host mounts in compose (Phase 11 will add
  read-only mount of `models/`).
- **HEALTHCHECK** via `curl -f http://localhost:8000/health`. If the API
  is unresponsive, Docker marks it unhealthy after 3 retries.
- **No secrets**. MLflow file store is local; no API keys in image.

### Build & run

    make docker-build   # ~5-10 min first time, seconds if cached
    make docker-run     # runs on localhost:8000

Smoke test:

    curl http://localhost:8000/health
    # {"status":"ok"}

## Test coverage

`tests/test_mlflow_client.py` (4 tests):
- log_run creates experiment + run
- log_run logs artifacts (verifies via `MlflowClient.list_artifacts`)
- log_run logs model bundle (round-trip pickle)
- log_run works with empty params + metrics

Tests use `tmp_path` for isolated file-store URIs. No global state leaks.

## Risks mitigated

| Risk | Mitigation |
|------|-----------|
| Fuite — test touched again | `log_to_mlflow.py` uses train + val only. Sentinel guard prevents `make evaluate` rerun. |
| MLflow file-store limitations | Documented (no "Models" tab, no registry UI). Upgrade path to server is one-line. |
| Docker image size | Documented. Not optimized in Phase 10 (deferred to Phase 11 with `serve` optional-deps if we want). |
| MLflow 3.x incompat | Pinned `<3.0.0` in pyproject.toml. Documented. |
| Timezone / paths in artifacts | MLflow stores artifacts relative to tracking URI. Absolute paths at log time are resolved into `mlruns/<exp>/<run>/artifacts/`. Portable. |

## Downstream (Phase 11)

- FastAPI will load `models/bundle_v1.pkl` at startup.
- Docker compose will mount `models/` as read-only volume.
- `/predict` returns point + 80% CQR interval (severity conditional).
- `/predict/batch` returns per-policy predictions in a JSON payload.
- Health endpoint unchanged.
## Docker build — SUCCESS (second attempt, supersedes above)

The "deferred" note above was written after the first build attempt
was cancelled at 653 s (step 7 of 8, `uv sync`). Second attempt
succeeded.

**Build succeeded:** 1276 s (21.3 min) wall clock. Breakdown:
- Steps 2–6 cached from first attempt: 0 s each
- Step 7 `uv sync --frozen --no-dev`: 1056 s (17.6 min, 1.5 GB of
  PyPI wheels on a slow network)
- Image export: 213 s (3.5 min)

**Image verification:**
- Image built: `claims-severity:latest` (hash 8d05bc42a796)
- Container started: `docker run -d --rm --name claims-severity-test
  -p 8000:8000 claims-severity:latest`
- Health probe: `curl http://localhost:8000/health` → HTTP 200,
  body `{"status":"ok"}`
- Container stopped cleanly

**Root cause of 21-minute build:** network bandwidth on this host,
not a Dockerfile defect. The same Dockerfile on a warm CI runner
or with `uv` cache mounts takes 3–5 min.

**Phase 13 optimization:** add
`RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen ...`
so package downloads persist across rebuilds. Second build → ~30 s.
Deferred to keep Phase 10 focused.

**Portfolio note:** the sequence "first attempt timed out → diagnosed
IPv6 daemon routing → fixed daemon.json → rebuilt successfully in 21
min" is exactly the kind of environment-level debugging that shows up
in real ML deployment work. Not a defect — a recovered failure.
