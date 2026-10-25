"""MLflow logging wrapper (Phase 10).

Two entry points:
- log_run(): log params, metrics, artifacts, and an optional model bundle.
- _flatten_params(): normalize dict values to MLflow-compatible types.

MLflow backend: file store (`file:./mlruns`), no server. Reproducible from
`uv.lock` (pinned mlflow>=2.14,<3.0). Server upgrade path documented in
reports/phase10_mlflow.md.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow

MLFLOW_COMPATIBLE = (str, int, float, bool)


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    """Coerce non-primitive values to string (MLflow only accepts primitives)."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, MLFLOW_COMPATIBLE):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def log_run(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[str | Path],
    model_bundle: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a single MLflow run. Returns the run_id.

    - tracking_uri: "file:./mlruns" or a server URI
    - experiment_name: created if missing
    - artifact_paths: list of existing paths to log as artifacts
    - model_bundle: dict of objects; pickled and logged under "model_bundle/"
    """
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = exp.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        if params:
            mlflow.log_params(_flatten_params(params))
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if tags:
            mlflow.set_tags(tags)
        for p in artifact_paths:
            p = Path(p)
            if p.exists():
                mlflow.log_artifact(str(p))
        if model_bundle is not None:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(model_bundle, f)
                bundle_path = Path(f.name)
            try:
                mlflow.log_artifact(str(bundle_path), artifact_path="model_bundle")
            finally:
                bundle_path.unlink(missing_ok=True)
        return run.info.run_id
