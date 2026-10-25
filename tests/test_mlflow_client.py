"""Unit tests for src.tracking.mlflow_client (Phase 10).

Uses tmp_path for isolated file-store tracking URIs. Each test gets its
own MLflow backend, so global state does not leak between tests.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow

from src.tracking.mlflow_client import log_run


def _uri(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'mlruns'}"


def test_log_run_creates_experiment_and_run(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    run_id = log_run(
        tracking_uri=uri,
        experiment_name="test_exp_1",
        run_name="run_1",
        params={"lr": 0.05, "n_est": 100},
        metrics={"acc": 0.5},
        artifact_paths=[],
    )
    assert isinstance(run_id, str) and len(run_id) > 0

    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("test_exp_1")
    assert exp is not None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1


def test_log_run_logs_artifacts(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    artifact = tmp_path / "note.txt"
    artifact.write_text("hello phase 10\n")

    run_id = log_run(
        tracking_uri=uri,
        experiment_name="test_exp_2",
        run_name="run_2",
        params={},
        metrics={},
        artifact_paths=[artifact],
    )

    client = mlflow.MlflowClient(tracking_uri=uri)
    artifacts = client.list_artifacts(run_id)
    names = [a.path for a in artifacts]
    assert "note.txt" in names


def test_log_run_logs_model_bundle(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    bundle = {"value": 42, "name": "phase10"}
    run_id = log_run(
        tracking_uri=uri,
        experiment_name="test_exp_3",
        run_name="run_3",
        params={},
        metrics={},
        artifact_paths=[],
        model_bundle=bundle,
    )

    client = mlflow.MlflowClient(tracking_uri=uri)
    bundle_files = client.list_artifacts(run_id, "model_bundle")
    assert len(bundle_files) >= 1

    # Round-trip check: download and unpickle
    local_path = client.download_artifacts(run_id, bundle_files[0].path)
    with Path(local_path).open("rb") as f:
        loaded = pickle.load(f)
    assert loaded == bundle


def test_log_run_with_empty_params_and_metrics(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    run_id = log_run(
        tracking_uri=uri,
        experiment_name="test_exp_4",
        run_name="run_4",
        params={},
        metrics={},
        artifact_paths=[],
    )
    assert isinstance(run_id, str) and len(run_id) > 0
