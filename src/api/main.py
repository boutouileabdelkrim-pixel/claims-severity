"""FastAPI stub (Phase 10). Full /predict endpoints in Phase 11.

Only /health is exposed here so the Docker healthcheck has a target.
Phase 11 will add /predict, /predict/batch, /info, loading the model
bundle from models/bundle_v1.pkl.
"""

from fastapi import FastAPI

app = FastAPI(title="claims-severity", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Liveness probe. Returns 200 if the API is up."""
    return {"status": "ok"}
