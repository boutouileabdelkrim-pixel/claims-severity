"""Centralized filesystem paths for the project.

All paths are derived from the repository root, detected relative to this
file's location. This makes the project portable: same code works locally,
in CI, and inside Docker, without modification.

Never hardcode a path like `/home/user/...` anywhere else in the codebase.
"""

from __future__ import annotations

from pathlib import Path

# Layout:
#   <repo>/src/utils/paths.py
#   parents[0] = src/utils
#   parents[1] = src
#   parents[2] = <repo root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Top-level directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURES_DIR: Path = DATA_DIR / "features"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
DOCKER_DIR: Path = PROJECT_ROOT / "docker"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"

# Directories that must exist for the pipeline to run
REQUIRED_DIRS: tuple[Path, ...] = (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)


def ensure_dirs() -> None:
    """Create all required directories if they don't exist (idempotent)."""
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
