"""Project-wide settings, loaded once and frozen.

Usage:
    from src.utils.config import SETTINGS
    SETTINGS.seed

Override for tests or experiments:
    Settings(seed=123, n_splits=10)

Override at process start (no code change needed):
    CLAIMS_SEED=123 MLFLOW_TRACKING_URI=file:/tmp/mlruns make test
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, dataclass, field  # noqa: F401
from pathlib import Path

from src.utils.paths import MLRUNS_DIR, PROJECT_ROOT


def _default_mlflow_uri() -> str:
    """Default MLflow tracking URI: local file store inside the repo."""
    return f"file:{MLRUNS_DIR}"


@dataclass(frozen=True)
class Settings:
    """Immutable project configuration.

    All defaults are production-safe: no secrets, no absolute paths,
    local-only tracking store.
    """

    # Reproducibility
    seed: int = 42

    # Data splits (must sum to 1.0)
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15

    # Cross-validation
    n_splits: int = 5

    # MLflow (file store by default; override via MLFLOW_TRACKING_URI env var)
    mlflow_tracking_uri: str = field(default_factory=_default_mlflow_uri)
    mlflow_experiment: str = "claims-severity"

    # Paths (informational; canonical paths live in src.utils.paths)
    project_root: Path = field(default=PROJECT_ROOT)

    def __post_init__(self) -> None:
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_size + val_size + test_size must sum to 1.0, got {total}")
        if not (0 < self.train_size < 1):
            raise ValueError("train_size must be in (0, 1)")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (0 <= self.seed < 2**32):
            raise ValueError("seed must be in [0, 2**32)")


def _load_settings_from_env() -> Settings:
    """Build Settings, honoring environment variable overrides."""
    kwargs: dict = {}
    if "CLAIMS_SEED" in os.environ:
        kwargs["seed"] = int(os.environ["CLAIMS_SEED"])
    if "MLFLOW_TRACKING_URI" in os.environ:
        kwargs["mlflow_tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    return Settings(**kwargs)


# Module-level singleton — import this, don't instantiate Settings yourself
SETTINGS: Settings = _load_settings_from_env()
