"""Unit tests for src.utils.paths (offline, no real data I/O)."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import (
    FEATURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    REPORTS_DIR,
)


def test_project_root_is_absolute_and_exists() -> None:
    """PROJECT_ROOT must be absolute and exist (it's the repo root)."""
    assert PROJECT_ROOT.is_absolute()
    assert PROJECT_ROOT.exists()


def test_all_dirs_are_descendants_of_project_root() -> None:
    """Every exported directory must live under PROJECT_ROOT."""
    for d in (RAW_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR):
        assert PROJECT_ROOT in d.parents or d == PROJECT_ROOT, f"{d} is not under {PROJECT_ROOT}"


def test_ensure_dirs_creates_missing_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    """ensure_dirs must create REQUIRED_DIRS and be safe to call twice."""
    import src.utils.paths as paths_mod

    fake_dirs = (tmp_path / "a", tmp_path / "b" / "c")
    monkeypatch.setattr(paths_mod, "REQUIRED_DIRS", fake_dirs)

    for d in fake_dirs:
        assert not d.exists()

    paths_mod.ensure_dirs()
    paths_mod.ensure_dirs()  # idempotent

    for d in fake_dirs:
        assert d.exists()
