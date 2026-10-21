"""Smoke test to validate the CI pipeline before Phase 1."""

import sys

import src


def test_python_version_is_311() -> None:
    """CI must run on Python 3.11 (see pyproject requires-python)."""
    assert sys.version_info[:2] == (3, 11)


def test_src_package_is_importable() -> None:
    """Ensure pythonpath=... from pyproject exposes the src package."""
    assert src.__version__ == "0.1.0"
