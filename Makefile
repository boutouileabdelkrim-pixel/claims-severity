.DEFAULT_GOAL := help

UV := uv
RUFF := uv run ruff
PYTEST := uv run pytest

.PHONY: help install sync lint format format-check test ci clean

help:  ## Show available targets
	@echo "Available targets:"
	@echo "  install       Install all dependencies (prod + dev)"
	@echo "  sync          Sync dependencies from uv.lock"
	@echo "  lint          Run ruff check on src/ and tests/"
	@echo "  format        Format code with ruff (modifies files)"
	@echo "  format-check  Check formatting without modifying files"
	@echo "  test          Run the full pytest suite"
	@echo "  ci            Full pipeline: lint + format-check + test"
	@echo "  clean         Remove caches and build artifacts"

install: sync

sync:
	$(UV) sync --extra dev

lint:
	$(RUFF) check src tests

format:
	$(RUFF) format src tests

format-check:
	$(RUFF) format --check src tests

test:
	$(PYTEST) tests/ -v

ci: lint format-check test

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
