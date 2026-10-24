.DEFAULT_GOAL := help

UV := uv
RUFF := uv run ruff
PYTEST := uv run pytest

.PHONY: help install sync lint format format-check test ci data validate processed eda features baselines gbm tune clean

help:  ## Show available targets
	@echo "Available targets:"
	@echo "  install       Install all dependencies (prod + dev)"
	@echo "  sync          Sync dependencies from uv.lock"
	@echo "  lint          Run ruff check on src/ and tests/"
	@echo "  format        Format code with ruff (modifies files)"
	@echo "  format-check  Check formatting without modifying files"
	@echo "  test          Run the full pytest suite"
	@echo "  ci            Full pipeline: lint + format-check + test"
	@echo "  data          Download raw freMTPL2 to data/raw/"
	@echo "  validate      Validate + clean + merge to data/processed/"
	@echo "  processed     Alias for validate"
	@echo "  eda           Print EDA report + save reports/phase3_eda.txt"
	@echo "  features      Build train/val/test features parquets"
	@echo "  baselines     Train + evaluate Dummy and GLM baselines"
	@echo "  gbm           Train + evaluate XGB + LGBM (freq, sev, tweedie)"
	@echo "  tune          Optuna tune LGBM frequency (15 trials, 3-fold CV)"
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

data:
	$(UV) run python -m src.data.download

validate processed:
	$(UV) run python -m src.data.validate

eda:
	$(UV) run python -m scripts.eda_report

features:
	$(UV) run python -m scripts.build_features

baselines:
	$(UV) run python -m scripts.train_baselines

gbm:
	$(UV) run python -m scripts.train_gbm

tune:
	$(UV) run python -m scripts.tune_lgbm

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
