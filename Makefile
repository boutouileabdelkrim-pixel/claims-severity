.DEFAULT_GOAL := help

UV := uv
RUFF := uv run ruff
PYTEST := uv run pytest
DOCKER_IMAGE := claims-severity
DOCKER_TAG := latest

.PHONY: help install sync lint format format-check test ci data validate processed eda features baselines gbm tune uncertainty evaluate mlflow docker-build docker-run monitor clean

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
	@echo "  uncertainty   Quantile + CQR intervals, coverage verification"
	@echo "  evaluate      Final evaluation on test (touches test once)"
	@echo "  mlflow        Log the frozen compound model + artifacts to MLflow"
	@echo "  docker-build  Build Docker image $(DOCKER_IMAGE):$(DOCKER_TAG)"
	@echo "  docker-run    Run Docker image on port 8000"
	@echo "  monitor       PSI + KS drift report (train vs test)"
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

uncertainty:
	$(UV) run python -m scripts.quantify_uncertainty

evaluate:
	$(UV) run python -m scripts.evaluate_final

mlflow:
	$(UV) run python -m scripts.log_to_mlflow

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run --rm -p 8000:8000 --name $(DOCKER_IMAGE) $(DOCKER_IMAGE):$(DOCKER_TAG)

monitor:
	$(UV) run python -m scripts.monitor_drift

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
