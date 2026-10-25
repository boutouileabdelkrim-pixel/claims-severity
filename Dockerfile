# Phase 10 — minimal serving image for claims-severity.
# - python:3.11-slim base (~50 MB)
# - uv installs dependencies from uv.lock (reproducible)
# - non-root user (appuser)
# - healthcheck via curl (installed explicitly)
# - uvicorn serves src.api.main:app (stub /health in Phase 10)

FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /uvx /bin/

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
