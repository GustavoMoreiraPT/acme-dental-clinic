# ── Stage 1: build ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

# Copy dependency manifests first (Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install production deps only (no dev extras)
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Make sure the venv is on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code and knowledge base
COPY src/ ./src/
COPY KNOWLEDGE_BASE.md ./

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    METRICS_ENABLED=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
