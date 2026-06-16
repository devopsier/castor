# =============================================================================
# Castor — Multi-Stage Dockerfile
# Stage 1: Dependency resolver using uv
# Stage 2: Lean production runtime (no build tools)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Install uv — the fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy dependency manifests first for better layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install project dependencies into an isolated virtual environment
RUN uv sync --no-dev --frozen

# ---------------------------------------------------------------------------
# Stage 2 — Production Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="castor"
LABEL org.opencontainers.image.description="Predictive Pod/Cluster Steering for Kubernetes"
LABEL org.opencontainers.image.source="https://github.com/your-org/castor"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Create a non-root user for security
RUN groupadd --system castor && useradd --system --gid castor castor

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /build/.venv /app/.venv

# Copy application source and default configuration
COPY src/ ./src/
COPY config.toml ./config.toml

# Create directories for model artefact persistence
RUN mkdir -p ./artifacts/models && chown -R castor:castor /app

USER castor

# Ensure the uv-managed venv is on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV CASTOR_CONFIG_PATH="/app/config.toml"

EXPOSE 8080

# Health-check probes the /healthz endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')"

CMD ["python", "-m", "castor.main"]
