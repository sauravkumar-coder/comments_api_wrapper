# ============================================================================
#  RM Remark Quality API — Dockerfile
# ============================================================================
#  Multi-stage build for a lean production image.
#
#  Build:   docker build -t rm-quality-api .
#  Run:     docker run -p 8000:8000 --env-file .env rm-quality-api
# ============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY .env.example ./.env

# Pre-download the embedding model at build time (avoids first-request delay)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-large-en-v1.5')" 2>/dev/null || true

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Expose API port (Render sets the PORT env var)
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 10000)}/health')" || exit 1

# Start the service
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
