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

# Clean up pycache to save space
RUN find /usr/local -type d -name __pycache__ -exec rm -rf {} +

# Copy application code
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY .env.example ./.env

# Removed local embedding model download to use Hugging Face Inference API

# Non-root user for security (Hugging Face requires UID 1000)
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose API port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 7860)}/health')" || exit 1

# Start the service
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
