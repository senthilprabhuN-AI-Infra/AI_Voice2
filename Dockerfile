# ════════════════════════════════════════════════════════════
# Stage 1: Dependency Builder
# ════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /app

# Install OS build tools required by Azure Speech SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ════════════════════════════════════════════════════════════
# Stage 2: Minimal Production Runtime (non-root)
# ════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runner

WORKDIR /app

# Runtime libraries required by Azure Speech SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source and static assets
COPY app.py           .
COPY orchestrator.py  .
COPY extractor_service.py .
COPY voice_profiler.py .
COPY static/          ./static/

# Environment defaults (secrets injected at runtime via --env-file)
ENV PATH="/opt/venv/bin:$PATH"
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# ── Security: run as non-root user ───────────────────────
RUN useradd -u 8888 -m appuser && chown -R appuser:appuser /app
USER appuser

# ── Health Check ─────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD ["python", "app.py"]
