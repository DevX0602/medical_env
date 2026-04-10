# ── HospitalSchedulingEnv — Production Dockerfile ─────────────────────────
# Compatible with Hugging Face Spaces (Docker SDK)
# Tagged: openenv
#
# Build:  docker build -t hospital-env .
# Run:    docker run -p 7860:7860 hospital-env
# Inference:
#   docker run -e API_BASE_URL=... -e MODEL_NAME=... -e HF_TOKEN=... \
#              hospital-env python inference.py

FROM python:3.11-slim

# Labels for HF Spaces
LABEL org.opencontainers.image.title="HospitalSchedulingEnv"
LABEL org.opencontainers.image.description="OpenEnv hospital scheduling RL environment"
LABEL tags="openenv"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

# Health check — HF Space validator pings /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default: launch FastAPI server (serves /reset, /step, /state, /health)
# Override with: docker run hospital-env python inference.py
CMD ["python", "server.py"]