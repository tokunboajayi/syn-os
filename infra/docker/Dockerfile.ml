# Syn OS ML Service
FROM python:3.11-slim-bookworm AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ml/pyproject.toml ./
RUN pip install --no-cache-dir -e ".[full]" 2>/dev/null || pip install --no-cache-dir torch numpy scikit-learn xgboost pandas optuna fastapi uvicorn pydantic tqdm loguru

# Copy source code
COPY ml/synos_ml ./synos_ml

# Create non-root user
RUN useradd -m -u 1000 synos
RUN chown -R synos:synos /app
USER synos

# Model directory
ENV MODEL_DIR=/app/models
RUN mkdir -p $MODEL_DIR

# Expose gRPC and REST ports
EXPOSE 50052 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "synos_ml.serving.server"]
