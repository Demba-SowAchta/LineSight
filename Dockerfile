# ---------------------------------------------------------------------------
# Multi-stage Dockerfile for the Industrial Vision Platform.
# Stage 1 installs dependencies; stage 2 is the slim runtime image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

# System libs OpenCV/Pillow need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching).
# Use requirements-core.txt for a small image without PyTorch; swap to
# requirements.txt to bake in the training/inference stack.
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Copy the source.
COPY src ./src
COPY scripts ./scripts

# Non-root user (container security best practice).
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV IVP_MODEL_BACKEND=auto \
    IVP_DB_PATH=/app/artifacts/ivp.db \
    IVP_IMAGE_STORE=/app/artifacts/images \
    IVP_MODELS_DIR=/app/artifacts/models

EXPOSE 8000 8501

# Default: run the API. Override the command to run the Streamlit app instead.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
