# ---- YoFilter AI Eraser backend ----
# CPU image by default (v1). For GPU: build with
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
#                --build-arg ONNXRUNTIME_PKG=onnxruntime-gpu -t yofilter-eraser:gpu .
# and set ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider at runtime.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG ONNXRUNTIME_PKG=onnxruntime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$ONNXRUNTIME_PKG" != "onnxruntime" ]; then \
        pip uninstall -y onnxruntime && pip install --no-cache-dir "$ONNXRUNTIME_PKG"==1.19.2; \
    fi

COPY app ./app
COPY models ./models

# Non-root
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# start-period bumped from 30s -> 60s: the model now loads synchronously
# during FastAPI startup (see app/main.py lifespan) instead of lazily on
# the first request, so the container won't accept ANY connection --
# including this healthcheck -- until loading finishes. 60s covers slower
# CPU-only hosts with margin; once started, requests reuse the same
# session and don't pay this cost again.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
