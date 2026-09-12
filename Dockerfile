FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NODE_ID=nodo-1 \
    PEER_URL="" \
    BACKEND=ollama \
    MODELO="phi3:mini" \
    PORT=8000 \
    HF_HOME=/app/.cache/huggingface \
    OLLAMA_HOST=http://ollama:11434

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Instalar llama-cpp-python desde wheels precompilados (CPU) para evitar compilación larga
RUN pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    && pip install -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY web/ ./web/

RUN mkdir -p .cache

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/estado || exit 1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
