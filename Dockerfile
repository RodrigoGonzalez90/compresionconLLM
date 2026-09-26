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
# llama-cpp-python. LLAMA_BUILD=native (por defecto) usa el wheel precompilado: rápido, pero los
# logits dependen de la arquitectura. LLAMA_BUILD=generic compila el backend de CPU en C escalar,
# sin AVX/NEON ni FMA fusionado, para que x86 y ARM produzcan los mismos logits y intercambien
# mensajes con compresión completa. Medido en x86: ~8.6x más lento y compila unos minutos.
# Ambos nodos deben usar el mismo modo:  LLAMA_BUILD=generic ./run_docker.sh build
ARG LLAMA_BUILD=native
ARG LLAMA_VERSION=0.3.35
# El toolchain declara un procesador desconocido: ggml usa entonces sus kernels genéricos en C.
RUN if [ "$LLAMA_BUILD" = "generic" ]; then \
        printf 'set(CMAKE_SYSTEM_NAME Linux)\nset(CMAKE_SYSTEM_PROCESSOR generic)\n' > /tmp/generic.cmake \
        && CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=/tmp/generic.cmake -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DCMAKE_C_FLAGS=-ffp-contract=off -DCMAKE_CXX_FLAGS=-ffp-contract=off" \
        pip install --no-binary llama-cpp-python "llama-cpp-python==${LLAMA_VERSION}"; \
    else \
        pip install "llama-cpp-python==${LLAMA_VERSION}" \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu; \
    fi \
    && pip install -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY web/ ./web/

RUN mkdir -p .cache

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/estado || exit 1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
