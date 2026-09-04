# Inference image: onnxruntime + FastAPI, CPU only, no torch.
# Build:  docker build -t eot-serve .
# Run:    docker run --rm -p 8000:8000 -v $PWD/artifacts:/models -e EOT_ONNX=/models/eot.onnx eot-serve
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 EOT_THREADS=2 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /usr/local/bin/uv

WORKDIR /app
COPY requirements/serve.txt requirements/serve.lock ./requirements/
COPY pyproject.toml ./
COPY src ./src
# Serving needs neither torch nor a downloaded Whisper model; WhisperFeatureExtractor is pure numpy.
RUN uv pip install --system --no-cache -r requirements/serve.lock \
    && uv pip install --no-cache --no-deps -e .
RUN find /app/pyproject.toml /app/requirements /app/src -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | cut -d ' ' -f 1 > /app/SOURCE_SHA256

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin eot
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/healthz').status_code==200 else 1)"
CMD ["eot-serve", "--host", "0.0.0.0", "--port", "8000"]
