# CPU inference image: onnxruntime + FastAPI, no torch (service `whisper-cpu` in compose.inference.yaml).
#   docker compose -f compose.inference.yaml up -d --build whisper-cpu
# Standalone:
#   docker build -t eot-serve .
#   docker run --rm -p 8000:8000 -v $PWD/artifacts/deployment/whisper-cpu:/models:ro -e EOT_ONNX=/models/eot.onnx eot-serve
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 EOT_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /usr/local/bin/uv

WORKDIR /app
COPY requirements/serve.txt requirements/serve.lock ./requirements/
# Serving needs neither torch nor a downloaded Whisper model; WhisperFeatureExtractor is pure numpy.
RUN uv pip install --system --no-cache --require-hashes -r requirements/serve.lock
COPY pyproject.toml README.md ./
COPY src ./src
# Reported by /healthz as source_sha256: one digest over everything that defines the image's code.
RUN find pyproject.toml requirements src -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d ' ' -f 1 > /app/SOURCE_SHA256
RUN uv pip install --system --no-cache --no-deps -e . \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin eot
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/healthz').status_code==200 else 1)"
CMD ["eot-serve", "--host", "0.0.0.0", "--port", "8000"]
