"""One HTTP surface for both inference services, over a small ``Scorer`` protocol (torch-free).

``build_app`` owns everything a request goes through: Content-Length checks, a bounded body read
with its own timeout, admission, decoding, the caller's ``deadline_ms`` budget and the response
schema. A scorer only turns a 16 kHz waveform into a ``Score`` (``eot.serving.service`` runs
onnxruntime on CPU, ``eot.serving.tensorrt_service`` batches onto one TensorRT worker), so the two
services cannot drift apart in contract.

Endpoints
- ``POST /v1/eot``  body = raw PCM16 LE mono (``?sr=16000``) or a WAV/FLAC file; optional
  ``?agent_text=...`` (previous agent utterance) and ``?threshold=0.5``. Returns ``p_eot``,
  ``p_fvad`` per horizon, ``decision`` (``eot``|``hold``), ``batch_size``, timings split into
  ``decode``, ``features``, ``inference`` and ``total`` ms, the model sha and the package version.
- ``GET /healthz``  version, model sha, admission limits, image source digest and the scorer's own
  runtime fields.

Design notes
- Binary body, not base64 JSON: 8 s of PCM16 is 256 KB; JSON would double it and add parse time.
- The body is read (bounded by ``BODY_READ_TIMEOUT_S``) *before* an admission slot is taken, so a
  stalled client cannot hold capacity. Admission fails fast with HTTP 503 instead of queueing beyond
  the latency budget. ``?deadline_ms=`` lets a caller declare that a stale answer is useless (the
  user resumed speaking) and get 409 quickly; native inference cannot be cancelled, so a timed-out
  request keeps its slot until the scorer finishes.
- This wrapper is the demo surface. In production VAD + ring buffer + the model share a process;
  the network hop only exists here so it can be stress-tested in isolation.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from eot import __version__
from eot.audio import SAMPLE_RATE, WINDOW_SECONDS, load_audio_bytes, resample, to_mono

MAX_INFLIGHT = 8
MAX_BODY_BYTES = 2_000_000
BODY_READ_TIMEOUT_S = 2.0
SOURCE_SHA256_PATH = Path("/app/SOURCE_SHA256")  # written by both Dockerfiles; reported by /healthz when present


@dataclass(frozen=True)
class Score:
    p_eot: float
    p_fvad: dict[str, float]
    features_ms: float
    inference_ms: float
    batch_size: int = 1
    context_used: bool = False


class Scorer(Protocol):
    sha: str

    async def start(self) -> None:
        """Called once from the app's lifespan startup (start workers that need the running loop)."""

    async def score(self, wave: np.ndarray, agent_text: str) -> Score:
        """``wave``: mono float32 at 16 kHz, at most ``WINDOW_SECONDS`` long."""

    def health(self) -> dict:
        """Runtime fields merged into ``/healthz``."""

    async def close(self) -> None:
        """Called from lifespan shutdown after in-flight work has drained."""


def decode_request(body: bytes, raw_sr: int | None) -> np.ndarray:
    """Body -> mono float32 at 16 kHz, decoding only the trailing window so resampling stays bounded."""
    x, in_sr = load_audio_bytes(body, assume_pcm16_sr=raw_sr, max_seconds=WINDOW_SECONDS)
    if not 8_000 <= in_sr <= 192_000:
        raise ValueError(f"sample rate must be between 8000 and 192000 Hz, got {in_sr}")
    x = to_mono(x)
    if len(x) == 0:
        raise ValueError("no audio samples")
    if not np.isfinite(x).all():
        raise ValueError("audio contains non-finite samples")
    return resample(x, in_sr, SAMPLE_RATE)


async def read_limited_body(request: Request, max_body_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_body_bytes:
            raise HTTPException(413, f"request body exceeds {max_body_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _check_content_length(request: Request, max_body_bytes: int) -> None:
    declared = request.headers.get("content-length")
    if declared is None:
        return
    try:
        size = int(declared)
    except ValueError:
        raise HTTPException(400, "invalid Content-Length header") from None
    if size < 0:
        raise HTTPException(400, "invalid negative Content-Length header")
    if size > max_body_bytes:
        raise HTTPException(413, f"request body exceeds {max_body_bytes} bytes")


def _source_sha256() -> str | None:
    return SOURCE_SHA256_PATH.read_text().strip() if SOURCE_SHA256_PATH.is_file() else None


def validate_limits(max_inflight: int, max_body_bytes: int) -> None:
    """Shared by ``build_app`` and the services, which check before loading a model."""
    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be positive")


def build_app(scorer: Scorer, *, title: str, max_inflight: int = MAX_INFLIGHT, max_body_bytes: int = MAX_BODY_BYTES) -> FastAPI:
    validate_limits(max_inflight, max_body_bytes)
    admission = asyncio.Semaphore(max_inflight)
    decoders = ThreadPoolExecutor(max_workers=max_inflight, thread_name_prefix="decode")
    pending: set[asyncio.Task] = set()
    source_sha256 = _source_sha256()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await scorer.start()
        try:
            yield
        finally:
            # Native work cannot be cancelled safely: drain it before the scorer frees its buffers.
            try:
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                await scorer.close()
            finally:
                decoders.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title=title, version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": True,
            "version": __version__,
            "model_sha": scorer.sha,
            "max_inflight": max_inflight,
            "max_body_bytes": max_body_bytes,
            "source_sha256": source_sha256,
            **scorer.health(),
        }

    @app.post("/v1/eot")
    async def eot(
        request: Request,
        sr: int | None = Query(None, ge=8_000, le=192_000, description="if set, body is raw PCM16 LE mono at this rate"),
        agent_text: str = Query(""),
        threshold: float = Query(0.5, ge=0.0, le=1.0),
        deadline_ms: float | None = Query(None, gt=0, description="end-to-end server budget including body read, decoding and inference"),
    ) -> JSONResponse:
        t_start = time.perf_counter()

        def remaining() -> float | None:
            if deadline_ms is None:
                return None
            return max(0.0, deadline_ms / 1000 - (time.perf_counter() - t_start))

        def expired() -> bool:
            return remaining() == 0.0

        _check_content_length(request, max_body_bytes)
        read_budget = BODY_READ_TIMEOUT_S if remaining() is None else min(remaining(), BODY_READ_TIMEOUT_S)
        try:
            body = await asyncio.wait_for(read_limited_body(request, max_body_bytes), timeout=read_budget)
        except TimeoutError:
            if expired():
                raise HTTPException(409, "request deadline exceeded; discard this result") from None
            raise HTTPException(408, f"request body not received within {BODY_READ_TIMEOUT_S:g} s") from None
        if not body:
            raise HTTPException(400, "empty body")
        if admission.locked():
            raise HTTPException(503, "at capacity; retry or fall back to timeout policy")
        await admission.acquire()

        async def process() -> tuple[int, float, Score]:
            t_decode = time.perf_counter()
            try:
                wave = await asyncio.get_running_loop().run_in_executor(decoders, decode_request, body, sr)
            except Exception as e:
                raise HTTPException(400, f"cannot decode audio: {e}") from e
            decode_ms = (time.perf_counter() - t_decode) * 1000
            if expired():
                raise HTTPException(409, "deadline exceeded during decoding")
            return len(wave), decode_ms, await scorer.score(wave, agent_text)

        task = asyncio.create_task(process())
        pending.add(task)
        try:
            n_samples, decode_ms, score = await asyncio.wait_for(asyncio.shield(task), timeout=remaining())
            if expired():
                raise HTTPException(409, "deadline exceeded during inference")
        except TimeoutError:
            raise HTTPException(409, "request deadline exceeded; discard this result") from None
        finally:
            if task.done():
                pending.discard(task)
                admission.release()
            else:
                # A timed-out or disconnected caller cannot cancel a running native forward. Its slot stays
                # occupied until completion so the executor queue cannot grow without bound.
                def finished(done: asyncio.Task) -> None:
                    if not done.cancelled():
                        done.exception()  # mark retrieved; the caller is gone
                    pending.discard(done)
                    admission.release()

                task.add_done_callback(finished)
        total_ms = (time.perf_counter() - t_start) * 1000
        return JSONResponse(
            {
                "p_eot": score.p_eot,
                "decision": "eot" if score.p_eot >= threshold else "hold",
                "threshold": threshold,
                "p_fvad": score.p_fvad,
                "audio_seconds": n_samples / SAMPLE_RATE,
                "context_used": score.context_used,
                "batch_size": score.batch_size,
                "timings_ms": {
                    "decode": round(decode_ms, 2),
                    "features": round(score.features_ms, 2),
                    "inference": round(score.inference_ms, 2),
                    "total": round(total_ms, 2),
                },
                "model_sha": scorer.sha,
                "version": __version__,
            }
        )

    return app


# ---- entry-point helpers shared by both services --------------------------------------------------------------------
def service_parser(description: str, model_flag: str, model_env: str) -> argparse.ArgumentParser:
    """Arguments common to ``eot-serve`` and ``eot-serve-gpu``; the model path falls back to ``model_env``."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(model_flag, default=None, help=f"model path (default: ${model_env})")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("EOT_THREADS", "2")), help="CPU threads (default: $EOT_THREADS or 2)")
    ap.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT, help="admitted requests before HTTP 503")
    ap.add_argument(
        "--max-body-bytes",
        type=int,
        default=int(os.environ.get("EOT_MAX_BODY_BYTES", MAX_BODY_BYTES)),
        help="request body limit (default: $EOT_MAX_BODY_BYTES or 2000000)",
    )
    return ap


def resolve_model_path(explicit: str | None, env_var: str, flag: str) -> str:
    path = explicit or os.environ.get(env_var)
    if not path:
        raise RuntimeError(f"set {env_var} or pass {flag}")
    return path


def run_service(app: FastAPI, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
