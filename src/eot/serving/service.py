"""FastAPI inference service over onnxruntime.

    EOT_ONNX=artifacts/eot.onnx uv run eot-serve --host 0.0.0.0 --port 8000 --threads 2

Endpoints
- ``POST /v1/eot``  body = raw PCM16 LE mono (``?sr=16000``) or a WAV/FLAC file; optional
  ``?agent_text=...`` (previous agent utterance) and ``?threshold=0.5``.
  Returns ``p_eot``, ``p_fvad`` per horizon, ``decision`` (``eot``|``hold``), timings split into
  ``decode_ms``, ``features_ms``, ``inference_ms``, ``total_ms`` and the model sha.
- ``GET /healthz``  warm-up state, version, sha, thread settings.

Design notes
- Binary body, not base64 JSON: 8 s of PCM16 is 256 KB; JSON would double it and add parse time.
- One warmed session, fixed intra/inter-op threads; the executor is bounded so we fail fast
  (HTTP 503) instead of queueing beyond the latency budget. ``?deadline_ms=`` lets a caller
  declare that a stale answer is useless (the user resumed speaking) and get 409 quickly.
- This wrapper is the demo surface. In production VAD + ring buffer + this model share a
  process; the network hop only exists here so it can be stress-tested in isolation.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from eot import __version__
from eot.audio import SAMPLE_RATE, WINDOW_SECONDS, load_audio_bytes, log_mel, resample, to_mono
from eot.context import CTX_LEN, hash_context
from eot.io import sha256_file
from eot.onnx import cpu_session, input_names, model_feed


class Engine:
    def __init__(self, onnx_path: str, threads: int = 2):
        self.sess = cpu_session(onnx_path, threads)
        self.input_names = input_names(self.sess)
        digest = sha256_file(Path(onnx_path))
        self.sha = digest[:16]
        meta = Path(onnx_path).with_suffix(".json")
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        if self.meta.get("accepted") is False:
            raise ValueError("model artifact failed export validation")
        recorded_sha = self.meta.get("sha256")
        if recorded_sha and recorded_sha != digest:
            raise ValueError("model checksum does not match its metadata")
        self.use_context = bool(self.meta.get("use_context", False))
        self.use_fvad = bool(self.meta.get("use_fvad", True))
        self.normalize_audio = bool(self.meta.get("normalize_audio", False))
        self.horizons = self.meta.get("horizons", [0.24, 0.64, 1.2, 2.0])
        self.threads = threads
        # Warm up the *whole* path (feature extractor construction + session) so the first real
        # request does not pay a cold start.
        for _ in range(3):
            self.infer(np.zeros(SAMPLE_RATE * 8, np.float32), "")

    def infer(self, x16k: np.ndarray, agent_text: str) -> tuple[float, list[float], float, float]:
        t0 = time.perf_counter()
        feats = log_mel(x16k, normalize=self.normalize_audio)[None]
        t1 = time.perf_counter()
        ctx = hash_context(agent_text)[None] if (self.use_context and agent_text) else np.zeros((1, CTX_LEN), np.int64)
        p, f = self.sess.run(None, model_feed(self.input_names, feats, ctx))
        t2 = time.perf_counter()
        return float(p[0]), [float(v) for v in f[0]], (t1 - t0) * 1000, (t2 - t1) * 1000


def _decode_request(body: bytes, raw_sr: int | None) -> np.ndarray:
    """Decode and bound work before resampling to prevent pathological expansion."""
    x, in_sr = load_audio_bytes(
        body,
        assume_pcm16_sr=raw_sr,
        max_seconds=WINDOW_SECONDS,
    )
    if not 8_000 <= in_sr <= 192_000:
        raise ValueError(f"sample rate must be between 8000 and 192000 Hz, got {in_sr}")
    x = to_mono(x)
    if not np.isfinite(x).all():
        raise ValueError("audio contains non-finite samples")
    max_input_samples = int(round(WINDOW_SECONDS * in_sr))
    if len(x) > max_input_samples:
        x = x[-max_input_samples:]
    return resample(x, in_sr, SAMPLE_RATE)


async def _read_limited_body(request: Request, max_body_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_body_bytes:
            raise HTTPException(413, f"request body exceeds {max_body_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    onnx_path: str | None = None,
    threads: int = 2,
    max_inflight: int = 8,
    max_body_bytes: int = 2_000_000,
) -> FastAPI:
    onnx_path = onnx_path or os.environ.get("EOT_ONNX")
    if not onnx_path:
        raise RuntimeError("set EOT_ONNX or pass --onnx")
    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be positive")
    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    engine = Engine(onnx_path, threads)
    pool = ThreadPoolExecutor(max_workers=max_inflight)
    sem = asyncio.Semaphore(max_inflight)
    @asynccontextmanager
    async def lifespan(_app):
        yield
        pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="eot", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "version": __version__, "model_sha": engine.sha, "use_context": engine.use_context,
                "threads": engine.threads, "max_inflight": max_inflight, "max_body_bytes": max_body_bytes}

    @app.post("/v1/eot")
    async def eot(
        request: Request,
        sr: int | None = Query(
            None,
            ge=8_000,
            le=192_000,
            description="if set, body is raw PCM16 LE mono at this rate",
        ),
        agent_text: str = Query(""),
        threshold: float = Query(0.5, ge=0.0, le=1.0),
        deadline_ms: float | None = Query(
            None,
            gt=0,
            description="end-to-end server budget including body read, decoding and inference",
        ),
    ):
        t_start = time.perf_counter()
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
                if declared_size < 0:
                    raise HTTPException(400, "invalid negative Content-Length header")
                if declared_size > max_body_bytes:
                    raise HTTPException(413, f"request body exceeds {max_body_bytes} bytes")
            except ValueError as error:
                raise HTTPException(400, "invalid Content-Length header") from error
        if sem.locked():
            raise HTTPException(503, "at capacity; retry or fall back to timeout policy")
        await sem.acquire()
        future = None
        def remaining():
            if deadline_ms is None:
                return None
            return max(0.0, deadline_ms / 1000 - (time.perf_counter() - t_start))
        try:
            body = await asyncio.wait_for(_read_limited_body(request, max_body_bytes), timeout=remaining())
            if not body:
                raise HTTPException(400, "empty body")
            loop = asyncio.get_running_loop()
            def process():
                td = time.perf_counter()
                try:
                    x = _decode_request(body, sr)
                except Exception as e:
                    raise HTTPException(400, f"cannot decode audio: {e}") from e
                decode_ms = (time.perf_counter() - td) * 1000
                if remaining() is not None and remaining() <= 0:
                    raise HTTPException(409, "deadline exceeded during decoding")
                return len(x), decode_ms, engine.infer(x, agent_text)
            future = loop.run_in_executor(pool, process)
            n_samples, t_dec, (p, f, feat_ms, inf_ms) = await asyncio.wait_for(asyncio.shield(future), timeout=remaining())
            if remaining() is not None and remaining() <= 0:
                raise HTTPException(409, "deadline exceeded during inference")
        except asyncio.TimeoutError:
            raise HTTPException(409, "request deadline exceeded; discard this result") from None
        finally:
            if future is not None and not future.done():
                # A cancelled HTTP request cannot cancel a running native forward. Keep its
                # capacity slot occupied until completion to prevent unbounded executor queues.
                def finished(done):
                    if not done.cancelled():
                        done.exception()
                    sem.release()
                future.add_done_callback(finished)
            else:
                sem.release()
        total = (time.perf_counter() - t_start) * 1000
        return JSONResponse({
            "p_eot": p, "decision": "eot" if p >= threshold else "hold", "threshold": threshold,
            "p_fvad": dict(zip([str(h) for h in engine.horizons], f)) if getattr(engine, "use_fvad", True) else {},
            "audio_seconds": n_samples / SAMPLE_RATE, "context_used": bool(engine.use_context and agent_text),
            "timings_ms": {"decode": round(t_dec, 2), "features": round(feat_ms, 2), "inference": round(inf_ms, 2), "total": round(total, 2)},
            "model_sha": engine.sha, "version": __version__,
        })

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("EOT_THREADS", "2")))
    ap.add_argument("--max-inflight", type=int, default=8)
    ap.add_argument("--max-body-bytes", type=int, default=int(os.environ.get("EOT_MAX_BODY_BYTES", "2000000")))
    args = ap.parse_args(argv)
    import uvicorn

    uvicorn.run(
        create_app(args.onnx, args.threads, args.max_inflight, args.max_body_bytes),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
