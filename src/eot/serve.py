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
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from . import __version__
from .audio import SAMPLE_RATE, WINDOW_SECONDS, load_audio_bytes, log_mel, resample, to_mono
from .context import CTX_LEN, hash_context


class Engine:
    def __init__(self, onnx_path: str, threads: int = 2):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.sess.get_inputs()}
        self.sha = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()[:16]
        meta = Path(onnx_path).with_suffix(".json")
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        self.use_context = bool(self.meta.get("use_context", False))
        self.horizons = self.meta.get("horizons", [0.24, 0.64, 1.2, 2.0])
        self.threads = threads
        # Warm up the *whole* path (feature extractor construction + session) so the first real
        # request does not pay a cold start.
        for _ in range(3):
            self.infer(np.zeros(SAMPLE_RATE * 8, np.float32), "")

    def infer(self, x16k: np.ndarray, agent_text: str) -> tuple[float, list[float], float, float]:
        t0 = time.perf_counter()
        feats = log_mel(x16k)[None]
        t1 = time.perf_counter()
        ctx = hash_context(agent_text)[None] if (self.use_context and agent_text) else np.zeros((1, CTX_LEN), np.int64)
        feed = {"input_features": feats}
        if "context_ids" in self.input_names:
            feed["context_ids"] = ctx
        p, f = self.sess.run(None, feed)
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
    app = FastAPI(title="eot", version=__version__)

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
            description="drop the request if it cannot start before this budget",
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
        async with sem:
            if deadline_ms is not None and (time.perf_counter() - t_start) * 1000 > deadline_ms:
                raise HTTPException(409, "deadline exceeded before inference; result would be stale")
            body = await _read_limited_body(request, max_body_bytes)
            if not body:
                raise HTTPException(400, "empty body")
            loop = asyncio.get_running_loop()
            try:
                x = await loop.run_in_executor(pool, _decode_request, body, sr)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(400, f"cannot decode audio: {e}") from e
            t_dec = (time.perf_counter() - t_start) * 1000
            p, f, feat_ms, inf_ms = await loop.run_in_executor(pool, engine.infer, x, agent_text)
        total = (time.perf_counter() - t_start) * 1000
        return JSONResponse({
            "p_eot": p, "decision": "eot" if p >= threshold else "hold", "threshold": threshold,
            "p_fvad": dict(zip([str(h) for h in engine.horizons], f)),
            "audio_seconds": len(x) / SAMPLE_RATE, "context_used": bool(engine.use_context and agent_text),
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
