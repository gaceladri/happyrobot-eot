"""CPU inference service: the verified ONNX export on onnxruntime (torch-free).

    EOT_ONNX=artifacts/deployment/whisper-cpu/eot.onnx uv run eot-serve --host 0.0.0.0 --port 8000 --threads 2

The HTTP contract (endpoints, admission, deadlines, response schema) lives in ``eot.serving.http``;
this module only builds the ``OnnxScorer``. One warmed session with fixed intra/inter-op threads;
inference runs on a thread pool bounded by ``--max-inflight``.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eot.audio import SAMPLE_RATE, WINDOW_SECONDS, log_mel
from eot.context import CTX_LEN, hash_context
from eot.io import sha256_file
from eot.onnx import cpu_session, input_names, model_feed
from eot.serving.http import (
    MAX_BODY_BYTES,
    MAX_INFLIGHT,
    Score,
    build_app,
    resolve_model_path,
    run_service,
    service_parser,
    validate_limits,
)


class Engine:
    """onnxruntime session plus the export sidecar (``eot.json``) that proves the graph passed parity."""

    def __init__(self, onnx_path: str, threads: int = 2):
        onnx = Path(onnx_path)
        sidecar = onnx.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"missing export sidecar {sidecar}; only verified exports are served")
        self.meta: dict = json.loads(sidecar.read_text())
        if self.meta.get("accepted") is not True:
            raise ValueError(f"{sidecar} does not record an accepted export")
        digest = sha256_file(onnx)
        if self.meta.get("sha256") != digest:
            raise ValueError("model checksum does not match its sidecar")
        self.sha = digest[:16]
        self.sess = cpu_session(onnx, threads)
        self.input_names = input_names(self.sess)
        self.use_context = bool(self.meta.get("use_context", False))
        self.use_fvad = bool(self.meta.get("use_fvad", True))
        self.normalize_audio = bool(self.meta.get("normalize_audio", False))
        self.horizons: list[float] = self.meta.get("horizons", [0.24, 0.64, 1.2, 2.0])
        self.threads = threads
        # Warm up the *whole* path (feature extractor construction + session) so the first real
        # request does not pay a cold start.
        for _ in range(3):
            self.infer(np.zeros(int(SAMPLE_RATE * WINDOW_SECONDS), np.float32), "")

    def infer(self, x16k: np.ndarray, agent_text: str) -> tuple[float, list[float], float, float]:
        """``(p_eot, p_fvad per horizon, features_ms, inference_ms)``."""
        t0 = time.perf_counter()
        feats = log_mel(x16k, normalize=self.normalize_audio)[None]
        t1 = time.perf_counter()
        ctx = hash_context(agent_text)[None] if (self.use_context and agent_text) else np.zeros((1, CTX_LEN), np.int64)
        p, f = self.sess.run(None, model_feed(self.input_names, feats, ctx))
        t2 = time.perf_counter()
        return float(p[0]), [float(v) for v in f[0]], (t1 - t0) * 1000, (t2 - t1) * 1000


class OnnxScorer:
    """``eot.serving.http.Scorer`` over an ``Engine``; each request runs on its own pool thread."""

    def __init__(self, engine: Engine, max_inflight: int):
        self.engine = engine
        self.sha = engine.sha
        self.pool = ThreadPoolExecutor(max_workers=max_inflight, thread_name_prefix="onnx")

    async def start(self) -> None:
        pass

    async def score(self, wave: np.ndarray, agent_text: str) -> Score:
        p, fvad, features_ms, inference_ms = await asyncio.get_running_loop().run_in_executor(
            self.pool, self.engine.infer, wave, agent_text
        )
        return Score(
            p_eot=p,
            p_fvad=dict(zip([str(h) for h in self.engine.horizons], fvad)) if self.engine.use_fvad else {},
            features_ms=features_ms,
            inference_ms=inference_ms,
            context_used=bool(self.engine.use_context and agent_text),
        )

    def health(self) -> dict:
        return {"runtime": "onnxruntime", "threads": self.engine.threads, "use_context": self.engine.use_context}

    async def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def create_app(onnx_path: str | None = None, threads: int = 2, max_inflight: int = MAX_INFLIGHT, max_body_bytes: int = MAX_BODY_BYTES):
    if threads < 1:
        raise ValueError("threads must be positive")
    validate_limits(max_inflight, max_body_bytes)
    engine = Engine(resolve_model_path(onnx_path, "EOT_ONNX", "--onnx"), threads)
    return build_app(OnnxScorer(engine, max_inflight), title="eot", max_inflight=max_inflight, max_body_bytes=max_body_bytes)


def main(argv: list[str] | None = None) -> None:
    args = service_parser(__doc__, "--onnx", "EOT_ONNX").parse_args(argv)
    run_service(create_app(args.onnx, args.threads, args.max_inflight, args.max_body_bytes), args.host, args.port)


if __name__ == "__main__":
    main()
