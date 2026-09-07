"""GPU inference service: the verified TensorRT plan behind one GPU executor with request batching.

    EOT_TRT_ENGINE=artifacts/deployment/cohere-gpu/eot.plan uv run eot-serve-gpu --host 0.0.0.0 --port 8000

The HTTP contract lives in ``eot.serving.http``; this module only builds the ``TensorRTScorer``,
which groups admitted requests for up to ``--wait-ms`` (at most ``MAX_BATCH`` per group) and keeps
every CUDA call on a single worker thread. ``tensorrt``/``torch`` are imported when the engine is
built, so the module imports on a CPU-only host (tests inject a fake engine).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import HTTPException

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
from eot.serving.tensorrt_engine import MAX_BATCH

_Request = tuple[np.ndarray, asyncio.Future]


class TensorRTScorer:
    """``eot.serving.http.Scorer`` that batches concurrent requests onto one engine worker.

    Requests wait in a queue sized like the admission limit, so a slot is always available to an
    admitted request; the worker collects a group for ``wait_ms`` (or until ``max_batch``) and
    runs ``engine.infer`` on its single thread. ``close`` sends a ``None`` sentinel: a group being
    collected when it arrives is still scored, and the worker exits on the next read.
    """

    def __init__(self, engine, *, max_inflight: int = MAX_INFLIGHT, max_batch: int = MAX_BATCH, wait_ms: float = 2.0):
        self.engine = engine
        self.sha: str = engine.sha
        self.max_batch, self.wait_ms = max_batch, wait_ms
        self.queue: asyncio.Queue[_Request | None] = asyncio.Queue(maxsize=max_inflight + 1)  # + the sentinel
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tensorrt")
        self.observed: Counter[int] = Counter()
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="tensorrt-batcher")

    async def score(self, wave: np.ndarray, agent_text: str) -> Score:
        if self.task is None or self.task.done():
            raise HTTPException(503, "inference worker is not running")
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.queue.put_nowait((wave, future))
        return await future

    def health(self) -> dict:
        return {
            "max_batch": self.max_batch,
            "batch_wait_ms": self.wait_ms,
            "observed_batch_sizes": dict(self.observed),
            **self.engine.metadata(),
        }

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            if self.task is not None:
                if not self.task.done():
                    await self.queue.put(None)
                await self.task
        finally:
            try:
                await loop.run_in_executor(self.pool, self.engine.close)
            finally:
                self.pool.shutdown(wait=True, cancel_futures=True)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            if first is None:
                return
            group = [first]
            deadline = loop.time() + self.wait_ms / 1000
            while len(group) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout)
                except TimeoutError:
                    break
                if item is None:
                    self.queue.put_nowait(None)  # finish this group; the sentinel is read again on the next loop
                    break
                group.append(item)
            await self._score_group(group, loop)

    async def _score_group(self, group: list[_Request], loop: asyncio.AbstractEventLoop) -> None:
        self.observed[len(group)] += 1
        try:
            values, features_ms, inference_ms = await loop.run_in_executor(self.pool, self.engine.infer, [w for w, _ in group])
            if len(values) != len(group):
                raise ValueError("Output batch length mismatch")
            for (_, future), p in zip(group, values, strict=True):
                if not future.done():
                    future.set_result(Score(float(p), {}, features_ms, inference_ms, batch_size=len(group)))
        except Exception as exc:  # noqa: BLE001 - propagate each engine failure to its callers
            for _, future in group:
                if not future.done():
                    future.set_exception(exc)


def create_app(
    engine_path: str | None = None,
    *,
    threads: int = 2,
    wait_ms: float = 2.0,
    graphs: bool = True,
    max_inflight: int = MAX_INFLIGHT,
    max_body_bytes: int = MAX_BODY_BYTES,
    _engine=None,
):
    """``_engine`` lets tests inject a fake engine; production loads the plan from ``engine_path`` or ``EOT_TRT_ENGINE``."""
    if threads < 1:
        raise ValueError("threads must be positive")
    if not 0 <= wait_ms <= 10:
        raise ValueError("wait_ms must be between 0 and 10")
    validate_limits(max_inflight, max_body_bytes)
    if _engine is None:
        from eot.serving.tensorrt_engine import TensorRTEngine

        _engine = TensorRTEngine(resolve_model_path(engine_path, "EOT_TRT_ENGINE", "--engine"), threads=threads, graphs=graphs)
    scorer = TensorRTScorer(_engine, max_inflight=max_inflight, wait_ms=wait_ms)
    return build_app(scorer, title="eot-tensorrt", max_inflight=max_inflight, max_body_bytes=max_body_bytes)


def main(argv: list[str] | None = None) -> None:
    ap = service_parser(__doc__, "--engine", "EOT_TRT_ENGINE")
    ap.add_argument("--wait-ms", type=float, default=2.0, help="batching window per group, 0..10 ms")
    ap.add_argument("--no-graphs", action="store_true", help="enqueue directly instead of replaying captured CUDA graphs")
    args = ap.parse_args(argv)
    app = create_app(
        args.engine,
        threads=args.threads,
        wait_ms=args.wait_ms,
        graphs=not args.no_graphs,
        max_inflight=args.max_inflight,
        max_body_bytes=args.max_body_bytes,
    )
    run_service(app, args.host, args.port)


if __name__ == "__main__":
    main()
