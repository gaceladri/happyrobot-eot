"""GPU service API/lifecycle over a fake engine (no CUDA); numerical tests use the real engine (scripts/verify_trt.py)."""

from __future__ import annotations

import asyncio
import io
import threading

import httpx
import numpy as np
import pytest
import soundfile as sf

from eot.serving.tensorrt_service import TensorRTScorer, create_app


class Engine:
    """Scores each waveform by its last sample; ``release`` gates inference so tests never sleep."""

    sha = "test"

    def __init__(self):
        self.closed = False
        self.calls: list[int] = []
        self.release = threading.Event()
        self.release.set()

    def infer(self, waves):
        self.release.wait(5)
        self.calls.append(len(waves))
        return np.array([float(w[-1]) for w in waves]), 0.0, 0.0

    def metadata(self):
        return {"runtime": "test"}

    def close(self):
        self.closed = True


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_batch_maps_each_probability_to_its_request():
    async def run():
        engine = Engine()
        app = create_app(_engine=engine)
        async with app.router.lifespan_context(app), _client(app) as client:
            bodies = [np.full(1600, n, dtype="<i2").tobytes() for n in (5000, 15000, 25000)]
            results = await asyncio.gather(*[client.post("/v1/eot?sr=16000", content=b) for b in bodies])
            assert all(r.status_code == 200 for r in results)
            assert np.allclose([r.json()["p_eot"] for r in results], np.array([5000, 15000, 25000]) / 32768)
            assert sum(engine.calls) == 3
            assert sum(r.json()["batch_size"] for r in results) >= 3
        assert engine.closed

    asyncio.run(run())


def test_wav_body_is_decoded_on_the_gpu_path():
    async def run():
        engine = Engine()
        app = create_app(_engine=engine)
        wav = io.BytesIO()
        sf.write(wav, np.full(1600, 0.5, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
        async with app.router.lifespan_context(app), _client(app) as client:
            response = await client.post("/v1/eot", content=wav.getvalue())
            assert response.status_code == 200
            body = response.json()
            assert abs(body["p_eot"] - 0.5) < 1e-3 and body["audio_seconds"] == 0.1 and body["p_fvad"] == {}

    asyncio.run(run())


def test_timed_out_requests_retain_capacity_until_native_work_finishes():
    async def run():
        engine = Engine()
        engine.release.clear()
        app = create_app(_engine=engine, wait_ms=2, max_inflight=4)
        body = np.full(1600, 1000, dtype="<i2").tobytes()
        async with app.router.lifespan_context(app), _client(app) as client:
            results = await asyncio.gather(*[client.post("/v1/eot?sr=16000&deadline_ms=30", content=body) for _ in range(4)])
            assert all(r.status_code == 409 for r in results)
            assert (await client.post("/v1/eot?sr=16000", content=body)).status_code == 503
            engine.release.set()
            for _ in range(50):
                await asyncio.sleep(0.01)
                if (await client.post("/v1/eot?sr=16000", content=body)).status_code == 200:
                    break
            else:
                raise AssertionError("capacity was not released after the native forward finished")

    asyncio.run(run())


def test_invalid_requests_do_not_consume_capacity():
    async def run():
        app = create_app(_engine=Engine(), max_body_bytes=64)
        async with app.router.lifespan_context(app), _client(app) as client:
            assert (await client.post("/v1/eot", content=b"")).status_code == 400
            assert (await client.post("/v1/eot?sr=16000", content=b"x")).status_code == 400
            assert (await client.post("/v1/eot?threshold=2", content=b"xx")).status_code == 422
            assert (await client.post("/v1/eot", content=b"x" * 65)).status_code == 413
            health = (await client.get("/healthz")).json()
            assert health["max_body_bytes"] == 64 and health["max_batch"] == 8 and health["runtime"] == "test"

    asyncio.run(run())


def test_exceptional_lifespan_exit_closes_engine():
    async def run():
        engine = Engine()
        app = create_app(_engine=engine)
        with pytest.raises(RuntimeError, match="application failure"):
            async with app.router.lifespan_context(app):
                raise RuntimeError("application failure")
        assert engine.closed

    asyncio.run(run())


def test_failed_inference_releases_capacity_and_worker_recovers():
    class FailingOnce(Engine):
        def infer(self, waves):
            if not self.calls:
                self.calls.append(len(waves))
                raise RuntimeError("inference failed")
            return super().infer(waves)

    async def run():
        engine = FailingOnce()
        app = create_app(_engine=engine)
        body = np.full(1600, 1000, dtype="<i2").tobytes()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert (await client.post("/v1/eot?sr=16000", content=body)).status_code == 500
                results = await asyncio.gather(*[client.post("/v1/eot?sr=16000", content=body) for _ in range(8)])
                assert all(response.status_code == 200 for response in results)
        assert engine.closed

    asyncio.run(run())


def test_sentinel_during_group_collection_still_scores_the_group():
    async def run():
        engine = Engine()
        scorer = TensorRTScorer(engine, wait_ms=10)
        await scorer.start()
        wave = np.full(1600, 0.25, dtype=np.float32)
        request = asyncio.create_task(scorer.score(wave, ""))
        await asyncio.sleep(0.001)  # the worker is now inside the 10 ms collection window
        await scorer.close()  # sentinel arrives mid-group
        assert (await request).p_eot == 0.25
        assert engine.calls == [1] and engine.closed
        with pytest.raises(Exception, match="not running"):
            await scorer.score(wave, "")

    asyncio.run(run())


def test_create_app_validates_configuration(monkeypatch):
    monkeypatch.delenv("EOT_TRT_ENGINE", raising=False)
    with pytest.raises(RuntimeError, match="EOT_TRT_ENGINE"):
        create_app()
    with pytest.raises(ValueError, match="wait_ms"):
        create_app(_engine=Engine(), wait_ms=11)
    with pytest.raises(ValueError, match="threads"):
        create_app(_engine=Engine(), threads=0)
