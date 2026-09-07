"""CPU service contract over a fake ONNX engine: validation, admission, deadlines and the torch-free import guarantee."""

from __future__ import annotations

import asyncio
import io
import subprocess
import sys
import threading

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import eot.serving.service as serve
from eot import __version__

PCM = np.zeros(160, dtype="<i2").tobytes()


class _FakeEngine:
    def __init__(self, _onnx_path: str, threads: int = 2):
        self.sha = "fake-sha"
        self.use_context = False
        self.use_fvad = True
        self.horizons = [0.24, 0.64, 1.2, 2.0]
        self.threads = threads
        self.release = threading.Event()  # tests block inference here instead of sleeping
        self.release.set()

    def infer(self, _x: np.ndarray, _agent_text: str):
        self.release.wait(5)
        return 0.75, [0.0, 0.0, 0.0, 0.0], 1.0, 2.0


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(serve, "Engine", _FakeEngine)


def test_service_rejects_pathological_raw_sample_rates_and_large_bodies(fake_engine) -> None:
    client = TestClient(serve.create_app("unused.onnx", max_body_bytes=64))
    payload = np.zeros(16, dtype="<i2").tobytes()

    assert client.post("/v1/eot?sr=0", content=payload).status_code == 422
    assert client.post("/v1/eot?sr=1", content=payload).status_code == 422
    assert client.post("/v1/eot?sr=16000", content=b"x" * 66).status_code == 413

    response = client.post("/v1/eot?sr=16000", content=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "eot" and body["batch_size"] == 1 and body["version"] == __version__
    assert body["p_fvad"] == {"0.24": 0.0, "0.64": 0.0, "1.2": 0.0, "2.0": 0.0}
    assert (body["timings_ms"]["features"], body["timings_ms"]["inference"]) == (1.0, 2.0)


def test_content_length_header_branches(fake_engine) -> None:
    client = TestClient(serve.create_app("unused.onnx", max_body_bytes=64))
    assert client.post("/v1/eot?sr=16000", content=PCM[:16], headers={"content-length": "abc"}).status_code == 400
    assert client.post("/v1/eot?sr=16000", content=PCM[:16], headers={"content-length": "-1"}).status_code == 400
    assert client.post("/v1/eot?sr=16000", content=PCM[:16], headers={"content-length": "65"}).status_code == 413
    assert client.post("/v1/eot?sr=16000", content=b"").status_code == 400


def test_healthz_reports_configured_values(fake_engine) -> None:
    client = TestClient(serve.create_app("unused.onnx", threads=3, max_inflight=2, max_body_bytes=64))
    health = client.get("/healthz").json()
    assert health["max_inflight"] == 2 and health["max_body_bytes"] == 64 and health["threads"] == 3
    assert health["model_sha"] == "fake-sha" and health["runtime"] == "onnxruntime" and "source_sha256" in health


def test_deadline_covers_inference_and_holds_capacity(monkeypatch):
    engines: list[_FakeEngine] = []

    class Gated(_FakeEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.release.clear()
            engines.append(self)

    monkeypatch.setattr(serve, "Engine", Gated)
    app = serve.create_app("unused", max_inflight=1)
    (engine,) = engines

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            stale = await client.post("/v1/eot?sr=16000&deadline_ms=20", content=PCM)
            assert stale.status_code == 409
            # The timed-out forward is still running: its slot stays taken.
            assert (await client.post("/v1/eot?sr=16000", content=PCM)).status_code == 503
            engine.release.set()
            for _ in range(50):
                await asyncio.sleep(0.01)
                if (await client.post("/v1/eot?sr=16000", content=PCM)).status_code == 200:
                    break
            else:
                raise AssertionError("capacity was not released after the native forward finished")

    asyncio.run(run())


def test_service_rejects_nonfinite_and_empty_audio(fake_engine):
    nonfinite, empty = io.BytesIO(), io.BytesIO()
    sf.write(nonfinite, np.array([np.nan] * 160, dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
    sf.write(empty, np.zeros(0, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    with TestClient(serve.create_app("unused")) as client:
        assert client.post("/v1/eot", content=nonfinite.getvalue()).status_code == 400
        assert client.post("/v1/eot", content=empty.getvalue()).status_code == 400


def test_engine_requires_accepted_sidecar(tmp_path):
    onnx = tmp_path / "eot.onnx"
    onnx.write_bytes(b"not a graph")
    with pytest.raises(FileNotFoundError, match="sidecar"):
        serve.Engine(str(onnx))
    onnx.with_suffix(".json").write_text('{"accepted": false}')
    with pytest.raises(ValueError, match="accepted"):
        serve.Engine(str(onnx))


def test_create_app_validates_configuration(fake_engine, monkeypatch):
    monkeypatch.delenv("EOT_ONNX", raising=False)
    with pytest.raises(RuntimeError, match="EOT_ONNX"):
        serve.create_app()
    with pytest.raises(ValueError, match="max_inflight"):
        serve.create_app("unused", max_inflight=0)
    with pytest.raises(ValueError, match="threads"):
        serve.create_app("unused", threads=0)


@pytest.mark.parametrize("module", ["eot.serving.http", "eot.serving.service"])
def test_cpu_service_imports_without_torch(module):
    code = f"import {module}, sys; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_to_pcm16_bytes_clips_and_truncates_like_the_load_test_payload():
    from eot.audio import to_pcm16_bytes

    assert np.frombuffer(to_pcm16_bytes(np.array([1.5, -1.5, 0.5, 0.0], dtype=np.float32)), "<i2").tolist() == [32767, -32767, 16383, 0]
