from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import eot.serving.service as serve


class _FakeEngine:
    def __init__(self, _onnx_path: str, threads: int = 2):
        self.sha = "fake-sha"
        self.use_context = False
        self.horizons = [0.24, 0.64, 1.2, 2.0]
        self.threads = threads

    def infer(self, _x: np.ndarray, _agent_text: str):
        return 0.75, [0.0, 0.0, 0.0, 0.0], 1.0, 2.0


def test_service_rejects_pathological_raw_sample_rates_and_large_bodies(monkeypatch) -> None:
    monkeypatch.setattr(serve, "Engine", _FakeEngine)
    client = TestClient(serve.create_app("unused.onnx", max_body_bytes=64))
    payload = np.zeros(16, dtype="<i2").tobytes()

    assert client.post("/v1/eot?sr=0", content=payload).status_code == 422
    assert client.post("/v1/eot?sr=1", content=payload).status_code == 422
    assert client.post("/v1/eot?sr=16000", content=b"x" * 66).status_code == 413

    response = client.post("/v1/eot?sr=16000", content=payload)
    assert response.status_code == 200
    assert response.json()["decision"] == "eot"



def test_deadline_covers_inference_and_holds_capacity(monkeypatch):
    import time
    import asyncio
    import httpx
    class Slow(_FakeEngine):
        def infer(self, x, text):
            time.sleep(.08)
            return super().infer(x, text)
    monkeypatch.setattr(serve, 'Engine', Slow)
    app=serve.create_app('unused', max_inflight=1)
    body=np.zeros(160,dtype='<i2').tobytes()
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            a=await client.post('/v1/eot?sr=16000&deadline_ms=10',content=body)
            assert a.status_code==409
            b=await client.post('/v1/eot?sr=16000',content=body)
            assert b.status_code==503
            await asyncio.sleep(.1)
            c=await client.post('/v1/eot?sr=16000',content=body)
            assert c.status_code==200
    asyncio.run(run())


def test_service_rejects_nonfinite_audio(monkeypatch):
    import io
    import soundfile as sf
    monkeypatch.setattr(serve, 'Engine', _FakeEngine)
    wav=io.BytesIO()
    sf.write(wav,np.array([np.nan]*160,dtype=np.float32),16000,format='WAV',subtype='FLOAT')
    with TestClient(serve.create_app('unused')) as client:
        assert client.post('/v1/eot',content=wav.getvalue()).status_code==400
