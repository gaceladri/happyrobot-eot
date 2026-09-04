from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import eot.serve as serve


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

