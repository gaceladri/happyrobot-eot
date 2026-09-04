from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def test_eotbench_adapter_import_is_torch_free() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    code = "import sys; import eot.eotbench_adapter; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)


def test_audio_only_onnx_adapter_omits_pruned_context(monkeypatch, tmp_path: Path) -> None:
    import onnxruntime as ort

    from eot import eotbench_adapter as module

    sessions = []

    class _Input:
        name = "input_features"

    class _Session:
        def __init__(self, *_args, **_kwargs):
            self.last_feed = None
            sessions.append(self)

        def get_inputs(self):
            return [_Input()]

        def run(self, _outputs, feed):
            self.last_feed = feed
            return np.array([0.25], dtype=np.float32), np.zeros((1, 4), dtype=np.float32)

    monkeypatch.setattr(ort, "InferenceSession", _Session)
    monkeypatch.setattr(module, "log_mel", lambda _audio: np.zeros((80, 800), dtype=np.float32))
    model_path = tmp_path / "audio-only.onnx"
    model_path.write_bytes(b"model-a")
    adapter = module.EOTAdapter(onnx=str(model_path))
    scores = adapter.predict_batch([
        {"audio": {"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}, "messages": []}
    ])
    assert scores == [0.25]
    assert set(sessions[0].last_feed) == {"input_features"}
    assert adapter.model_sha[:12] in adapter.adapter_id


def test_audio_decoder_accepts_path_only_huggingface_payload(tmp_path: Path) -> None:
    from eot.eotbench_adapter import _audio_to_16k

    path = tmp_path / "audio.wav"
    sf.write(path, np.linspace(-0.1, 0.1, 800, dtype=np.float32), 8000)
    decoded = _audio_to_16k({"bytes": None, "path": str(path)})
    assert decoded.dtype == np.float32
    assert len(decoded) == 1600
