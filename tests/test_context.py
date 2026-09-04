from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eot.context import CTX_LEN, hash_context


def test_hash_context_is_deterministic_and_padding_zero() -> None:
    empty = hash_context("")
    first = hash_context("What is your MC number?")
    second = hash_context("What is your MC number?")
    assert empty.dtype == np.int64
    assert empty.shape == (CTX_LEN,)
    assert not empty.any()
    np.testing.assert_array_equal(first, second)
    assert first.any()


@pytest.mark.parametrize("max_len,vocab", [(0, 10), (2, 1)])
def test_hash_context_rejects_invalid_shape(max_len: int, vocab: int) -> None:
    with pytest.raises(ValueError):
        hash_context("hello", max_len=max_len, vocab=vocab)


def test_serve_import_does_not_import_torch() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    code = "import sys; import eot.serve; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)


def test_audio_only_serve_feed_omits_pruned_context_input(tmp_path: Path) -> None:
    """The audio-only ONNX graph has no context input and must still be runnable."""
    from unittest.mock import patch

    from eot import serve

    class _Input:
        name = "input_features"

    class _Session:
        def __init__(self, *_args, **_kwargs):
            self.last_feed = None

        def get_inputs(self):
            return [_Input()]

        def run(self, _outputs, feed):
            self.last_feed = feed
            return np.array([0.5], dtype=np.float32), np.zeros((1, 4), dtype=np.float32)

    model_path = tmp_path / "audio-only.onnx"
    model_path.write_bytes(b"test-model")
    with patch("onnxruntime.InferenceSession", _Session), patch.object(serve.Engine, "infer", autospec=True) as warmup:
        # Avoid invoking feature extraction in __init__; exercise infer explicitly below.
        warmup.return_value = (0.5, [0.0] * 4, 0.0, 0.0)
        engine = serve.Engine(str(model_path))
    with patch("eot.serve.log_mel", return_value=np.zeros((80, 800), dtype=np.float32)):
        serve.Engine.infer(engine, np.zeros(16000, dtype=np.float32), "")
    assert set(engine.sess.last_feed) == {"input_features"}
