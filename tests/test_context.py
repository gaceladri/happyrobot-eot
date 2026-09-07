from __future__ import annotations

import json
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


def test_audio_only_serve_feed_omits_pruned_context_input(tmp_path: Path) -> None:
    """The audio-only ONNX graph has no context input and must still be runnable."""
    from unittest.mock import patch

    from conftest import fake_audio_only_session

    from eot.io import sha256_file
    from eot.serving import service as serve

    _Session = fake_audio_only_session(0.5)
    model_path = tmp_path / "audio-only.onnx"
    model_path.write_bytes(b"test-model")
    model_path.with_suffix(".json").write_text(json.dumps({"accepted": True, "sha256": sha256_file(model_path)}))
    with patch("onnxruntime.InferenceSession", _Session), patch.object(serve.Engine, "infer", autospec=True) as warmup:
        # Avoid invoking feature extraction in __init__; exercise infer explicitly below.
        warmup.return_value = (0.5, [0.0] * 4, 0.0, 0.0)
        engine = serve.Engine(str(model_path))
    with patch("eot.serving.service.log_mel", return_value=np.zeros((80, 800), dtype=np.float32)):
        serve.Engine.infer(engine, np.zeros(16000, dtype=np.float32), "")
    assert set(engine.sess.last_feed) == {"input_features"}
