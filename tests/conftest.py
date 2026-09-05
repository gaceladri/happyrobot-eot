"""Shared test doubles: a tiny Whisper config for model tests and a fake audio-only ONNX session."""

from __future__ import annotations

import numpy as np

TINY_WHISPER = dict(d_model=16, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32, num_mel_bins=80, max_source_positions=1500)


def tiny_model(seed: int = 17, **overrides):
    import torch

    from eot.modeling.model import EOTConfig, EOTModel

    torch.manual_seed(seed)
    return EOTModel(EOTConfig(use_fvad=False, whisper_config=TINY_WHISPER, **overrides), pretrained=False).eval()


def fake_audio_only_session(p_eot: float, registry: list | None = None):
    """A stand-in for ``onnxruntime.InferenceSession`` whose graph kept only ``input_features``."""

    class _Input:
        name = "input_features"

    class _Session:
        def __init__(self, *_args, **_kwargs):
            self.last_feed = None
            if registry is not None:
                registry.append(self)

        def get_inputs(self):
            return [_Input()]

        def run(self, _outputs, feed):
            self.last_feed = feed
            return np.array([p_eot], dtype=np.float32), np.zeros((1, 4), dtype=np.float32)

    return _Session
