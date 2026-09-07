"""Shared test doubles: a tiny Whisper config for model tests, a fake audio-only ONNX session and a fake backbone."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from eot.modeling import backbones

TINY_WHISPER = dict(d_model=16, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32, num_mel_bins=80, max_source_positions=1500)
FAKE_D, FAKE_LAYERS = 8, 3


def tiny_model(seed: int = 17, **overrides):
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


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(FAKE_D, FAKE_D)
        self.linear1 = nn.Linear(FAKE_D, FAKE_D)
        self.norm = nn.LayerNorm(FAKE_D)

    def forward(self, x):
        return self.norm(x + torch.tanh(self.linear1(self.linear_q(x))))


class FakeBackbone(backbones.Backbone):
    """Deterministic stand-in: mean-pooled frames -> 3 residual layers; 'pretrained' weights come from a fixed seed."""

    name = "fake"
    lora_targets = ("linear_q", "linear1")
    frame_rate_hz = 10.0
    snapshots: dict = {}

    def __init__(self, repo="fake/repo", revision="r1", pretrained=True, backbone_config=None, frontend_config=None):
        super().__init__()
        self.repo, self.revision = repo, revision
        self._backbone_config = dict(backbone_config or {"d_model": FAKE_D, "n_layers": FAKE_LAYERS})
        self._frontend_config = dict(frontend_config or {"hop": 1600})
        self.d_model, self.n_layers = FAKE_D, FAKE_LAYERS
        self.embed = nn.Linear(1, FAKE_D)
        self.layers = nn.ModuleList(_FakeLayer() for _ in range(FAKE_LAYERS))
        self.final_norm = nn.LayerNorm(FAKE_D)
        if pretrained:
            generator = torch.Generator().manual_seed(1234)
            for p in self.parameters():
                p.data.copy_(torch.randn(p.shape, generator=generator) * 0.2)

    @classmethod
    def snapshot_dir(cls, repo, revision):
        return cls.snapshots[(repo, revision)]

    def config_dict(self):
        return dict(self._backbone_config)

    def frontend_dict(self):
        return dict(self._frontend_config)

    def frontend(self, wave):
        hop = self._frontend_config["hop"]
        return wave[:, : wave.shape[1] // hop * hop].reshape(wave.shape[0], -1, hop).mean(-1, keepdim=True)

    def layer_modules(self):
        return list(self.layers)

    def pre_layer_modules(self):
        return [self.embed]

    def post_layer_modules(self):
        return [self.final_norm]

    def forward(self, wave, layer=-1):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.pre_trainable):
            x = self.embed(self.frontend(wave))
        layers = self.layer_modules()
        for i in range(self.layers_used(layer)):
            x = self.run_layer(i, layers[i], x)
        return self.final_norm(x) if layer == -1 else x


@pytest.fixture
def fake_backbone(monkeypatch, tmp_path):
    """Register ``FakeBackbone`` under ``"fake"`` with a snapshot directory holding fake weights; returns that directory."""
    monkeypatch.setitem(backbones.REGISTRY, FakeBackbone.name, FakeBackbone)
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"fake weights")
    monkeypatch.setitem(FakeBackbone.snapshots, ("fake/repo", "r1"), snap)
    return snap
