"""Backbone models: freezing modes, LoRA placement, full/delta checkpoints and the harness adapter, on a tiny fake backbone."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import FAKE_D, FAKE_LAYERS

from eot.audio import SAMPLE_RATE
from eot.io import sha256_file
from eot.modeling import backbone_model as bm
from eot.modeling.lora import LoRALinear, lora_parameters, merge_lora

pytestmark = pytest.mark.usefixtures("fake_backbone")
D, LAYERS = FAKE_D, FAKE_LAYERS


def _cfg(**kw):
    defaults = dict(backbone="fake", repo="fake/repo", revision="r1", window_s=2.0, input_s=4.0, frozen_dtype="float32")
    return bm.BackboneEOTConfig(**{**defaults, **kw})


def test_lora_mode_trains_only_adapters_and_head_and_crops_the_window():
    torch.manual_seed(0)
    model = bm.BackboneEOTModel(_cfg(lora_r=2, lora_alpha=4.0)).eval()  # eval: the head has dropout
    assert model.n_lora == LAYERS * 2
    adapters = sum(p.numel() for p in lora_parameters(model.backbone))
    final_norm = sum(p.numel() for p in model.backbone.final_norm.parameters())
    assert adapters == LAYERS * 2 * (2 * D + D * 2)  # per wrapped linear: A [r, D] + B [D, r] with r = 2
    assert model.trainable["mode"].startswith("lora r=2") and model.trainable["encoder_trainable"] == adapters + final_norm
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.endswith(("lora_A", "lora_B")) or n.startswith(("pool.", "classifier.", "backbone.final_norm.")) for n in names)
    wave = torch.randn(3, 4 * SAMPLE_RATE)
    out = model(wave, labels=torch.tensor([0, 1, 1]), weight=torch.tensor([1.0, 2.0, 0.0]))
    assert out["logit"].shape == (3,) and torch.isfinite(out["loss"])
    torch.testing.assert_close(out["logit"], model(wave[:, -2 * SAMPLE_RATE :])["logit"])  # the model crops the trailing window itself
    out["loss"].backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad)


@pytest.mark.parametrize(
    "kw,mode,pre",
    [({"freeze_encoder": True}, "frozen", False), ({"unfreeze_top_k": 1}, "top 1 of 3 layers (2..2)", False), ({}, "full", True)],
)
def test_freezing_modes(kw, mode, pre):
    model = bm.BackboneEOTModel(_cfg(**kw))
    assert model.trainable["mode"] == mode and model.backbone.pre_trainable is pre
    enc_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    assert enc_trainable == model.trainable["encoder_trainable"]
    if kw.get("freeze_encoder"):
        assert enc_trainable == 0
    if kw.get("unfreeze_top_k"):
        assert all(not p.requires_grad for p in model.backbone.layers[0].parameters())
        assert all(p.requires_grad for p in model.backbone.layers[2].parameters())


@pytest.mark.parametrize(
    "kw,error",
    [
        ({"freeze_encoder": True, "lora_r": 2}, "freeze_encoder"),
        ({"freeze_encoder": True, "unfreeze_top_k": 1}, "freeze_encoder"),
        ({"unfreeze_top_k": 0}, "unfreeze_top_k"),
        ({"lora_r": -1}, "lora_r"),
        ({"lora_dropout": 1.5}, "lora_dropout"),
        ({"frozen_dtype": "float16"}, "frozen_dtype"),
    ],
)
def test_config_rejects_contradictory_or_out_of_range_values(kw, error):
    with pytest.raises(ValueError, match=error):
        _cfg(**kw)


def test_full_and_delta_checkpoints_rebuild_the_same_model(tmp_path, fake_backbone):
    torch.manual_seed(1)
    model = bm.BackboneEOTModel(_cfg(lora_r=2))
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    wave = torch.randn(2, 4 * SAMPLE_RATE)
    expected = model.eval()(wave)["logit"]
    bm.save_checkpoint(model, tmp_path / "model.pt")
    base_sha = bm.base_weights_sha256(fake_backbone)
    assert base_sha == sha256_file(fake_backbone / "model.safetensors") and (fake_backbone / "model.safetensors.sha256.json").exists()
    bm.save_delta_checkpoint(model, tmp_path / "delta.pt", base_weights_sha256=base_sha)
    delta = torch.load(tmp_path / "delta.pt", weights_only=False)
    assert delta["format"] == bm.FORMAT_DELTA and set(delta["state_dict"]) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert (tmp_path / "delta.pt").stat().st_size < (tmp_path / "model.pt").stat().st_size
    for path in ("model.pt", "delta.pt"):
        rebuilt = bm.load_checkpoint(tmp_path / path)
        torch.testing.assert_close(rebuilt(wave)["logit"], expected)
        assert not any(p.requires_grad for p in rebuilt.parameters())
    assert bm.check_delta(tmp_path / "model.pt", tmp_path / "delta.pt") == 0.0
    assert bm.is_backbone_checkpoint(tmp_path / "delta.pt") and not bm.is_backbone_checkpoint(tmp_path / "snapshot" / "model.safetensors")
    merged = bm.load_checkpoint(tmp_path / "delta.pt")
    assert merge_lora(merged) == LAYERS * 2 and not any(isinstance(m, LoRALinear) for m in merged.modules())
    torch.testing.assert_close(merged(wave)["logit"], expected, rtol=1e-5, atol=1e-5)


def test_delta_refuses_base_weights_it_was_not_trained_on(tmp_path, fake_backbone):
    model = bm.BackboneEOTModel(_cfg(lora_r=2))
    bm.save_delta_checkpoint(model, tmp_path / "delta.pt", base_weights_sha256=bm.base_weights_sha256(fake_backbone))
    bm.load_checkpoint(tmp_path / "delta.pt")
    (fake_backbone / "model.safetensors").write_bytes(b"other weights")  # a different snapshot under the same repo/revision
    with pytest.raises(ValueError, match="base weights"):
        bm.load_checkpoint(tmp_path / "delta.pt")


def test_legacy_research_formats_are_accepted(tmp_path):
    model = bm.BackboneEOTModel(_cfg(lora_r=2))
    bm.save_delta_checkpoint(model, tmp_path / "delta.pt")  # no base sha recorded: nothing to check
    ck = torch.load(tmp_path / "delta.pt", weights_only=False)
    ck["format"] = "l031-delta-v1"
    torch.save(ck, tmp_path / "legacy.pt")
    bm.load_checkpoint(tmp_path / "legacy.pt")
    ck["format"] = "unknown"
    torch.save(ck, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="unknown backbone checkpoint format"):
        bm.load_checkpoint(tmp_path / "bad.pt")


def test_backbone_adapter_matches_the_model_and_names_itself(tmp_path, monkeypatch):
    from eot.eval.backbone import BackboneAdapter

    model = bm.BackboneEOTModel(_cfg(lora_r=2))
    bm.save_checkpoint(model, tmp_path / "model.pt")
    monkeypatch.setenv("EOT_CHECKPOINT", str(tmp_path / "model.pt"))
    monkeypatch.setenv("EOT_ADAPTER_BATCH_SIZE", "2")
    adapter = BackboneAdapter(device="cpu")
    assert adapter.adapter_id == f"fake-w2s-audio-{sha256_file(tmp_path / 'model.pt')[:12]}-ieee-fp32"
    audio = [np.random.default_rng(i).normal(0, 0.1, 3 * SAMPLE_RATE).astype(np.float32) for i in range(3)]
    p = adapter.predict_batch([{"audio": {"array": a, "sampling_rate": SAMPLE_RATE}, "messages": []} for a in audio])
    with torch.no_grad():
        expected = model.eval()(torch.from_numpy(np.stack([np.pad(a, (SAMPLE_RATE, 0)) for a in audio])))["p_eot"]
    np.testing.assert_allclose(p, expected.numpy(), rtol=1e-5, atol=1e-6)
