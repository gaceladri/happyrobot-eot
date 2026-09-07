"""``eot-train-backbone`` end to end on the fake backbone: exact resume, provenance, evaluation and argument checks."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from eot.audio import SAMPLE_RATE
from eot.data.dataset import collate
from eot.io import atomic_write_jsonl, sha256_file, write_wav
from eot.metrics import roc_auc
from eot.modeling import backbone_model as bm
from eot.modeling import train_backbone as trainer

pytestmark = pytest.mark.usefixtures("fake_backbone")


@pytest.fixture
def manifest(tmp_path):
    wav = tmp_path / "audio.wav"
    write_wav(wav, np.random.default_rng(17).normal(0, 0.1, SAMPLE_RATE).astype(np.float32), SAMPLE_RATE)
    path = tmp_path / "samples.jsonl"
    atomic_write_jsonl(
        path, [dict(id=str(i), path="audio.wav", label=i % 2, source=f"call-{i // 4}", clip_id=f"clip-{i // 4}") for i in range(24)]
    )
    return path


def _args(manifest, out, **overrides):
    values = {
        "samples": manifest,
        "out": out,
        "backbone": "fake",
        "repo": "fake/repo",
        "revision": "r1",
        "window-s": 0.5,
        "input-s": 1.0,
        "frozen-dtype": "float32",
        "lora-r": 2,
        "lora-dropout": 0.1,
        "device": "cpu",
        "epochs": 2,
        "max-steps": 7,
        "batch-size": 4,
        "micro-batch": 2,
        "checkpoint-every": 3,
        "log-every": 2,
        "seed": 17,
        "workers": 0,
        "eval-batch": 8,
        "max-dev-rows": 3,
        **overrides,
    }
    argv = [
        f"--{key}" if value is True else f"--{key}={value}" for key, value in values.items() if value is not None and value is not False
    ]
    return trainer.build_parser().parse_args(argv)


def _assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_equal(a, b)
    else:
        assert left == right


def test_resume_preserves_updates_rng_and_records_complete_timing(tmp_path, manifest, monkeypatch, fake_backbone):
    (fake_backbone / "modeling_fake.py").write_text("x = 1\n")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    trainer.train(_args(manifest, full, **{"keep-resume": True}))
    original_save = trainer.atomic_torch_save

    def interrupt(payload, path):
        original_save(payload, path)
        if payload.get("format") == trainer.RESUME_FORMAT and payload["step"] == 3:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(trainer, "atomic_torch_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.train(_args(manifest, resumed))
    monkeypatch.setattr(trainer, "atomic_torch_save", original_save)
    with pytest.raises(FileExistsError, match="already contains a run"):
        trainer.train(_args(manifest, resumed))
    with pytest.raises(ValueError, match="seed=17"):
        trainer.train(_args(manifest, resumed, resume=True, seed=18))
    trainer.train(_args(manifest, resumed, resume=True, **{"keep-resume": True}))

    # The final resume checkpoint (written at step 6 of both runs) must be byte-identical: weights, optimizer, RNG.
    a, b = [torch.load(root / "checkpoints" / "last.pt", weights_only=False) for root in (full, resumed)]
    for key in ("step", "epoch", "samples_done", "trainable_state", "optimizer_state_dict", "rng_state"):
        _assert_state_equal(a[key], b[key])
    for name in ("model.pt", "delta.pt"):
        a, b = [torch.load(root / name, weights_only=False)["state_dict"] for root in (full, resumed)]
        _assert_state_equal(a, b)
    assert (full / "dev_scores.jsonl").read_bytes() == (resumed / "dev_scores.jsonl").read_bytes()
    assert bm.check_delta(full / "model.pt", full / "delta.pt") == 0.0
    provenance = json.loads((full / "provenance.json").read_text())
    assert provenance["remote_code"] == {"modeling_fake.py": sha256_file(fake_backbone / "modeling_fake.py")}
    assert provenance["init_weights_sha256"] == bm.base_weights_sha256(fake_backbone)
    for root in (full, resumed):
        timing = json.loads((root / "timing.json").read_text())
        phases = [timing[k] for k in ("initialization_s", "train_s", "validation_s", "checkpoint_s", "other_s")]
        assert min(phases) >= 0
        assert sum(phases) == pytest.approx(timing["trainer_elapsed_s"])
        assert timing["global_step"] == 7 and timing["scope"] == "this invocation"
    assert json.loads((resumed / "timing.json").read_text())["resume_from"] == str(resumed / "checkpoints" / "last.pt")
    assert [h["step"] for h in json.loads((full / "history.json").read_text())["history"]] == [4, 7]  # 16 train rows -> 4 updates per epoch


def test_evaluate_matches_the_model_forward_and_row_order():
    torch.manual_seed(3)
    model = bm.BackboneEOTModel(
        bm.BackboneEOTConfig(backbone="fake", repo="fake/repo", revision="r1", window_s=0.5, input_s=1.0, frozen_dtype="float32")
    )
    rows = [{"wave": torch.randn(SAMPLE_RATE), "labels": torch.tensor(i % 2), "weight": torch.tensor(1.0)} for i in range(5)]
    loader = DataLoader(rows, batch_size=2, collate_fn=collate)
    metrics, scores = trainer.evaluate(model, loader, torch.device("cpu"), torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected = model.eval()(torch.stack([r["wave"] for r in rows]))["p_eot"].float()
    np.testing.assert_allclose([s["p_eot"] for s in scores], expected.numpy(), rtol=1e-6, atol=1e-6)
    assert [s["label"] for s in scores] == [0, 1, 0, 1, 0]
    labels, p = np.array([0, 1, 0, 1, 0]), expected.numpy()
    assert metrics == {"auc": roc_auc(labels, p), "acc@0.5": float(((p > 0.5) == (labels == 1)).mean()), "n": 5, "pos_rate": 0.4}
    with pytest.raises(ValueError, match="empty"):
        trainer.evaluate(model, DataLoader([], batch_size=2, collate_fn=collate), torch.device("cpu"), torch.bfloat16)


@pytest.mark.parametrize(
    "override,error",
    [
        ({"micro-batch": 3}, "multiple"),
        ({"lora-r": -1}, "lora-r"),
        ({"warmup-ratio": 1.5}, "warmup-ratio"),
        ({"lr-head": 0}, "lr-head"),
        ({"log-every": 0}, "log-every"),
        ({"workers": -1}, "workers"),
        ({"max-steps": 0}, "max-steps"),
        ({"eval-batch": 0}, "eval-batch"),
    ],
)
def test_invalid_arguments_are_rejected_before_any_work(tmp_path, manifest, override, error):
    with pytest.raises(ValueError, match=error):
        trainer.train(_args(manifest, tmp_path / "run", **override))
    assert not (tmp_path / "run").exists()


def test_rejects_legacy_resume_before_loading_data_or_model(tmp_path, monkeypatch):
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"format": "eot-backbone-resume-v1"}, checkpoint)

    def unexpected_read(*args, **kwargs):
        pytest.fail("legacy checkpoint must be rejected before reading the manifest")

    monkeypatch.setattr(trainer, "read_samples", unexpected_read)
    # No manifest or model is needed to reject an incompatible RNG/sampler format.
    with pytest.raises(ValueError, match="incompatible resume checkpoint format"):
        trainer.train(_args(tmp_path / "missing.jsonl", tmp_path / "resumed", resume=checkpoint))
