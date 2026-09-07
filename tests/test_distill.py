"""Distillation: the KL objective, the model's soft-target path, the teacher cache and the exact-view guard."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import TINY_WHISPER, tiny_model
from torch.utils.data import DataLoader

from eot.audio import SAMPLE_RATE, WINDOW_SECONDS, last_window
from eot.data.dataset import MinedDataset, collate
from eot.data.teacher_cache import CacheWriter, DistillationDataset, cache_identity, load_cache, waveform_sha256
from eot.io import atomic_write_jsonl, write_wav
from eot.modeling import distill
from eot.modeling import train as trainer
from eot.modeling.model import EOTConfig, EOTModel, load_checkpoint, save_checkpoint


def test_bernoulli_kl_matches_torch_distributions_and_analytic_gradient():
    s = torch.tensor([-7.0, -0.4, 2.0, 8.0], requires_grad=True)
    t = torch.tensor([3.0, -0.2, 1.0, -4.0], requires_grad=True)
    w = torch.tensor([1.0, 0.0, 2.0, 3.0])
    loss = distill.bernoulli_kl(s, t, 2.0, w)
    per = (
        torch.distributions.kl_divergence(torch.distributions.Bernoulli(logits=t.detach() / 2), torch.distributions.Bernoulli(logits=s / 2))
        * 4
    )
    torch.testing.assert_close(loss, (per * w).sum() / w.sum(), atol=1e-6, rtol=0)
    loss.backward()
    torch.testing.assert_close(s.grad, 2 * (torch.sigmoid(s.detach() / 2) - torch.sigmoid(t.detach() / 2)) * w / w.sum(), atol=1e-6, rtol=0)
    assert t.grad is None  # teacher targets are detached


def test_bernoulli_kl_edge_cases():
    s = torch.tensor([-1000.0, 1000.0, 0.0], requires_grad=True)
    loss = distill.bernoulli_kl(s, -s.detach())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(s.grad).all()
    assert distill.bernoulli_kl(s, -s.detach(), weight=torch.zeros(3)).item() == 0.0
    assert distill.bernoulli_kl(s, s.detach()).item() == 0.0
    with pytest.raises(ValueError):
        distill.bernoulli_kl(s, s, temperature=0.0)


def test_model_soft_target_path_has_no_extra_parameters_and_keeps_fvad(tmp_path):
    model = tiny_model(pos_weight=3.0, max_source_positions=4)
    x, y = torch.randn(2, 80, 8), torch.tensor([0, 1])
    hard = model(x, labels=y)
    soft = model(x, labels=y, teacher_logits=torch.tensor([-1.0, 1.0]), kd_temperature=2.0)
    torch.testing.assert_close(soft["loss"], distill.bernoulli_kl(hard["logit"], torch.tensor([-1.0, 1.0]), 2.0))
    assert "loss_eot" not in soft and "loss_kd" in soft and "loss_eot" in hard
    save_checkpoint(model, str(tmp_path / "model.pt"))
    torch.testing.assert_close(load_checkpoint(str(tmp_path / "model.pt"))(x)["logit"], hard["logit"])
    torch.manual_seed(0)
    fvad_model = EOTModel(EOTConfig(use_fvad=True, whisper_config=TINY_WHISPER, max_source_positions=4), pretrained=False).eval()
    out = fvad_model(x, labels=y, fvad=torch.zeros(2, 4), fvad_mask=torch.ones(2, 4), teacher_logits=torch.tensor([0.5, -0.5]))
    torch.testing.assert_close(out["loss"], out["loss_kd"] + fvad_model.cfg.fvad_weight * out["loss_fvad"])


def test_evaluate_reports_hard_bce_for_soft_trained_models():
    model = tiny_model(pos_weight=2.0, max_source_positions=4)
    rows = [
        {
            "input_features": torch.randn(80, 8),
            "context_ids": torch.zeros(32, dtype=torch.long),
            "labels": torch.tensor(i % 2),
            "fvad": torch.zeros(4),
            "fvad_mask": torch.zeros(4),
            "weight": torch.tensor(1.0 + i),
            "teacher_logits": torch.tensor(9.0),
        }
        for i in range(4)
    ]
    loader = DataLoader(rows, batch_size=4, collate_fn=collate)
    metrics, _ = trainer.evaluate(model, loader, torch.device("cpu"))
    batch = next(iter(loader))
    with torch.no_grad():
        logit = model(batch["input_features"])["logit"]
    per = F.binary_cross_entropy_with_logits(logit, batch["labels"].float(), pos_weight=torch.tensor(2.0), reduction="none")
    assert metrics["hard_bce"] == pytest.approx(float((per * batch["weight"]).sum() / batch["weight"].sum()))


@pytest.fixture
def rows(tmp_path):
    result = []
    for i, n in enumerate((3200, 48000, 128000, 160000)):
        x = (0.15 * np.sin(np.arange(n, dtype=np.float32) * 0.071)).astype(np.float32)
        x[-3200:] = 0
        path = tmp_path / f"{i}.wav"
        write_wav(path, x, SAMPLE_RATE)
        result.append(
            {
                "id": f"synthetic{i}",
                "path": str(path),
                "label": i % 2,
                "source": "synthetic",
                "noise_fill": i % 2,
                "cut_time": n / SAMPLE_RATE,
                "pause_start": n / SAMPLE_RATE - 0.2,
                "weight": 1.0 + i / 4,
            }
        )
    return result


def _windows(dataset):
    return [np.ascontiguousarray(last_window(dataset.waveform(i), WINDOW_SECONDS), dtype=np.float32) for i in range(len(dataset))]


def test_cache_round_trip_resume_and_integrity(tmp_path, rows):
    dataset = MinedDataset(rows, telephony_prob=0.3, seed=111, tail_texture_prob=0.5)
    identity = cache_identity(rows, dataset, {"checkpoint_sha256": "t"})
    hashes = [waveform_sha256(w) for w in _windows(dataset)]
    cache = tmp_path / "cache"
    with CacheWriter(cache, identity, rows) as writer:
        writer.append(0, np.array([0.25, -1.25], np.float32), hashes[:2])
    assert not (cache / "metadata.json").exists()
    with CacheWriter(cache, identity, rows) as writer:  # resume continues after the durable shard
        assert writer.next_start() == 2
        with pytest.raises(ValueError):
            writer.append(0, [1.0, 2.0], hashes[:2])
        with pytest.raises(ValueError):
            writer.finish()
        with pytest.raises(ValueError):
            writer.append(2, [np.inf, 1.0], hashes[2:])
        writer.append(2, np.array([2.0, -3.0], np.float32), hashes[2:])
        metadata = writer.finish()
    logits, cached_hashes, _ = load_cache(cache, rows, identity)
    assert logits.tolist() == [0.25, -1.25, 2.0, -3.0] and [h.decode() for h in cached_hashes] == hashes
    assert metadata["status"] == "COMPLETE"
    with pytest.raises(ValueError):
        load_cache(cache, rows[::-1])
    with pytest.raises(ValueError):
        load_cache(cache, rows, {**identity, "teacher": {"checkpoint_sha256": "other"}})
    with pytest.raises(ValueError):
        CacheWriter(cache, {**identity, "view": {**identity["view"], "seed": 222}}, rows)
    corrupt = np.load(cache / "logits.npy")
    corrupt[0] = 1.0
    np.save(cache / "logits.npy", corrupt)
    with pytest.raises(ValueError, match="integrity"):
        load_cache(cache, rows)


@pytest.mark.parametrize("epoch", [0, 1])
def test_distillation_dataset_pairs_exact_views_and_rejects_others(tmp_path, rows, epoch):
    kwargs = {"telephony_prob": 0.3, "seed": 111, "tail_texture_prob": 0.5, "normalize_audio": True}
    dataset = MinedDataset(rows, **kwargs)
    dataset.set_epoch(epoch)
    hashes = [waveform_sha256(w) for w in _windows(dataset)]
    cache = tmp_path / "cache"
    with CacheWriter(cache, cache_identity(rows, dataset, {"checkpoint_sha256": "t"}), rows) as writer:
        writer.append(0, np.arange(4, dtype=np.float32) - 1.5, hashes)
        writer.finish()
    distilled = DistillationDataset(rows, cache, epoch=epoch, **kwargs)
    for i in range(len(rows)):
        item, reference = distilled[i], dataset[i]
        for key in reference:
            assert torch.equal(item[key], reference[key]), key
        assert item["teacher_logits"].item() == i - 1.5
    with pytest.raises(ValueError, match="epoch"):
        distilled.set_epoch(epoch + 1)
    with pytest.raises(ValueError, match="view"):
        DistillationDataset(rows, cache, epoch=epoch, **{**kwargs, "tail_texture_prob": 0.0})
    # The same view over changed audio: the identity still matches, the per-row waveform hash does not.
    changed = np.zeros(48000, np.float32)
    changed[::7] = 0.2
    write_wav(rows[1]["path"], changed, SAMPLE_RATE)
    distilled[0]
    with pytest.raises(ValueError, match="differs from the one the teacher scored"):
        distilled[1]


def test_build_cache_with_a_fake_teacher_is_resumable_and_validates(tmp_path, rows):
    dataset = MinedDataset(rows, telephony_prob=0.3, seed=111)
    calls = []

    def teacher(waves):
        calls.append(len(waves))
        return waves.mean(axis=1) * 100.0

    out = tmp_path / "cache"
    first = distill.build_cache(dataset, teacher, out, {"checkpoint_sha256": "fake"}, batch_size=3, workers=2)
    second = distill.build_cache(dataset, teacher, out, {"checkpoint_sha256": "fake"}, batch_size=3, workers=2)
    assert first["fingerprint"] == second["fingerprint"]
    logits, _, _ = load_cache(out, rows)
    expected = np.array([teacher(w[None])[0] for w in _windows(dataset)], np.float32)
    np.testing.assert_allclose(logits, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="parity"):
        distill.batching_parity(lambda w: np.arange(len(w), dtype=np.float32), np.zeros((4, 128000), np.float32), 2)


def test_trainer_distils_from_a_cache_and_records_it(tmp_path, monkeypatch):
    torch.manual_seed(17)
    model = EOTModel(EOTConfig(whisper_config=TINY_WHISPER, max_source_positions=4, use_fvad=False), pretrained=False)
    init = tmp_path / "init.pt"
    save_checkpoint(model, str(init))
    wav = tmp_path / "audio.wav"
    write_wav(wav, np.random.default_rng(17).normal(0, 0.1, 16000).astype(np.float32), SAMPLE_RATE)
    manifest = tmp_path / "samples.jsonl"
    rows = [
        dict(id=str(i), path="audio.wav", label=i % 2, source=f"call-{i // 4}", clip_id=f"clip-{i // 4}", cut_time=1.0, pause_start=0.7)
        for i in range(24)
    ]
    atomic_write_jsonl(manifest, rows)

    def train_args(out, **overrides):
        values = {
            "samples": manifest,
            "init-checkpoint": init,
            "no-fvad": True,
            "device": "cpu",
            "epochs": 1,
            "batch-size": 4,
            "seed": 17,
            "split-seed": 3,
            "tail-texture-prob": 0.5,
            "log-every": 1,
            "out": out,
            "teacher-cache": tmp_path / "cache",
            **overrides,
        }
        argv = [f"--{key}" if value is True else f"--{key}={value}" for key, value in values.items() if value is not None]
        return trainer.build_parser().parse_args(argv)

    # Build the cache on exactly the rows and view the trainer will use, with a stand-in teacher.
    class FakeTeacher:
        identity = {"checkpoint_sha256": "fake"}

        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, waves):
            return waves.std(axis=1) * 10.0

    monkeypatch.setattr(distill, "BackboneTeacher", FakeTeacher)
    distill.main(
        [
            "--samples",
            str(manifest),
            "--teacher",
            "unused",
            "--out",
            str(tmp_path / "cache"),
            "--seed",
            "17",
            "--split-seed",
            "3",
            "--tail-texture-prob",
            "0.5",
            "--batch-size",
            "5",
            "--workers",
            "1",
        ]
    )
    trainer.train(train_args(tmp_path / "kd", **{"kd-temperature": 3}))
    provenance = json.loads((tmp_path / "kd" / "provenance.json").read_text())
    assert provenance["distillation"]["kd_temperature"] == 3.0
    assert provenance["distillation"]["teacher"] == {"checkpoint_sha256": "fake"}
    assert provenance["augmentation"]["tail_texture_prob"] == 0.5
    history = json.loads((tmp_path / "kd" / "history.json").read_text())["history"]
    assert "hard_bce" in history[-1]
    with pytest.raises(ValueError, match="single-pass"):
        trainer.train(train_args(tmp_path / "bad", epochs=2))
    with pytest.raises(ValueError, match="view"):
        trainer.train(train_args(tmp_path / "bad2", **{"tail-texture-prob": 0.0}))
