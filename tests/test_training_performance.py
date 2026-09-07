"""Performance changes must preserve predictions, updates and restart safety."""

from __future__ import annotations

import errno
import json
import os

import numpy as np
import pytest
import torch
from conftest import TINY_WHISPER
from torch.utils.data import DataLoader

from eot.data.dataset import collate
from eot.io import atomic_write_jsonl, write_wav
from eot.metrics import roc_auc
from eot.modeling import train as trainer
from eot.modeling.model import EOTConfig, EOTModel, save_checkpoint


@pytest.mark.parametrize("use_fvad,use_context", [(False, False), (True, False), (False, True), (True, True)])
def test_evaluation_matches_target_bearing_forward(use_fvad, use_context):
    torch.manual_seed(17)
    model = EOTModel(
        EOTConfig(
            whisper_config=TINY_WHISPER,
            max_source_positions=4,
            use_fvad=use_fvad,
            use_context=use_context,
        ),
        pretrained=False,
    ).eval()
    rows = [
        {
            "input_features": torch.randn(80, 8),
            "context_ids": torch.arange(32),
            "labels": torch.tensor(i % 2),
            "fvad": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "fvad_mask": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "weight": torch.tensor(float(i)),
        }
        for i in range(7)
    ]
    loader = DataLoader(rows, batch_size=3, collate_fn=collate)
    expected = []
    with torch.no_grad():
        for b in loader:
            out = model(b["input_features"], b["context_ids"], b["labels"], b["fvad"], b["fvad_mask"], weight=b["weight"])
            expected.extend(
                {
                    "label": int(b["labels"][i]),
                    "p_eot": float(out["p_eot"][i]),
                    "p_fvad": out["p_fvad"][i].tolist() if use_fvad else None,
                }
                for i in range(len(b["labels"]))
            )
    metrics, actual = trainer.evaluate(model, loader, torch.device("cpu"))
    assert actual == expected
    labels = np.array([r["label"] for r in expected])
    scores = np.array([r["p_eot"] for r in expected])
    hard_bce = metrics.pop("hard_bce")
    assert metrics == {"auc": roc_auc(labels, scores), "acc@0.5": ((scores > 0.5) == labels).mean(), "n": 7, "pos_rate": labels.mean()}
    weights = np.arange(7, dtype=np.float64)
    logits = np.log(scores) - np.log1p(-scores)
    per_row = np.logaddexp(0, -logits) * labels + np.logaddexp(0, logits) * (1 - labels)  # pos_weight is 1.0 here
    assert hard_bce == pytest.approx(float((per_row * weights).sum() / weights.sum()), rel=1e-5)


def test_loss_logging_preserves_float_sum_and_detaches_reused_storage():
    loss = torch.tensor(0.0, dtype=torch.float16, requires_grad=True)
    saved = []
    expected = []
    for value in [1000.0, 0.1, -1000.0, 0.2]:
        with torch.no_grad():
            loss.fill_(value)
        saved.append(loss.detach().clone())
        expected.append(float(loss.detach()))
    assert trainer.mean_logged_loss(saved) == sum(expected) / len(expected)
    assert all(not value.requires_grad for value in saved)


def test_checkpoint_alias_survives_new_last_and_stale_temp(tmp_path):
    best, last, model = [tmp_path / name for name in ("best.pt", "last.pt", "model.pt")]
    trainer.atomic_torch_save({"step": 1}, best)
    trainer.publish_checkpoint(best, last)
    trainer.publish_checkpoint(best, model)
    assert os.path.samefile(best, last)
    trainer.atomic_torch_save({"step": 2}, last)
    assert torch.load(best)["step"] == torch.load(model)["step"] == 1
    # Simulate a killed publication that left a hard-linked temporary file.
    os.link(best, model.with_name(f".{model.name}.{os.getpid()}.tmp"))
    trainer.publish_checkpoint(last, model)
    assert torch.load(best)["step"] == 1
    assert torch.load(model)["step"] == 2


def test_checkpoint_alias_fallback_and_failed_publication(tmp_path, monkeypatch):
    best, model = tmp_path / "best.pt", tmp_path / "model.pt"
    trainer.atomic_torch_save({"step": 1}, best)

    def cannot_link(*args):
        raise OSError(errno.EXDEV, "different filesystems")

    monkeypatch.setattr(trainer.os, "link", cannot_link)
    trainer.publish_checkpoint(best, model)
    assert not os.path.samefile(best, model)
    trainer.atomic_torch_save({"step": 2}, best)

    def failed_copy(source, destination):
        destination.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(trainer.shutil, "copyfile", failed_copy)
    with pytest.raises(OSError, match="disk full"):
        trainer.publish_checkpoint(best, model)
    assert torch.load(model)["step"] == 1
    assert not list(tmp_path.glob(".*.tmp"))


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


def test_training_resume_preserves_updates_and_records_complete_timing(tmp_path, monkeypatch):
    torch.manual_seed(17)
    model = EOTModel(EOTConfig(whisper_config=TINY_WHISPER, max_source_positions=4, use_fvad=False), pretrained=False)
    init = tmp_path / "init.pt"
    save_checkpoint(model, str(init))
    wav = tmp_path / "audio.wav"
    write_wav(wav, np.random.default_rng(17).normal(0, 0.1, 1600).astype(np.float32), 16000)
    manifest = tmp_path / "samples.jsonl"
    atomic_write_jsonl(
        manifest, [dict(id=str(i), path="audio.wav", label=i % 2, source=f"call-{i // 4}", clip_id=f"clip-{i // 4}") for i in range(24)]
    )

    def args(out):
        return trainer.build_parser().parse_args(
            [
                "--samples",
                str(manifest),
                "--out",
                str(out),
                "--init-checkpoint",
                str(init),
                "--no-fvad",
                "--device",
                "cpu",
                "--epochs",
                "2",
                "--max-steps",
                "7",
                "--batch-size",
                "4",
                "--checkpoint-every",
                "3",
                "--log-every",
                "2",
                "--seed",
                "17",
            ]
        )

    full, resumed = tmp_path / "full", tmp_path / "resumed"
    trainer.train(args(full))
    original_save = trainer.atomic_torch_save

    def interrupt(payload, path):
        original_save(payload, path)
        if payload["global_step"] == 3:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(trainer, "atomic_torch_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.train(args(resumed))
    monkeypatch.setattr(trainer, "atomic_torch_save", original_save)
    resumed_args = args(resumed)
    resumed_args.resume = "auto"
    trainer.train(resumed_args)
    for name in ("checkpoints/last.pt", "checkpoints/best.pt", "model.pt"):
        a, b = [torch.load(root / name, weights_only=False) for root in (full, resumed)]
        for key in (
            "state_dict",
            "optimizer_state_dict",
            "scaler_state_dict",
            "global_step",
            "best_auc",
            "loader_generator_state",
            "rng_state",
        ):
            _assert_state_equal(a[key], b[key])
    assert (full / "dev_scores.jsonl").read_bytes() == (resumed / "dev_scores.jsonl").read_bytes()
    for root in (full, resumed):
        timing = json.loads((root / "timing.json").read_text())
        phases = [timing[k] for k in ("initialization_s", "train_s", "validation_s", "checkpoint_s", "other_s")]
        assert min(phases) >= 0
        assert sum(phases) == pytest.approx(timing["trainer_elapsed_s"])
        assert timing["global_step"] == 7
