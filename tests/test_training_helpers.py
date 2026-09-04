from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from argparse import Namespace

from eot.train import _validate_resume_args, atomic_torch_save, pick_device, should_save_best


def test_best_selection_is_nan_safe_and_always_selects_first() -> None:
    assert should_save_best(math.nan, None, has_best=False)
    assert not should_save_best(math.nan, 0.8, has_best=True)
    assert should_save_best(0.9, 0.8, has_best=True)
    assert not should_save_best(0.7, 0.8, has_best=True)


def test_atomic_torch_save_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "last.pt"
    atomic_torch_save({"step": 3, "value": torch.tensor([1, 2])}, path)
    restored = torch.load(path, weights_only=False)
    assert restored["step"] == 3
    assert restored["value"].tolist() == [1, 2]
    assert not list(path.parent.glob("*.tmp"))


def test_require_cuda_rejects_cpu() -> None:
    with pytest.raises(RuntimeError, match="CUDA is required"):
        pick_device("cpu", require_cuda=True)


def test_resume_rejects_changed_semantic_arguments() -> None:
    args = Namespace(
        base_model="openai/whisper-tiny", base_revision="abc", epochs=3, batch_size=32,
        lr=5e-5, weight_decay=0.01, warmup_ratio=0.2, dev_frac=0.15,
        telephony_prob=0.3, fvad_weight=0.5, use_context=False,
        min_context_coverage=0.5, no_fvad=False, freeze_encoder=False, workers=4,
        seed=0, max_steps=None,
    )
    saved = vars(args).copy()
    saved["seed"] = 99
    with pytest.raises(ValueError, match="seed"):
        _validate_resume_args({"args": saved}, args)
