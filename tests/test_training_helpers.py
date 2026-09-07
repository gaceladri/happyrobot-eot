"""Trainer plumbing shared by ``eot-train`` and ``eot-train-backbone``."""

from __future__ import annotations

import math
import time
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from eot.modeling.train import atomic_torch_save, pick_device, should_save_best, validate_resume_args
from eot.modeling.training_utils import PhaseTimer, cosine_scale, validate_common_args, warmup_steps


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


def _whisper_args(**overrides) -> Namespace:
    values = dict(
        base_model="openai/whisper-tiny",
        base_revision="abc",
        epochs=1,
        batch_size=32,
        lr=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.2,
        clip_grad=1.0,
        dev_frac=0.15,
        telephony_prob=0.3,
        fvad_weight=0.5,
        use_context=False,
        min_context_coverage=0.5,
        no_fvad=True,
        freeze_encoder=False,
        workers=4,
        seed=0,
        max_steps=None,
        fused_adamw=None,
        compile_model=False,
        compile_mode="reduce-overhead",
        init_checkpoint=None,
        exclude_train_ids=None,
        exclude_train_sha256=None,
        split_seed=17,
        max_source_positions=200,
        normalize_audio=True,
        tail_texture_prob=0.5,
        teacher_cache="cache",
        kd_temperature=2.0,
        log_every=20,
        checkpoint_every=200,
    )
    return Namespace(**{**values, **overrides})


@pytest.mark.parametrize(
    "key,saved",
    [
        ("seed", 99),
        ("compile_model", True),
        ("clip_grad", 0.5),
        ("kd_temperature", 4.0),
        ("tail_texture_prob", 0.0),
        ("exclude_train_sha256", "old"),
    ],
)
def test_resume_rejects_changed_semantic_arguments(key, saved) -> None:
    args = _whisper_args()
    with pytest.raises(ValueError, match=key):
        validate_resume_args({"args": {**vars(args), key: saved}}, args)
    validate_resume_args({"args": vars(args)}, args)


@pytest.mark.parametrize(
    "override,error",
    [
        ({"epochs": 0}, "epochs"),
        ({"warmup_ratio": -0.1}, "warmup-ratio"),
        ({"lr": 0.0}, "lr"),
        ({"log_every": 0}, "log-every"),
        ({"checkpoint_every": 0}, "checkpoint-every"),
        ({"workers": -1}, "workers"),
        ({"max_steps": 0}, "max-steps"),
        ({"clip_grad": -1.0}, "clip-grad"),
    ],
)
def test_common_argument_ranges(override, error) -> None:
    validate_common_args(_whisper_args(), learning_rates=("lr",))
    with pytest.raises(ValueError, match=error):
        validate_common_args(_whisper_args(**override), learning_rates=("lr",))


def test_cosine_scale_warms_up_linearly_then_decays_to_zero() -> None:
    planned, ratio = 100, 0.2
    warm = warmup_steps(planned, ratio)
    assert warm == 20
    assert cosine_scale(0, planned, ratio) == 0.0
    assert cosine_scale(10, planned, ratio) == pytest.approx(0.5)
    assert cosine_scale(warm, planned, ratio) == 1.0  # the warm-up boundary is the peak
    assert cosine_scale(60, planned, ratio) == pytest.approx(0.5)
    assert cosine_scale(planned, planned, ratio) == pytest.approx(0.0)
    assert cosine_scale(planned + 5, planned, ratio) == pytest.approx(0.0)
    scales = [cosine_scale(s, planned, ratio) for s in range(warm, planned + 1)]
    assert scales == sorted(scales, reverse=True)
    assert cosine_scale(0, planned, 0.0) == 1.0  # no warm-up: start at the peak
    assert cosine_scale(50, planned, 0.0) == pytest.approx(0.5)


def test_phase_timer_attributes_nested_phases_to_the_inner_phase() -> None:
    timer = PhaseTimer()
    time.sleep(0.01)
    timer.initialized()
    with timer.phase("train"):
        time.sleep(0.02)
        with timer.phase("checkpoint"):
            time.sleep(0.02)
    with timer.phase("validation"):
        pass
    with pytest.raises(ValueError, match="unknown phase"):
        with timer.phase("lunch"):
            pass
    timing = timer.finish()
    assert set(timing) == {"initialization_s", "train_s", "validation_s", "checkpoint_s", "other_s", "trainer_elapsed_s"}
    assert timing["initialization_s"] >= 0.01 and timing["checkpoint_s"] >= 0.02
    assert 0.02 <= timing["train_s"] < 0.04 == pytest.approx(timing["train_s"] + 0.02, abs=0.02) or timing["train_s"] >= 0.02
    assert timer.seconds("train") == timing["train_s"]
    assert min(timing.values()) >= 0
    assert sum(timing[k] for k in ("initialization_s", "train_s", "validation_s", "checkpoint_s", "other_s")) == pytest.approx(
        timing["trainer_elapsed_s"]
    )
