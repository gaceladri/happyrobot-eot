"""Plumbing shared by the two trainers (``eot-train`` and ``eot-train-backbone``).

Seeding and RNG capture, the manifest row filter, resume-argument protection, the warm-up + cosine
schedule, environment provenance, phase timing, atomic checkpoint writes and a Weights & Biases
logger that never raises into a training loop. Nothing here knows which model is being trained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import platform
import random
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eot.io import atomic_replace

# Bumped whenever ``filter_time_to_onset`` changes; a resume across versions is refused.
DATA_FILTER_VERSION = 2
# ``time_to_onset`` in (-FLOAT_NOISE_S, 0) is grid arithmetic noise (about -1e-15 on AppTek rows):
# clamp to 0 and keep the row. Anything more negative is a cut placed after the next speech onset.
FLOAT_NOISE_S = 1e-6
PROVENANCE_PACKAGES = ("eot", "numpy", "torch", "transformers", "safetensors")

# Arguments of ``eot-train`` whose change on resume would alter samples, gradients or the LR schedule.
WHISPER_PROTECTED_ARGS = (
    "base_model",
    "base_revision",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "clip_grad",
    "dev_frac",
    "telephony_prob",
    "fvad_weight",
    "use_context",
    "min_context_coverage",
    "no_fvad",
    "freeze_encoder",
    "workers",
    "seed",
    "max_steps",
    "fused_adamw",
    "compile_model",
    "compile_mode",
    "init_checkpoint",
    "exclude_train_ids",
    "exclude_train_sha256",
    "split_seed",
    "max_source_positions",
    "normalize_audio",
    "tail_texture_prob",
    "teacher_cache",
    "kd_temperature",
)


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keep the module docstring's layout and show every default."""


# ---- seeding ---------------------------------------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def pick_device(name: str | None = None, *, require_cuda: bool = False) -> torch.device:
    if name:
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError(
            f"CUDA is required but unavailable (selected={device}, torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return device


# ---- rows ------------------------------------------------------------------------------------------------------------
def filter_time_to_onset(rows: list[dict]) -> tuple[list[dict], dict]:
    """Clamp float-noise negative ``time_to_onset`` to 0 and exclude genuine cuts after the onset.

    Older energy/VAD manifests occasionally place a cut just beyond the earlier VAD speech onset;
    those are not valid pause decisions and are excluded. Rows whose value is negative only by float
    noise are kept with ``time_to_onset = 0.0``. Both populations are counted so neither is silent.
    """
    kept, excluded, clamped = [], 0, 0
    for row in rows:
        tto = row.get("time_to_onset")
        if tto is not None and tto < 0:
            if tto <= -FLOAT_NOISE_S:
                excluded += 1
                continue
            row["time_to_onset"] = 0.0
            clamped += 1
        kept.append(row)
    return kept, {"excluded_cuts_after_speech_onset": excluded, "clamped_float_noise_time_to_onset": clamped}


def validate_binary_rows(rows: list[dict], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} split is empty")
    labels = {int(row["label"]) for row in rows}
    if labels != {0, 1}:
        raise ValueError(f"{name} split must contain labels 0 and 1; found {sorted(labels)}")


# ---- arguments -------------------------------------------------------------------------------------------------------
def validate_common_args(args: argparse.Namespace, *, learning_rates: tuple[str, ...]) -> None:
    """Range checks shared by both trainers; ``learning_rates`` names the LR arguments that must be positive."""
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must lie in [0, 1]")
    for name in learning_rates:
        if not getattr(args, name) > 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.log_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("--log-every and --checkpoint-every must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive when set")
    if args.clip_grad < 0:
        raise ValueError("--clip-grad cannot be negative (0 disables clipping)")


def validate_resume_args(checkpoint: dict, args: argparse.Namespace, protected: tuple[str, ...] = WHISPER_PROTECTED_ARGS) -> None:
    """Reject changes to ``protected`` arguments that would alter samples, gradients or the LR schedule."""
    saved = checkpoint.get("args") or {}
    mismatches = {key: (saved[key], getattr(args, key)) for key in protected if key in saved and saved[key] != getattr(args, key)}
    if mismatches:
        details = ", ".join(f"{key}={old!r} (requested {new!r})" for key, (old, new) in sorted(mismatches.items()))
        raise ValueError(f"resume arguments changed training semantics: {details}")


def resolve_resume(value: str | None, out_dir: Path) -> Path | None:
    """``--resume [PATH]``: ``None`` = fresh run, ``"auto"`` = ``<out>/checkpoints/last.pt``, else an explicit path."""
    if value is None:
        return None
    path = out_dir / "checkpoints" / "last.pt" if value == "auto" else Path(value)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


# ---- schedule --------------------------------------------------------------------------------------------------------
def warmup_steps(planned: int, warmup_ratio: float) -> int:
    return int(warmup_ratio * planned)


def cosine_scale(step: int, planned: int, warmup_ratio: float) -> float:
    """Learning-rate multiplier at ``step``: linear warm-up over ``warmup_ratio`` of ``planned`` updates, then cosine to 0."""
    warm = warmup_steps(planned, warmup_ratio)
    if step < warm:
        return step / max(1, warm)
    progress = (step - warm) / max(1, planned - warm)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ---- provenance ------------------------------------------------------------------------------------------------------
def source_digest(directory: Path) -> str:
    """sha256 over the ``*.py`` files of ``directory`` (name + bytes, sorted), identifying the trainer source."""
    h = hashlib.sha256()
    for p in sorted(Path(directory).glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment_provenance(device: torch.device) -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dependencies": {package: importlib.metadata.version(package) for package in PROVENANCE_PACKAGES},
    }


# ---- timing ----------------------------------------------------------------------------------------------------------
class PhaseTimer:
    """Wall-clock split of a trainer invocation into initialization / train / validation / checkpoint / other.

    Phases nest: a checkpoint written inside the train phase counts as checkpoint time, not train time.
    ``initialized()`` closes the initialization phase; ``finish()`` returns the ``timing.json`` fields.
    """

    PHASES = ("initialization", "train", "validation", "checkpoint")

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._seconds = dict.fromkeys(self.PHASES, 0.0)
        self._stack: list[str] = []

    def initialized(self) -> None:
        self._seconds["initialization"] = time.perf_counter() - self._started

    def seconds(self, name: str) -> float:
        """Seconds attributed to ``name`` so far (completed uses of the phase only)."""
        return self._seconds[name]

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in self._seconds:
            raise ValueError(f"unknown phase {name!r}")
        self._stack.append(name)
        t = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t
            self._stack.pop()
            self._seconds[name] += elapsed
            if self._stack:
                self._seconds[self._stack[-1]] -= elapsed

    def finish(self) -> dict[str, float]:
        elapsed = time.perf_counter() - self._started
        timing = {f"{name}_s": seconds for name, seconds in self._seconds.items()}
        timing["other_s"] = elapsed - sum(self._seconds.values())
        timing["trainer_elapsed_s"] = elapsed
        return timing


# ---- checkpoints and logging -----------------------------------------------------------------------------------------
def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint next to its destination and atomically publish it."""
    atomic_replace(path, lambda tmp: torch.save(payload, tmp))


def mean_logged_loss(values: list[torch.Tensor]) -> float:
    """Mean of per-update scalars kept on device (losses, gradient norms): one transfer per log interval."""
    return sum(torch.stack(values).cpu().tolist()) / len(values)


class WandbLogger:
    """Optional Weights & Biases logging that never raises into the trainer (a network problem must not kill a GPU run).

    ``project=None`` disables logging; the remaining keyword arguments go to ``wandb.init``.
    """

    def __init__(self, project: str | None, **init_kwargs: Any):
        self.run = None
        if not project:
            return
        try:
            import wandb

            self.run = wandb.init(project=project, **init_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"wandb disabled: {exc!r}", flush=True)

    def log(self, values: dict, step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.log(values, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"wandb log failed: {exc!r}", flush=True)

    def summarize(self, summary: dict) -> None:
        if self.run is None:
            return
        try:
            for key, value in summary.items():
                self.run.summary[key] = value
        except Exception as exc:  # noqa: BLE001
            print(f"wandb summary failed: {exc!r}", flush=True)

    def finish(self, summary: dict | None = None) -> None:
        if self.run is None:
            return
        self.summarize(summary or {})
        try:
            self.run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"wandb finish failed: {exc!r}", flush=True)
