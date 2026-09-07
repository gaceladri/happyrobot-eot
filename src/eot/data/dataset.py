"""Torch datasets over ``samples.jsonl`` manifests; audio is augmented and featurised on the fly.

``MinedDataset`` feeds the Whisper models (log-mel), ``WaveformDataset`` feeds backbone models that
own their front-end (``eot.modeling.backbones``). Both share ``MinedDataset.waveform``, the one
place where the per-row augmentation stream is drawn, so a teacher and a student see the same
realised audio (see ``eot.data.teacher_cache``).

Targets are re-derived from the row's observation window when it carries one
(``future_observed_until`` + ``observation_end_reason``, rev. 4 manifests): a future-speech target
at horizon h is asserted only where the window covers h, or where the speaker's own resumption
ended the window before h. Older manifests without both fields keep today's rule (stored mask; EOT
rows contribute no fvad targets) and the number of such rows is counted in
``MinedDataset.legacy_counts``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from eot.audio import SAMPLE_RATE, add_room_tone, last_window, load_wav, log_mel, tail_texture, telephony_augment
from eot.context import hash_context
from eot.io import atomic_write_jsonl, safe_audio_filename, seed_from_key, write_wav
from eot.labeling.samples import HORIZONS, Clip, observed_fvad_targets

log = logging.getLogger(__name__)


def has_observation_window(row: dict) -> bool:
    """True for rev. 4 rows: both ``future_observed_until`` and ``observation_end_reason`` present."""
    return row.get("future_observed_until") is not None and row.get("observation_end_reason") is not None


def row_targets(row: dict) -> tuple[list[int], list[int], bool]:
    """``(fvad, fvad_mask, derived)`` for one manifest row.

    ``derived`` is True when the mask came from the row's observation window, which requires both
    rev. 4 fields (``future_observed_until`` and ``observation_end_reason``): an earlier labeler
    wrote ``future_observed_until`` as the raw span to the file end, not the window observed under
    waiting, and deriving from it would assert silence while the agent was already speaking.
    Otherwise the stored mask is used (a scalar is expanded to every horizon) and, as before, a
    label-1 row contributes no fvad target: an EOT clip label inferred unobserved future silence,
    which is exactly the censoring the window field makes explicit.
    """
    if has_observation_window(row):
        fvad, mask = observed_fvad_targets(
            row.get("time_to_onset"),
            float(row["future_observed_until"]),
            ended_by_resumption=row["observation_end_reason"] == "customer_resumed",
        )
        return fvad, mask, True
    mask = row.get("fvad_mask", 0)
    if isinstance(mask, (int, float)):
        mask = [int(mask)] * len(HORIZONS)
    mask = [int(v) for v in mask]
    if int(row["label"]) == 1:
        mask = [0] * len(HORIZONS)
    fvad = [int(v) for v in row.get("fvad", [0] * len(HORIZONS))]
    return fvad, mask, False


def row_weight(row: dict) -> float:
    """Per-row loss weight; absent or null means 1.0 (today's behaviour). Must be finite and >= 0."""
    value = row.get("weight")
    weight = 1.0 if value is None else float(value)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"row {row.get('id')!r} has an invalid weight {value!r}; expected a finite value >= 0")
    return weight


def tail_length_s(row: dict) -> float:
    """Seconds of pause stored after the cut: ``cut_time - pause_start`` (the clip ends at the cut)."""
    try:
        tail = float(row["cut_time"]) - float(row["pause_start"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row {row.get('id')!r} needs numeric cut_time and pause_start for tail texture") from exc
    if not math.isfinite(tail) or tail < 0:
        raise ValueError(f"row {row.get('id')!r} has an invalid tail length {tail!r}")
    return tail


class MinedDataset(Dataset):
    """Samples from ``eot-mine``; computes log-mel on the fly (cheap relative to the encoder).

    Augmentation order per row: room tone on ``noise_fill`` rows -> telephony leg (``telephony_prob``)
    -> tail texture (``tail_texture_prob``). The first two draw from the historical per-row stream
    ``seed:epoch:id``; tail texture draws from a second stream ``tail:seed:epoch:id`` so enabling it
    leaves every untouched row bit-identical to a run without it (a matched control).
    """

    def __init__(
        self,
        rows: list[dict],
        telephony_prob: float = 0.0,
        seed: int = 0,
        noise_fill: bool = True,
        normalize_audio: bool = False,
        tail_texture_prob: float = 0.0,
    ):
        if not 0.0 <= telephony_prob <= 1.0 or not 0.0 <= tail_texture_prob <= 1.0:
            raise ValueError("augmentation probabilities must lie in [0, 1]")
        self.rows = rows
        self.normalize_audio = normalize_audio
        self.telephony_prob = telephony_prob
        self.tail_texture_prob = tail_texture_prob
        self.seed = int(seed)
        self.epoch = 0
        # Rows flagged ``noise_fill`` (digital-silence sources such as AppTek) get continuous room
        # tone so 'exact zero = pause' can never be learned. Disable only for the raw diagnostic slice.
        self.noise_fill = bool(noise_fill)
        for row in rows:
            row_weight(row)  # fail at construction, not mid-epoch
            if tail_texture_prob > 0:
                tail_length_s(row)
        self.legacy_counts = {
            "rows": len(rows),
            "rows_without_observation_window": sum(not has_observation_window(r) for r in rows),
            "rows_without_weight": sum(r.get("weight") is None for r in rows),
        }
        if self.legacy_counts["rows_without_observation_window"] or self.legacy_counts["rows_without_weight"]:
            log.info("legacy manifest rows (fallback to stored fvad mask / weight 1.0): %s", self.legacy_counts)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _row_identity(self, i: int) -> str:
        return str(self.rows[i].get("id", self.rows[i].get("clip_id", i)))

    def _sample_rng(self, i: int) -> np.random.Generator:
        # Independent of worker assignment/prefetch, but different across epochs.
        return np.random.default_rng(seed_from_key(f"{self.seed}:{self.epoch}:{self._row_identity(i)}"))

    def _tail_rng(self, i: int) -> np.random.Generator:
        return np.random.default_rng(seed_from_key(f"tail:{self.seed}:{self.epoch}:{self._row_identity(i)}"))

    def __len__(self) -> int:
        return len(self.rows)

    def waveform(self, i: int) -> np.ndarray:
        """The augmented 16 kHz waveform of row ``i`` at the current epoch (what the features are computed from)."""
        r = self.rows[i]
        x = load_wav(r["path"])
        rng = self._sample_rng(i)
        if self.noise_fill and int(r.get("noise_fill", 0)):
            x = add_room_tone(x, rng)
        if self.telephony_prob > 0 and rng.random() < self.telephony_prob:
            x = telephony_augment(x, SAMPLE_RATE, rng)
        if self.tail_texture_prob > 0:
            tail_rng = self._tail_rng(i)
            if tail_rng.random() < self.tail_texture_prob:
                x, _ = tail_texture(x, tail_length_s(r), tail_rng)
        return x

    def targets(self, i: int) -> dict:
        """Label, future-speech targets and loss weight of row ``i`` as tensors (no audio)."""
        r = self.rows[i]
        fvad, mask, _ = row_targets(r)
        return {
            "labels": torch.tensor(int(r["label"])),
            "fvad": torch.tensor(fvad, dtype=torch.float32),
            "fvad_mask": torch.tensor(mask, dtype=torch.float32),
            "weight": torch.tensor(row_weight(r), dtype=torch.float32),
        }

    def _item(self, i: int, wave: np.ndarray) -> dict:
        """Features and targets of row ``i`` from an already generated ``wave`` (subclasses reuse the waveform)."""
        r = self.rows[i]
        return {
            "input_features": torch.from_numpy(log_mel(wave, normalize=self.normalize_audio)),
            "context_ids": torch.from_numpy(hash_context(r.get("agent_text", ""))),
            **self.targets(i),
        }

    def __getitem__(self, i: int) -> dict:
        return self._item(i, self.waveform(i))


class WaveformDataset(MinedDataset):
    """The same rows and augmentation stream as ``MinedDataset`` but returning the trailing ``input_s`` seconds of
    waveform (left zero-padded like ``eot.audio.last_window``), for backbones that compute their own front-end."""

    def __init__(self, rows: list[dict], *, input_s: float = 8.0, **kwargs):
        super().__init__(rows, **kwargs)
        self.input_s = float(input_s)

    def __getitem__(self, i: int) -> dict:
        wave = last_window(self.waveform(i), self.input_s)
        return {"wave": torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)), **self.targets(i)}


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def clips_to_manifest(clips: Iterable[Clip], out_dir: Path) -> Path:
    """Persist raw clips to wav + jsonl so mining/training are reproducible offline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "clips.jsonl"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    rows = []
    for c in clips:
        p = audio_dir / safe_audio_filename(c.id)
        rows.append(
            {
                "id": c.id,
                "path": p.relative_to(out_dir).as_posix(),
                "label": c.label,
                "source": c.source,
                "agent_text": c.agent_text,
                "sha256": write_wav(p, c.audio, SAMPLE_RATE),
            }
        )
    atomic_write_jsonl(manifest, rows)
    return manifest
