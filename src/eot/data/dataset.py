"""Torch dataset over ``samples.jsonl`` manifests; log-mel is computed on the fly.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from eot.audio import SAMPLE_RATE, add_room_tone, load_wav, log_mel, telephony_augment
from eot.context import hash_context
from eot.io import atomic_write_jsonl, safe_audio_filename, seed_from_key, write_wav
from eot.labeling.samples import HORIZONS, Clip


class MinedDataset(Dataset):
    """Samples from ``eot-mine``; computes log-mel on the fly (cheap relative to the encoder)."""

    def __init__(self, rows: list[dict], telephony_prob: float = 0.0, seed: int = 0, noise_fill: bool = True, normalize_audio: bool = False):
        self.rows = rows
        self.normalize_audio = normalize_audio
        self.telephony_prob = telephony_prob
        self.seed = int(seed)
        self.epoch = 0
        # Rows flagged ``noise_fill`` (digital-silence sources such as AppTek) get continuous room
        # tone so 'exact zero = pause' can never be learned. Disable only for the raw diagnostic slice.
        self.noise_fill = bool(noise_fill)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_rng(self, i: int) -> np.random.Generator:
        # Independent of worker assignment/prefetch, but different across epochs.
        identity = str(self.rows[i].get("id", self.rows[i].get("clip_id", i)))
        return np.random.default_rng(seed_from_key(f"{self.seed}:{self.epoch}:{identity}"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        x = load_wav(r["path"])
        rng = self._sample_rng(i)
        if self.noise_fill and int(r.get("noise_fill", 0)):
            x = add_room_tone(x, rng)
        if self.telephony_prob > 0 and rng.random() < self.telephony_prob:
            x = telephony_augment(x, SAMPLE_RATE, rng)
        mask = r.get("fvad_mask", 0)
        if isinstance(mask, (int, float)):
            mask = [int(mask)] * len(HORIZONS)
        # Old manifests inferred unobserved future silence from an EOT clip label.
        # Ignore those auxiliary targets until explicitly relabeled from observed audio.
        if int(r["label"]) == 1 and "future_observed_until" not in r:
            mask = [0] * len(HORIZONS)
        return {
            "input_features": torch.from_numpy(log_mel(x, normalize=self.normalize_audio)),
            "context_ids": torch.from_numpy(hash_context(r.get("agent_text", ""))),
            "labels": torch.tensor(int(r["label"])),
            "fvad": torch.tensor(r.get("fvad", [0] * len(HORIZONS)), dtype=torch.float32),
            "fvad_mask": torch.tensor(mask, dtype=torch.float32),
        }


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
        rows.append({
            "id": c.id,
            "path": p.relative_to(out_dir).as_posix(),
            "label": c.label,
            "source": c.source,
            "agent_text": c.agent_text,
            "sha256": write_wav(p, c.audio, SAMPLE_RATE),
        })
    atomic_write_jsonl(manifest, rows)
    return manifest
