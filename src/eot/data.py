"""Datasets, source-grouped splits and collation.

Two entry points:

- ``load_smart_turn_clips``: stream ``pipecat-ai/smart-turn-data-v3.2-train`` from the Hub,
  filter by language, cap per label, and yield ``Clip`` objects ready for prefix mining.
  Columns used: ``audio``, ``endpoint_bool``, ``language`` and ``dataset`` (the contributing
  source, used for grouped splits since the corpus has no speaker ids).
- ``MinedDataset``: torch Dataset over a ``samples.jsonl`` produced by ``eot-mine``.

Split policy: hold out *whole sources* (``dataset`` column) for dev so a high score cannot come
from recognising a voice or a TTS vocoder. If only one source exists, fall back to a split by
``clip_id`` so prefixes of the same clip never straddle train/dev.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import SAMPLE_RATE, log_mel, resample, telephony_augment, to_float32, to_mono
from .context import CTX_LEN, hash_context
from .prefix_mining import HORIZONS, Clip

SMART_TURN_TRAIN = "pipecat-ai/smart-turn-data-v3.2-train"
SMART_TURN_TEST = "pipecat-ai/smart-turn-data-v3.2-test"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_audio_filename(identifier: str) -> str:
    """Make a readable filename while hashing the full id to avoid collisions/traversal."""
    raw = str(identifier)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")[:64] or "audio"
    return f"{slug}-{hashlib.sha256(raw.encode()).hexdigest()[:12]}.wav"


def resolve_record_path(manifest: Path, value: str | Path) -> Path:
    """Resolve an artifact path relative to the JSONL file that contains it."""
    path = Path(value)
    if not path.is_absolute():
        path = Path(manifest).parent / path
    return path.resolve()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Durably replace a JSON file without exposing a partially-written version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_jsonl_records(path: Path, repair_trailing: bool = False) -> list[dict]:
    """Read JSONL and optionally discard/rewrite only a torn final record."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_text()
    lines = raw.splitlines()
    rows: list[dict] = []
    repaired = bool(raw and not raw.endswith("\n"))
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if repair_trailing and index == len(lines) - 1:
                repaired = True
                break
            raise
    if repaired and repair_trailing:
        atomic_write_jsonl(path, rows)
    return rows


def append_jsonl_record(path: Path, row: dict) -> None:
    """Append one complete record and fsync it before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(f"short JSONL append: wrote {written} of {len(payload)} bytes")
        os.fsync(fd)
    finally:
        os.close(fd)


def _decode_audio(a) -> tuple[np.ndarray, int]:
    if isinstance(a, dict) and "array" in a and a["array"] is not None:
        return np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"])
    if isinstance(a, dict) and a.get("bytes") is not None:
        import io

        import soundfile as sf

        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
        return np.asarray(x), int(sr)
    if isinstance(a, dict) and a.get("path"):
        import soundfile as sf

        x, sr = sf.read(a["path"], dtype="float32", always_2d=False)
        return np.asarray(x), int(sr)
    raise ValueError("unsupported audio payload")


def load_smart_turn_clips(
    name: str = SMART_TURN_TRAIN,
    language: str = "eng",
    per_label: int = 8_000,
    seed: int = 0,
    streaming: bool = True,
    revision: str | None = None,
    max_rows: int | None = None,
) -> Iterator[Clip]:
    """Yield up to ``per_label`` clips of each label for ``language`` from the Smart Turn corpus."""
    from datasets import Audio, load_dataset

    ds = load_dataset(name, split="train", streaming=streaming, revision=revision)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=seed, buffer_size=2_000) if streaming else ds.shuffle(seed=seed)
    taken = {0: 0, 1: 0}
    for i, row in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        if str(row.get("language", "")) != language:
            continue
        label = int(bool(row["endpoint_bool"]))
        if taken[label] >= per_label:
            if all(v >= per_label for v in taken.values()):
                break
            continue
        x, sr = _decode_audio(row["audio"])
        x = resample(to_mono(to_float32(x)), sr)
        taken[label] += 1
        yield Clip(
            id=str(row.get("id", i)), audio=x, label=label,
            source=str(row.get("dataset", "unknown")), agent_text=str(row.get("agent_text", "") or ""),
        )


def split_diagnostics(
    train: list[dict], dev: list[dict], *, group_key: str, requested_group_key: str | None = None
) -> dict:
    """Summarize split balance and prove that groups do not cross the boundary."""
    def describe(part: list[dict]) -> dict:
        labels = Counter(int(row["label"]) for row in part)
        return {
            "n": len(part),
            "labels": {str(k): labels.get(k, 0) for k in (0, 1)},
            "groups": len({str(row.get(group_key, "unknown")) for row in part}),
        }

    train_groups = {str(row.get(group_key, "unknown")) for row in train}
    dev_groups = {str(row.get(group_key, "unknown")) for row in dev}
    return {
        "group_key": group_key,
        "requested_group_key": requested_group_key or group_key,
        "train": describe(train),
        "dev": describe(dev),
        "group_overlap": sorted(train_groups & dev_groups),
    }


def grouped_split(
    rows: list[dict],
    dev_frac: float = 0.15,
    seed: int = 0,
    group_key: str = "source",
    *,
    return_diagnostics: bool = False,
) -> tuple[list[dict], list[dict]] | tuple[list[dict], list[dict], dict]:
    """Make a deterministic grouped split while avoiding a dominant-group overshoot.

    The split prioritizes retaining both labels on both sides when the group structure permits it.
    If there is only one source group it falls back to clip ids, never individual prefixes.
    """
    if not 0.0 < dev_frac < 1.0:
        raise ValueError("dev_frac must be between zero and one")
    if len(rows) < 2:
        raise ValueError("at least two rows are required for a train/dev split")
    rng = random.Random(seed)
    requested_group_key = group_key
    source_groups = {str(row.get(group_key, "unknown")) for row in rows}
    if len(source_groups) < 2:
        group_key = "clip_id"
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[str(r.get(group_key, "unknown"))].append(r)
    if len(by_group) < 2:
        raise ValueError(f"cannot split: only one {group_key!r} group")

    group_names = sorted(by_group)
    rng.shuffle(group_names)
    target = max(1, min(len(rows) - 1, round(dev_frac * len(rows))))
    total_labels = Counter(int(row["label"]) for row in rows)
    group_label_counts = {
        name: Counter(int(row["label"]) for row in by_group[name]) for name in group_names
    }

    def leaves_train_labels(names: set[str]) -> bool:
        selected = sum((group_label_counts[name] for name in names), Counter())
        return all(total_labels[label] - selected[label] > 0 for label in (0, 1))

    # Seed dev with both labels when possible, using small groups so a dominant source cannot
    # consume the entire validation budget on its own.
    selected: set[str] = set()
    mixed = [
        name for name in group_names
        if set(group_label_counts[name]) >= {0, 1} and leaves_train_labels({name})
    ]
    if set(total_labels) >= {0, 1}:
        zero_groups = sorted(
            (name for name in group_names if group_label_counts[name][0] and total_labels[0] > group_label_counts[name][0]),
            key=lambda name: len(by_group[name]),
        )
        one_groups = sorted(
            (name for name in group_names if group_label_counts[name][1] and total_labels[1] > group_label_counts[name][1]),
            key=lambda name: len(by_group[name]),
        )
        for zero_name in zero_groups:
            one_name = next((name for name in one_groups if name != zero_name), None)
            if one_name is not None and leaves_train_labels({zero_name, one_name}):
                selected.update((zero_name, one_name))
                break

    # Prefer two small single-label groups over one dominant mixed source. If that is not
    # feasible, a mixed group still gives both labels without crossing a group boundary.
    if not selected and mixed:
        selected.add(min(mixed, key=lambda name: (abs(len(by_group[name]) - target), len(by_group[name]))))

    if not selected:
        selected.add(min(group_names, key=lambda name: (abs(len(by_group[name]) - target), len(by_group[name]))))

    # Add only groups that improve closeness to the target. This is the key difference from the
    # former "keep adding until >= target" algorithm, which could overshoot by a dominant source.
    while True:
        current = sum(len(by_group[name]) for name in selected)
        candidates = []
        for name in group_names:
            if name in selected:
                continue
            proposed = selected | {name}
            if set(total_labels) >= {0, 1} and not leaves_train_labels(proposed):
                continue
            candidates.append(name)
        if not candidates:
            break
        best = min(candidates, key=lambda name: (abs(current + len(by_group[name]) - target), len(by_group[name])))
        if abs(current + len(by_group[best]) - target) >= abs(current - target):
            break
        selected.add(best)

    train = [row for row in rows if str(row.get(group_key, "unknown")) not in selected]
    dev = [row for row in rows if str(row.get(group_key, "unknown")) in selected]
    achieved_dev_frac = len(dev) / len(rows)
    if (
        group_key == requested_group_key
        and group_key != "clip_id"
        and achieved_dev_frac > max(2.0 * dev_frac, dev_frac + 0.10)
    ):
        # With only a few large sources, strict source grouping can discard an
        # unreasonable fraction of training data. Clip grouping still prevents
        # prefixes of one utterance from leaking across the boundary.
        clip_train, clip_dev, clip_diagnostics = grouped_split(
            rows,
            dev_frac=dev_frac,
            seed=seed,
            group_key="clip_id",
            return_diagnostics=True,
        )
        clip_diagnostics["requested_group_key"] = requested_group_key
        clip_diagnostics["fallback_reason"] = (
            f"source-grouped dev fraction {achieved_dev_frac:.4f} exceeded guardrail"
        )
        if return_diagnostics:
            return clip_train, clip_dev, clip_diagnostics
        return clip_train, clip_dev
    diagnostics = split_diagnostics(
        train, dev, group_key=group_key, requested_group_key=requested_group_key
    )
    if diagnostics["group_overlap"]:
        raise AssertionError("grouped split leaked groups across train and dev")
    if return_diagnostics:
        return train, dev, diagnostics
    return train, dev


def read_samples(path: Path) -> list[dict]:
    path = Path(path)
    rows = read_jsonl_records(path)
    for row in rows:
        row["path"] = str(resolve_record_path(path, row["path"]))
    return rows


class MinedDataset(Dataset):
    """Samples from ``eot-mine``; computes log-mel on the fly (cheap relative to the encoder)."""

    def __init__(self, rows: list[dict], telephony_prob: float = 0.0, seed: int = 0):
        self.rows = rows
        self.telephony_prob = telephony_prob
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_rng(self, i: int) -> np.random.Generator:
        # Independent of worker assignment/prefetch, but different across epochs.
        identity = str(self.rows[i].get("id", self.rows[i].get("clip_id", i)))
        key = f"{self.seed}:{self.epoch}:{identity}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        return np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        import soundfile as sf

        r = self.rows[i]
        x, sr = sf.read(r["path"], dtype="float32", always_2d=False)
        x = resample(to_mono(to_float32(np.asarray(x))), sr)
        rng = self._sample_rng(i)
        if self.telephony_prob > 0 and rng.random() < self.telephony_prob:
            x = telephony_augment(x, SAMPLE_RATE, rng)
        return {
            "input_features": torch.from_numpy(log_mel(x)),
            "context_ids": torch.from_numpy(hash_context(r.get("agent_text", ""))),
            "labels": torch.tensor(int(r["label"])),
            "fvad": torch.tensor(r.get("fvad", [0] * len(HORIZONS)), dtype=torch.float32),
            "fvad_mask": torch.tensor(int(r.get("fvad_mask", 0))),
        }


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def clips_to_manifest(clips: Iterable[Clip], out_dir: Path) -> Path:
    """Persist raw clips to wav + jsonl so mining/training are reproducible offline."""
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "clips.jsonl"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    rows = []
    for c in clips:
        p = audio_dir / safe_audio_filename(c.id)
        sf.write(p, c.audio, SAMPLE_RATE, subtype="PCM_16")
        rows.append({
            "id": c.id,
            "path": p.relative_to(out_dir).as_posix(),
            "label": c.label,
            "source": c.source,
            "agent_text": c.agent_text,
            "sha256": sha256_file(p),
        })
    atomic_write_jsonl(manifest, rows)
    return manifest


__all__ = [
    "CTX_LEN", "MinedDataset", "append_jsonl_record", "atomic_write_json",
    "atomic_write_jsonl", "clips_to_manifest", "collate", "grouped_split",
    "load_smart_turn_clips", "read_jsonl_records", "read_samples", "resolve_record_path",
    "safe_audio_filename", "sha256_file", "split_diagnostics",
]
