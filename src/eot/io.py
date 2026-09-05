"""Small, durable file helpers shared by every stage (torch-free).

JSONL manifests are the interchange format between acquisition, labeling, training and
evaluation. Writes are atomic (temp file + rename) so a crash never leaves a torn manifest;
per-record fsync is opt-in through ``EOT_DURABLE_WRITES=1`` because it throttles mining on
shared disks to a few rows per second.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# fsync every record/wav only when asked: on a shared local disk two fsyncs per sample throttle
# mining to ~5 rows/s. Torn tails are repaired on read either way (``repair_trailing``).
DURABLE_WRITES = os.environ.get("EOT_DURABLE_WRITES", "0") == "1"


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


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16_000, *, atomic: bool = True) -> None:
    """Write mono float32 audio as PCM16.

    ``atomic`` publishes through a temp file + rename (fsync when ``EOT_DURABLE_WRITES=1``).
    Parallel mining passes ``atomic=False``: thousands of concurrent tmp+rename operations from
    many workers saturated NVMe metadata throughput and stalled every writer.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(audio, dtype=np.float32)
    if not atomic:
        sf.write(path, mono, sample_rate, subtype="PCM_16")
        return
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.partial.wav")
    try:
        sf.write(tmp, mono, sample_rate, subtype="PCM_16")
        if DURABLE_WRITES:
            with tmp.open("rb") as f:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
        if DURABLE_WRITES:
            os.fsync(fd)
    finally:
        os.close(fd)
