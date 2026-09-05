"""Small, durable file helpers shared by every stage (torch-free).

JSONL manifests are the interchange format between acquisition, labeling, training and
evaluation. Publishing writes go through one primitive, ``atomic_replace`` (sibling temp file +
rename), so a crash never leaves a torn file. Per-record fsync is opt-in through
``EOT_DURABLE_WRITES=1`` because it throttles mining on shared disks to a few rows per second.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

DURABLE_WRITES = os.environ.get("EOT_DURABLE_WRITES", "0") == "1"


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_from_key(key: str) -> int:
    """Deterministic 64-bit seed from a string key (per-sample RNG streams, matched noise pads)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def safe_audio_filename(identifier: str) -> str:
    """Readable filename with the full id hashed in, so ids can never collide or escape the directory."""
    raw = str(identifier)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")[:64] or "audio"
    return f"{slug}-{hashlib.sha256(raw.encode()).hexdigest()[:12]}.wav"


def resolve_record_path(manifest: Path, value: str | Path) -> Path:
    """Resolve an artifact path relative to the JSONL file that contains it."""
    path = Path(value)
    if not path.is_absolute():
        path = Path(manifest).parent / path
    return path.resolve()


def atomic_replace(path: Path, write: Callable[[Path], None], *, fsync: bool = True) -> None:
    """Let ``write`` fill a sibling temp file, optionally fsync it, then publish it with a rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        if fsync:
            with tmp.open("rb") as f:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_replace(path, lambda tmp: tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"))


def atomic_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    def write(tmp: Path) -> None:  # stream: manifests reach 100k rows, never hold them as one string
        with tmp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    atomic_replace(path, write)


def atomic_copy(source: Path, destination: Path) -> None:
    atomic_replace(destination, lambda tmp: shutil.copyfile(source, tmp))


def read_jsonl_records(path: Path, repair_trailing: bool = False) -> list[dict]:
    """Read JSONL; with ``repair_trailing`` a torn final record is dropped and the file rewritten."""
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


def read_samples(path: Path) -> list[dict]:
    """Manifest rows with ``path`` made absolute relative to the manifest, so callers never depend on cwd.

    The manifest directory is resolved once; rows are joined without per-row syscalls. Rows are for
    reading: write manifests back with the relative paths from ``read_jsonl_records``.
    """
    path = Path(path)
    rows = read_jsonl_records(path)
    base = str(path.parent.resolve())
    for row in rows:
        value = str(row["path"])
        row["path"] = os.path.normpath(value if os.path.isabs(value) else os.path.join(base, value))
    return rows


def append_jsonl_record(path: Path, row: dict) -> None:
    """Append one complete record (fsync only under ``EOT_DURABLE_WRITES``)."""
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


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """PCM16 bytes for mono float32 audio; other shapes/dtypes are rejected instead of silently
    clipped or written multi-channel (normalise with ``eot.audio.to_16k`` first)."""
    import soundfile as sf

    audio = np.asarray(audio)
    if audio.ndim != 1 or not np.issubdtype(audio.dtype, np.floating):
        raise ValueError(f"write_wav expects a 1-D float waveform, got shape {audio.shape} dtype {audio.dtype}")
    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32, copy=False), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> str:
    """Plain PCM16 write into an existing directory; returns the file's SHA-256.

    Used by parallel labelers: thousands of concurrent temp+rename operations from many workers
    saturated NVMe metadata throughput, and those paths have no resume to protect anyway.
    """
    data = _encode_wav(audio, sample_rate)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def atomic_write_wav(path: Path, audio: np.ndarray, sample_rate: int, *, fsync: bool = DURABLE_WRITES) -> str:
    """PCM16 write published atomically; returns the SHA-256.

    Mining keeps the ``EOT_DURABLE_WRITES`` default; acquisition passes ``fsync=True`` so a clip is
    on disk before its manifest record (a torn clip with a live record cannot be resumed).
    """
    data = _encode_wav(audio, sample_rate)
    atomic_replace(path, lambda tmp: tmp.write_bytes(data), fsync=fsync)
    return hashlib.sha256(data).hexdigest()
