"""Cached teacher logits for exact-view distillation (numpy only; the producer lives in ``eot.modeling.distill``).

A distillation run trains the student on the *same realised waveform* the teacher scored: room
tone, telephony leg and tail texture are drawn per row from seeded streams, so the teacher can be
run once, offline, over the training split at a fixed epoch and its logits stored next to a hash
of every waveform it saw. At training time ``DistillationDataset`` regenerates the waveform,
compares its hash with the cached one and refuses to pair a target with a different view.

Layout of a cache directory::

    identity.json          what the cache was built for (rows digest, seed, epoch, view, teacher)
    shards/NNNNNNNN.npz    contiguous [start, end) blocks written atomically; resume continues from the last one
    logits.npy             float32 [n_rows]           (after ``CacheWriter.finish``)
    waveform_sha256.npy    S64 ascii hex [n_rows]
    row_ids.jsonl          the row ids in cache order
    metadata.json          status COMPLETE, fingerprint, file digests

The student always trains on float32 targets; nothing here touches labels, splits or weights.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from eot.audio import SAMPLE_RATE, WINDOW_SECONDS, last_window
from eot.data.dataset import MinedDataset
from eot.io import atomic_replace, atomic_write_json, sha256_file

CACHE_VERSION = "eot-teacher-cache-v1"
CACHE_FILES = ("logits.npy", "waveform_sha256.npy", "row_ids.jsonl")


def canonical_json(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data) -> str:
    return hashlib.sha256(canonical_json(data).encode()).hexdigest()


def waveform_sha256(wave: np.ndarray, seconds: float = WINDOW_SECONDS) -> str:
    """SHA-256 of a finite float32 window of exactly ``seconds`` at 16 kHz (little-endian bytes)."""
    wave = np.asarray(wave)
    n = int(seconds * SAMPLE_RATE)
    if wave.shape != (n,) or wave.dtype != np.float32 or not np.isfinite(wave).all():
        raise ValueError(f"expected a finite float32 waveform of {n} samples, got {wave.dtype} {wave.shape}")
    return hashlib.sha256(np.ascontiguousarray(wave, dtype="<f4").tobytes()).hexdigest()


def rows_digest(rows: list[dict]) -> str:
    """Order-sensitive digest of the row ids and labels: the cache belongs to exactly this ordered split."""
    h = hashlib.sha256()
    for row in rows:
        h.update(canonical_json({"id": str(row["id"]), "label": int(row["label"])}).encode() + b"\n")
    return h.hexdigest()


def view_identity(dataset: MinedDataset) -> dict:
    """The augmentation configuration that, with the row id, seed and epoch, determines a waveform."""
    return {
        "seed": dataset.seed,
        "epoch": dataset.epoch,
        "noise_fill": dataset.noise_fill,
        "telephony_prob": dataset.telephony_prob,
        "tail_texture_prob": dataset.tail_texture_prob,
        "input_s": float(WINDOW_SECONDS),  # the hashed window; kept in the identity so older caches still match
    }


def cache_identity(rows: list[dict], dataset: MinedDataset, teacher: dict) -> dict:
    return {
        "version": CACHE_VERSION,
        "n_rows": len(rows),
        "rows_sha256": rows_digest(rows),
        "view": view_identity(dataset),
        "teacher": dict(teacher),
    }


def _save_array(path: Path, save) -> None:
    """Publish a numpy file atomically; ``save`` receives an open binary handle (numpy would append its own suffix to a name)."""

    def write(tmp: Path) -> None:
        with tmp.open("wb") as f:
            save(f)

    atomic_replace(path, write)


class CacheWriter:
    """Single-writer, resumable shard writer; ``finish`` assembles the arrays and publishes ``metadata.json``.

    The next shard start is scanned once at construction (verifying every existing shard) and then kept
    in memory, so appends do not re-hash the shards written so far.
    """

    def __init__(self, path: Path, identity: dict, rows: list[dict]):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.identity, self.fingerprint, self.rows = identity, digest(identity), rows
        self._lock = (self.path / "cache.lock").open("a")
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = self.path / "identity.json"
        if existing.exists():
            if json.loads(existing.read_text()) != identity:
                raise ValueError(f"{self.path} holds a cache for a different view, split or teacher")
        else:
            atomic_write_json(existing, identity)
        self.shards = self.path / "shards"
        self.shards.mkdir(exist_ok=True)
        self._next_start = self._scan_next_start()

    def close(self) -> None:
        self._lock.close()

    def __enter__(self) -> CacheWriter:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def completed(self) -> list[tuple[dict, Path]]:
        """Verified, contiguous shards written so far (in order)."""
        result, start = [], 0
        for meta_path in sorted(self.shards.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            array = meta_path.with_suffix(".npz")
            if meta["fingerprint"] != self.fingerprint or meta["start"] != start or not start < meta["end"] <= len(self.rows):
                raise ValueError(f"shard {meta_path.name} is out of order or belongs to another cache")
            if sha256_file(array) != meta["sha256"]:
                raise ValueError(f"shard {array.name} is corrupt")
            result.append((meta, array))
            start = meta["end"]
        return result

    def _scan_next_start(self) -> int:
        done = self.completed()
        return done[-1][0]["end"] if done else 0

    def next_start(self) -> int:
        """Row index the next shard must start at (``len(rows)`` once every row is cached)."""
        return self._next_start

    def append(self, start: int, logits: np.ndarray, waveform_hashes: list[str]) -> None:
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        if not np.isfinite(logits).all():
            raise ValueError("teacher logits must be finite")
        hashes = np.asarray(waveform_hashes, dtype="S64")
        if hashes.shape != logits.shape or any(len(h) != 64 for h in hashes.tolist()):
            raise ValueError("one 64-hex waveform hash is required per logit")
        if start != self._next_start:
            raise ValueError(f"shard must start at {self._next_start}, got {start}")
        array = self.shards / f"{start:08d}.npz"
        _save_array(array, lambda f: np.savez(f, logits=logits, hashes=hashes))
        atomic_write_json(
            array.with_suffix(".json"),
            {"start": start, "end": start + len(logits), "sha256": sha256_file(array), "fingerprint": self.fingerprint},
        )
        self._next_start = start + len(logits)

    def finish(self) -> dict:
        shards = self.completed()
        if not shards or shards[-1][0]["end"] != len(self.rows):
            raise ValueError("cannot finish a partial teacher cache")
        logits = np.empty(len(self.rows), dtype=np.float32)
        hashes = np.empty(len(self.rows), dtype="S64")
        for meta, array in shards:
            with np.load(array, allow_pickle=False) as z:
                logits[meta["start"] : meta["end"]] = z["logits"]
                hashes[meta["start"] : meta["end"]] = z["hashes"]
        for name, data in (("logits.npy", logits), ("waveform_sha256.npy", hashes)):
            _save_array(self.path / name, lambda f, data=data: np.save(f, data))
        atomic_replace(self.path / "row_ids.jsonl", lambda tmp: tmp.write_text("".join(json.dumps(str(r["id"])) + "\n" for r in self.rows)))
        metadata = {
            "status": "COMPLETE",
            "version": CACHE_VERSION,
            "n_rows": len(self.rows),
            "identity": self.identity,
            "fingerprint": self.fingerprint,
            "files": {name: sha256_file(self.path / name) for name in CACHE_FILES},
            "completed_unix": time.time(),
        }
        atomic_write_json(self.path / "metadata.json", metadata)
        return metadata


def load_cache(path: Path, rows: list[dict], identity: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Validate a complete cache against ``rows`` (and ``identity`` when given); returns ``(logits, hashes, metadata)``."""
    path = Path(path)
    if not (path / "metadata.json").exists():
        raise FileNotFoundError(f"{path} has no metadata.json: the teacher cache is incomplete")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("status") != "COMPLETE" or metadata.get("fingerprint") != digest(metadata.get("identity")):
        raise ValueError(f"{path} is not a complete teacher cache")
    if identity is not None and metadata["identity"] != identity:
        raise ValueError("teacher cache identity differs from the requested split, view or teacher")
    if set(metadata["files"]) != set(CACHE_FILES) or any(sha256_file(path / n) != metadata["files"][n] for n in CACHE_FILES):
        raise ValueError(f"{path}: cache file integrity mismatch")
    logits = np.load(path / "logits.npy", mmap_mode="r", allow_pickle=False)
    hashes = np.load(path / "waveform_sha256.npy", mmap_mode="r", allow_pickle=False)
    n = len(rows)
    if logits.shape != (n,) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("cached logits have the wrong shape or are not finite")
    if hashes.shape != (n,) or hashes.dtype != np.dtype("S64"):
        raise ValueError("cached waveform hashes have the wrong shape")
    ids = [json.loads(line) for line in (path / "row_ids.jsonl").read_text().splitlines()]
    if ids != [str(r["id"]) for r in rows]:
        raise ValueError("teacher cache rows differ from the training rows (identity or order)")
    return logits, hashes, metadata


class DistillationDataset(MinedDataset):
    """``MinedDataset`` plus ``teacher_logits`` from a cache built on the same view.

    Every item regenerates its waveform and checks the hash the teacher saw, so a cache can never be
    consumed with a different seed, epoch, augmentation or manifest. Only the cached epoch is valid.
    """

    def __init__(self, rows: list[dict], cache_dir: Path, *, epoch: int = 0, **kwargs):
        super().__init__(rows, **kwargs)
        self.epoch = int(epoch)
        self.logits, self.hashes, self.metadata = load_cache(cache_dir, rows)
        view = self.metadata["identity"]["view"]
        if view != view_identity(self):
            raise ValueError(f"teacher cache view {view} differs from the dataset view {view_identity(self)}")

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) != self.epoch:
            raise ValueError(f"teacher cache was built for epoch {self.epoch}; build one cache per epoch")

    def __getitem__(self, i: int) -> dict:
        wave = self.waveform(i)
        expected = bytes(self.hashes[i]).decode("ascii")
        if waveform_sha256(last_window(wave, WINDOW_SECONDS).astype(np.float32, copy=False)) != expected:
            raise ValueError(f"row {self.rows[i].get('id')!r}: waveform differs from the one the teacher scored")
        item = self._item(i, wave)
        item["teacher_logits"] = torch.tensor(float(self.logits[i]), dtype=torch.float32)
        return item
