"""Manifest and wav I/O invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eot.io import read_samples, write_wav


def test_write_wav_rejects_int16_and_multichannel(tmp_path: Path):
    with pytest.raises(ValueError, match="float"):
        write_wav(tmp_path / "a.wav", np.arange(-3000, 3000, dtype=np.int16), 16000)
    with pytest.raises(ValueError, match="1-D"):
        write_wav(tmp_path / "b.wav", np.zeros((100, 2), dtype=np.float32), 16000)
    assert write_wav(tmp_path / "c.wav", np.zeros(100, dtype=np.float64), 16000)


def test_read_samples_joins_without_touching_rows_on_disk(tmp_path: Path):
    manifest = tmp_path / "m" / "samples.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"id": "x", "path": "audio/../audio/x.wav"}) + "\n")
    rows = read_samples(manifest)
    assert rows[0]["path"] == str(tmp_path / "m" / "audio" / "x.wav")
