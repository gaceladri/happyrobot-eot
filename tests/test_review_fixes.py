"""Regression tests for the /code-review findings on the unified package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import tiny_model
from eot.io import read_samples, write_wav
from eot.labeling.turn_events import Event, label_pause
from eot.modeling.soup import resolve_split


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


def test_krisp_metrics_fails_loudly_on_missing_predictions(tmp_path: Path):
    from eot.eval.krisp import metrics

    with pytest.raises(FileNotFoundError):
        metrics(tmp_path / "missing.jsonl", bootstrap_reps=1)


def test_export_rejects_missing_or_empty_validation_manifest(tmp_path: Path):
    from eot.modeling.export_onnx import _parity_inputs

    rng = np.random.default_rng(0)
    with pytest.raises(FileNotFoundError):
        _parity_inputs(tiny_model(), tmp_path / "missing.jsonl", rng)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        _parity_inputs(tiny_model(), empty, rng)


def test_own_annotation_spanning_listener_floor_abstains():
    target = label_pause(
        pause_start=5.0, cut_time=5.2, speaker="s1",
        events=[Event(3.0, 7.0, "s1", "Normal Turn"), Event(5.4, 6.0, "s2", "Bounded Response")],
        observed_until=12.0, next_own_onset=9.0,
    )
    assert target.label is None and target.reason == "own_annotation_spans_transfer"


def test_resolve_split_follows_parents_and_refuses_mismatches():
    a = {"split_seed": 0, "dev_frac": 0.15, "samples_sha256": "s"}
    b = {"split_seed": 0, "dev_frac": 0.15, "samples_sha256": "s"}
    assert resolve_split([a, b], None, None, "s") == (0, 0.15)
    assert resolve_split([a, None], None, None, "s") == (0, 0.15)
    with pytest.raises(ValueError, match="different split_seed"):
        resolve_split([a, {**b, "split_seed": 17}], None, None, "s")
    with pytest.raises(ValueError, match="requested split_seed=17"):
        resolve_split([a, b], 17, None, "s")
    with pytest.raises(ValueError, match="samples manifest"):
        resolve_split([a, b], None, None, "other")
    with pytest.raises(ValueError, match="--split-seed"):
        resolve_split([None, None], None, None, "s")
    assert resolve_split([None, None], 3, 0.2, "s") == (3, 0.2)
