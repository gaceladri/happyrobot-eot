"""Held-out scoring and summaries: score-point selection and the room-tone RNG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eot.audio import SAMPLE_RATE
from eot.eval.report import room_tone_seed, score_heldout, summarise_heldout
from eot.io import atomic_write_jsonl, write_wav

SR = SAMPLE_RATE


class _EnergyAdapter:
    adapter_id = "energy"

    def predict_batch(self, items):
        return [float(np.mean(np.abs(item["audio"]["array"]))) for item in items]


def test_room_tone_rng_is_keyed_by_sample_id_not_row_position(tmp_path: Path):
    # Speech followed by digital silence: add_room_tone scales its noise to the speech power and is
    # a no-op on an all-zero signal, so the fixture needs an audible stretch.
    wav = tmp_path / "gated.wav"
    x = np.zeros(SR, np.float32)
    x[: SR // 2] = 0.1 * np.sin(2 * np.pi * 220 * np.arange(SR // 2) / SR)
    write_wav(wav, x, SR)
    rows = [
        {"id": "later-cut", "path": wav.name, "label": 1, "noise_fill": 1, "cut_time": 1.4, "pause_start": 1.0},
        {"id": "score-point", "path": wav.name, "label": 0, "noise_fill": 1, "cut_time": 1.2, "pause_start": 1.0},
    ]
    manifest = tmp_path / "samples.jsonl"
    atomic_write_jsonl(manifest, rows)
    only = score_heldout(manifest, _EnergyAdapter(), score_point_only=True, seed=3)
    every = score_heldout(manifest, _EnergyAdapter(), score_point_only=False, seed=3)
    assert [r["id"] for r in only] == ["score-point"] and [r["id"] for r in every] == ["later-cut", "score-point"]
    assert only[0]["p_eot"] == every[1]["p_eot"]  # same sample, different row position, same room tone
    assert every[0]["p_eot"] != every[1]["p_eot"]  # different samples draw different room tone
    other_seed = score_heldout(manifest, _EnergyAdapter(), score_point_only=True, seed=4)
    assert other_seed[0]["p_eot"] != only[0]["p_eot"]
    assert room_tone_seed(3, "score-point") != room_tone_seed(3, "later-cut")
    assert room_tone_seed(3, "score-point") == room_tone_seed(3, "score-point")


def test_summarise_heldout_refuses_to_score_off_point_rows_as_the_score_point():
    later = [
        {"label": 1, "cut_time": 1.4, "pause_start": 1.0, "p_eot": 0.9},
        {"label": 0, "cut_time": 1.6, "pause_start": 1.0, "p_eot": 0.1},
    ]
    with pytest.raises(ValueError, match="score point"):
        summarise_heldout(later)
    assert summarise_heldout(later, score_point_only=False)["n"] == 2
    at_point = later + [
        {"label": 1, "cut_time": 1.2, "pause_start": 1.0, "p_eot": 0.8},
        {"label": 0, "cut_time": 1.2, "pause_start": 1.0, "p_eot": 0.2},
    ]
    assert summarise_heldout(at_point)["n"] == 2
