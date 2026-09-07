"""Tail texture rewrites only the pause tail, never the speech, and leaves untouched rows bit-identical."""

from __future__ import annotations

import numpy as np
import pytest

from eot.audio import SAMPLE_RATE, frame_db, noise_floor_dbfs, tail_texture
from eot.data.dataset import MinedDataset, WaveformDataset, tail_length_s
from eot.io import atomic_write_jsonl, write_wav


def _speech_then_pause(seconds: float = 1.0, tail_s: float = 0.4, floor_dbfs: float = -50.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    x = 0.3 * np.sin(np.arange(n) * 0.05).astype(np.float32)
    x += (rng.normal(0, 1, n) * 10 ** (floor_dbfs / 20)).astype(np.float32)
    x[-int(tail_s * SAMPLE_RATE) :] = 0.0  # gated tail as stored by VoIP sources
    return x


def test_gated_and_room_noise_tails():
    x = _speech_then_pause()
    cut = len(x) - int(0.4 * SAMPLE_RATE)
    gated, info = tail_texture(x, 0.4, np.random.default_rng(1), gated_prob=1.0)
    assert info["kind"] == "gated"
    np.testing.assert_array_equal(gated[:cut], x[:cut])
    assert not gated[cut:].any()
    noisy, info = tail_texture(x, 0.4, np.random.default_rng(1), gated_prob=0.0)
    assert info["kind"] == "room_noise"
    np.testing.assert_array_equal(noisy[:cut], x[:cut])
    tail_db = frame_db(noisy[cut:])
    assert abs(float(np.median(tail_db)) - info["floor_dbfs"]) < 6.0  # pink noise scaled to the estimated floor
    assert abs(info["floor_dbfs"] - noise_floor_dbfs(x, cut / SAMPLE_RATE)) < 1e-9
    assert noisy.dtype == np.float32 and np.abs(noisy).max() <= 1.0
    with pytest.raises(ValueError):
        tail_texture(x, -0.1, np.random.default_rng(0))


def test_texture_is_deterministic_per_generator_and_label_independent():
    x = _speech_then_pause()
    a, _ = tail_texture(x, 0.4, np.random.default_rng(7))
    b, _ = tail_texture(x, 0.4, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_noise_floor_falls_back_when_no_audible_frames():
    silence = np.zeros(SAMPLE_RATE, np.float32)
    assert noise_floor_dbfs(silence, 1.0) == -60.0


@pytest.fixture
def manifest(tmp_path):
    write_wav(tmp_path / "a.wav", _speech_then_pause(seed=1), SAMPLE_RATE)
    write_wav(tmp_path / "b.wav", _speech_then_pause(seed=2), SAMPLE_RATE)
    rows = [
        {"id": "a", "path": str(tmp_path / "a.wav"), "label": 1, "source": "s", "noise_fill": 1, "cut_time": 3.0, "pause_start": 2.6},
        {"id": "b", "path": str(tmp_path / "b.wav"), "label": 0, "source": "s", "noise_fill": 0, "cut_time": 5.0, "pause_start": 4.6},
    ]
    atomic_write_jsonl(tmp_path / "samples.jsonl", rows)
    return rows


def test_dataset_untouched_rows_are_bit_identical_and_touched_rows_change_only_the_tail(manifest):
    control = MinedDataset(manifest, telephony_prob=0.3, seed=5)
    never = MinedDataset(manifest, telephony_prob=0.3, seed=5, tail_texture_prob=0.0)
    always = MinedDataset(manifest, telephony_prob=0.3, seed=5, tail_texture_prob=1.0)
    for i in range(len(manifest)):
        np.testing.assert_array_equal(control.waveform(i), never.waveform(i))
        base, textured = control.waveform(i), always.waveform(i)
        cut = len(base) - int(round(tail_length_s(manifest[i]) * SAMPLE_RATE))
        np.testing.assert_array_equal(textured[:cut], base[:cut])  # room tone / telephony draws unchanged
        assert not np.array_equal(textured[cut:], base[cut:]) or not textured[cut:].any()
    # A coin miss must reproduce the control exactly: the texture stream is separate from the base stream.
    prob = MinedDataset(manifest, telephony_prob=0.3, seed=5, tail_texture_prob=0.5)
    for i in range(len(manifest)):
        w = prob.waveform(i)
        if np.array_equal(w, control.waveform(i)):
            continue
        cut = len(w) - int(round(tail_length_s(manifest[i]) * SAMPLE_RATE))
        np.testing.assert_array_equal(w[:cut], control.waveform(i)[:cut])


def test_dataset_requires_tail_fields_only_when_texture_is_enabled(manifest):
    rows = [{k: v for k, v in manifest[0].items() if k not in ("cut_time", "pause_start")}]
    MinedDataset(rows)
    with pytest.raises(ValueError, match="cut_time"):
        MinedDataset(rows, tail_texture_prob=0.5)
    with pytest.raises(ValueError):
        MinedDataset(manifest, tail_texture_prob=1.5)


def test_waveform_dataset_shares_the_stream_and_returns_the_last_window(manifest):
    mel = MinedDataset(manifest, telephony_prob=0.3, seed=9, tail_texture_prob=1.0)
    wave = WaveformDataset(manifest, telephony_prob=0.3, seed=9, tail_texture_prob=1.0, input_s=2.0)
    item = wave[0]
    assert item["wave"].shape == (2 * SAMPLE_RATE,) and item["wave"].dtype.is_floating_point
    expected = mel.waveform(0)
    np.testing.assert_array_equal(item["wave"].numpy()[-len(expected) :], expected)
    assert not item["wave"].numpy()[: 2 * SAMPLE_RATE - len(expected)].any()
    assert set(item) == {"wave", "labels", "fvad", "fvad_mask", "weight"}
