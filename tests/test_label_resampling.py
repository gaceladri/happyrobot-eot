import numpy as np
import pytest

from eot.audio import causal_prefix, resample, resample_antialiased


def tone(hz):
    return np.sin(2 * np.pi * hz * np.arange(48000) / 48000).astype(np.float32)


def rms(x):
    return np.sqrt(np.mean(x[800:-800] ** 2))


def test_stopband_is_rejected_instead_of_aliasing_into_speech():
    x = tone(12000)
    assert rms(resample(x, 48000)) > 0.7  # linear serving resampler aliases
    assert rms(resample_antialiased(x, 48000)) < 0.007  # > 40 dB attenuation


def test_speech_passband_and_duration_are_preserved():
    y = resample_antialiased(tone(1000), 48000)
    assert len(y) == 16000 and abs(rms(y) / (2 ** -0.5) - 1) < 0.01


def test_future_changes_cannot_change_model_prefix():
    rng = np.random.default_rng(17)
    x = rng.normal(0, 0.1, 48000).astype(np.float32)
    changed = x.copy()
    changed[24000:] = 100
    assert np.array_equal(causal_prefix(x, 48000, 0.5), causal_prefix(changed, 48000, 0.5))
    # the failure avoided by truncating before symmetric filtering
    assert not np.array_equal(resample_antialiased(x, 48000)[:8000], resample_antialiased(changed, 48000)[:8000])


def test_same_rate_is_exact_and_invalid_cut_rejected():
    x = tone(1000)
    assert np.array_equal(x, resample_antialiased(x, 16000))
    with pytest.raises(ValueError):
        causal_prefix(x, 48000, 2.0)
