"""Training-time audio augmentation: room tone, a PSTN leg and Krisp-style pause tails.

``telephony_augment`` simulates the path HappyRobot actually hears (8 kHz band-limit, G.711
mu-law, gain jitter, additive noise, short packet drops). It is used as training augmentation
and as a separate evaluation slice; clean and telephony metrics are never pooled.

``tail_texture`` rewrites the audio *after* the pause start, independently of the label, so that
the texture of the pause (gated digital zeros or room noise at the row's own floor) carries no
information about whether the turn ended. The transform reproduces the aggregate signature of the
Krisp test set measured in L036 (85 % of official EoT tails and 83 % of Krisp tails are gated) and
was the augmentation kept from the L037 study (arm A1) and combined with distillation in L040.
"""

from __future__ import annotations

import numpy as np

from eot.audio.frontend import SAMPLE_RATE, resample
from eot.audio.pauses import DIGITAL_SILENCE_DB, frame_db


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate 1/f noise (Voss-McCartney style via cumulative octave sums); unit RMS."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    n_rows = 12
    array = np.zeros(n, dtype=np.float64)
    for row in range(n_rows):
        step = 2**row
        values = rng.normal(0.0, 1.0, size=(n + step - 1) // step)
        array += np.repeat(values, step)[:n]
    array += rng.normal(0.0, 1.0, size=n)
    array /= np.sqrt(np.mean(array**2) + 1e-12)
    return array.astype(np.float32)


def add_room_tone(
    x: np.ndarray,
    rng: np.random.Generator,
    snr_db: tuple[float, float] = (15.0, 35.0),
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Add continuous coloured noise at a random SNR relative to the *speech* power.

    Used for sources whose silence is digital zero (VoIP gating, e.g. AppTek): a model trained on
    those would learn 'exact zero = pause', a shortcut that does not exist on the PSTN.
    """
    db = frame_db(x, sr)
    audible = db[db > DIGITAL_SILENCE_DB]
    if len(audible) == 0:
        return x
    speech_db = float(np.percentile(audible, 90))
    speech_power = 10 ** (speech_db / 10)
    snr = float(rng.uniform(*snr_db))
    noise = pink_noise(len(x), rng) * np.sqrt(speech_power / (10 ** (snr / 10)))
    return np.clip(x + noise, -1.0, 1.0).astype(np.float32)


def mu_law(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """G.711 mu-law companding round trip (8-bit) — the dominant codec on US PSTN."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0
    return (np.sign(q) * (1.0 / mu) * ((1.0 + mu) ** np.abs(q) - 1.0)).astype(np.float32)


def telephony_augment(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
    snr_db: tuple[float, float] = (12.0, 35.0),
    drop_prob: float = 0.3,
    drop_ms: tuple[float, float] = (20.0, 80.0),
) -> np.ndarray:
    """Simulate a PSTN leg: 8 kHz band-limit, mu-law, gain jitter, noise, packet loss.

    Returns 16 kHz float32 of the same length as the input.
    """
    rng = rng or np.random.default_rng()
    y = resample(x, sr, 8_000)
    y = y * float(rng.uniform(0.5, 1.4))
    y = mu_law(y)
    noise = rng.normal(0.0, 1.0, size=len(y)).astype(np.float32)
    sig_p = float((y**2).mean() + 1e-9)
    snr = float(rng.uniform(*snr_db))
    noise *= np.sqrt(sig_p / (10 ** (snr / 10)))
    y = y + noise
    if rng.random() < drop_prob and len(y) > 800:
        n_drop = int(8_000 * rng.uniform(*drop_ms) / 1000)
        start = int(rng.integers(0, max(1, len(y) - n_drop)))
        y[start : start + n_drop] = 0.0
    y = resample(y, 8_000, sr)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return np.clip(y[: len(x)], -1.0, 1.0).astype(np.float32)


def noise_floor_dbfs(x: np.ndarray, speech_end_s: float, sr: int = SAMPLE_RATE, default_dbfs: float = -60.0) -> float:
    """Estimate the noise floor (dBFS) of the speech region: p10 of the audible 20 ms frames.

    Frames at digital silence are ignored; when fewer than five audible frames exist the estimate is
    not meaningful and ``default_dbfs`` is returned.
    """
    speech = x[: max(1, int(round(speech_end_s * sr)))]
    db = frame_db(speech, sr)
    audible = db[db > DIGITAL_SILENCE_DB]
    return float(np.percentile(audible, 10)) if len(audible) >= 5 else float(default_dbfs)


def tail_texture(
    x: np.ndarray,
    tail_len_s: float,
    rng: np.random.Generator,
    *,
    gated_prob: float = 0.5,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, dict]:
    """Replace the trailing ``tail_len_s`` seconds (the pause after the cut) with a label-independent texture.

    With probability ``gated_prob`` the tail becomes exact zeros (a VoIP/AEC gate); otherwise pink
    noise at the noise floor estimated on the speech region. The speech itself is never modified.
    Returns the new waveform and a record ``{"kind": "gated" | "room_noise", "floor_dbfs": ...}``.
    """
    if tail_len_s < 0:
        raise ValueError(f"tail_len_s must be >= 0, got {tail_len_s}")
    start = int(np.clip(round(len(x) - tail_len_s * sr), 0, len(x)))
    y = np.array(x, dtype=np.float32, copy=True)
    if rng.random() < gated_prob:
        y[start:] = 0.0
        return y, {"kind": "gated", "floor_dbfs": None}
    floor = noise_floor_dbfs(x, start / sr, sr)
    noise = pink_noise(len(x) - start, rng)
    y[start:] = noise * 10 ** (floor / 20)
    return np.clip(y, -1.0, 1.0).astype(np.float32), {"kind": "room_noise", "floor_dbfs": floor}
