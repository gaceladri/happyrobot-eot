"""Training-time audio augmentation: room tone for digital-silence sources and a PSTN leg.

``telephony_augment`` simulates the path HappyRobot actually hears (8 kHz band-limit, G.711
mu-law, gain jitter, additive noise, short packet drops). It is used as training augmentation
and as a separate evaluation slice; clean and telephony metrics are never pooled.
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
