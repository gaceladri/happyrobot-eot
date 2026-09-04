"""Audio front-end shared by training, mining, serving and evaluation.

Conventions (identical to Smart Turn v3 so results stay comparable):

- 16 kHz mono float32 in [-1, 1].
- The model sees at most the last ``WINDOW_SECONDS`` (8 s) of the *current* user turn.
- Shorter windows are left-padded with zeros (the informative part, the pause, stays at the end).
- Features: Whisper 80-bin log-mel, ``chunk_length=8`` -> ``[80, 800]``.

Telephony augmentation (``telephony_augment``) simulates the PSTN path HappyRobot actually hears:
8 kHz band-limit, G.711 mu-law companding, gain jitter, additive noise, short packet drops.
It is used for training augmentation and as a separate evaluation slice; never mix clean and
telephony metrics into one number.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 8.0
N_MELS = 80
N_FRAMES = 800  # 8 s * 100 frames/s (hop 160 @ 16 kHz)


def to_float32(x: np.ndarray) -> np.ndarray:
    """Convert PCM16 / int arrays to float32 in [-1, 1]; pass float through."""
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        return x.astype(np.float32) / max(abs(info.min), info.max)
    return x.astype(np.float32, copy=False)


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=-1) if x.shape[-1] <= 8 else x.mean(axis=0)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """Linear-interpolation resampler. Good enough for 8k<->16k<->24k/48k speech at this scale.

    Deliberately dependency-free; swap in ``torchaudio.functional.resample`` if quality matters
    for a given slice (e.g. Mimi 24 kHz features).
    """
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def load_audio_bytes(
    data: bytes,
    assume_pcm16_sr: int | None = None,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """Decode WAV/FLAC or raw PCM16, optionally reading only the trailing duration."""
    if assume_pcm16_sr is not None:
        pcm = np.frombuffer(data, dtype="<i2")
        if max_seconds is not None:
            pcm = pcm[-int(round(max_seconds * assume_pcm16_sr)):]
        return to_float32(pcm), assume_pcm16_sr
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(data)) as stream:
        sr = int(stream.samplerate)
        frames = -1
        if max_seconds is not None:
            frames = int(round(max_seconds * sr))
            if stream.frames > frames:
                stream.seek(stream.frames - frames)
        audio = stream.read(frames=frames, dtype="float32", always_2d=False)
    return to_mono(np.asarray(audio)), sr


def last_window(x: np.ndarray, seconds: float = WINDOW_SECONDS, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Keep the last ``seconds`` of audio; left-pad with zeros if shorter."""
    n = int(seconds * sr)
    if len(x) >= n:
        return x[-n:]
    out = np.zeros(n, dtype=np.float32)
    if len(x):
        out[-len(x):] = x
    return out


@lru_cache(maxsize=1)
def _feature_extractor():
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor(chunk_length=int(WINDOW_SECONDS), feature_size=N_MELS, sampling_rate=SAMPLE_RATE)


def log_mel(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Waveform -> ``[80, 800]`` float32 log-mel, Whisper normalisation, last-8 s window."""
    x = last_window(resample(to_mono(to_float32(x)), sr))
    feats = _feature_extractor()(x, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length", truncation=True)
    return feats["input_features"][0].astype(np.float32)


# ---------------------------------------------------------------------------
# Silence / speech framing used by prefix mining and the causal replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_rms(x: np.ndarray, sr: int = SAMPLE_RATE, frame_ms: float = 20.0) -> np.ndarray:
    hop = int(sr * frame_ms / 1000)
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = x[: n * hop].reshape(n, hop)
    return np.sqrt((frames**2).mean(axis=1) + 1e-12).astype(np.float32)


def silence_spans(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: float = 20.0,
    min_silence: float = 0.1,
    rel_db: float = -35.0,
    floor_db: float = -60.0,
) -> list[Span]:
    """Energy-based silence detector returning spans >= ``min_silence`` seconds.

    Threshold = max(peak_db + rel_db, floor_db). This is intentionally simple and fully
    causal-compatible; use Silero/TEN VAD for production trigger, this is for *offline mining*.
    """
    rms = frame_rms(x, sr, frame_ms)
    if len(rms) == 0:
        return []
    db = 20 * np.log10(rms + 1e-9)
    thr = max(db.max() + rel_db, floor_db)
    quiet = db < thr
    hop = frame_ms / 1000.0
    spans: list[Span] = []
    i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            start, end = i * hop, j * hop
            # Frame arithmetic such as 0.6 - 0.4 may land one ulp below 0.2.
            # Treat a frame-aligned pause exactly at the requested threshold as valid.
            if end - start + 1e-9 >= min_silence:
                spans.append(Span(start, end))
            i = j
        else:
            i += 1
    return spans


# ---------------------------------------------------------------------------
# Telephony augmentation
# ---------------------------------------------------------------------------


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
