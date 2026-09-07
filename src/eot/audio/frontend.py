"""Waveform conventions and the Whisper log-mel front-end (torch-free).

Identical to Smart Turn v3 so results stay comparable: 16 kHz mono float32 in [-1, 1]; the model
sees at most the last ``WINDOW_SECONDS`` (8 s) of the current user turn, left-padded with zeros;
features are the Whisper 80-bin log-mel with ``chunk_length=8`` -> ``[80, 800]``.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 8.0
N_MELS = 80
N_FRAMES = 800  # 8 s * 100 frames/s (hop 160 @ 16 kHz)
SCORE_POINT = 0.2  # first decision point inside a pause (EoT Bench / Smart Turn convention)
GRID_STEP = 0.1  # the harness scores every 100 ms of silence after the score point


def to_float32(x: np.ndarray) -> np.ndarray:
    """Convert PCM16 / int arrays to float32 in [-1, 1]; pass float through."""
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        return x.astype(np.float32) / max(abs(info.min), info.max)
    return x.astype(np.float32, copy=False)


def to_pcm16_bytes(x: np.ndarray) -> bytes:
    """Float waveform in [-1, 1] -> little-endian PCM16 bytes (the raw body ``POST /v1/eot`` takes with ``?sr=``)."""
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=-1) if x.shape[-1] <= 8 else x.mean(axis=0)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """Linear-interpolation resampler. Good enough for 8k<->16k<->24k/48k speech at this scale.

    This is the resampler the shipped model was trained, exported and validated with, so serving
    and evaluation keep it. Data preparation may use ``resample_antialiased`` instead.
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
            pcm = pcm[-int(round(max_seconds * assume_pcm16_sr)) :]
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


def resample_antialiased(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """Polyphase FIR resampler (Kaiser 5.0) for *data preparation*.

    Linear interpolation lets out-of-band recording energy alias into speech frequencies; the
    polyphase filter attenuates the stopband by more than 40 dB. Serving keeps ``resample`` so
    the deployed graph matches the parity-validated one.
    """
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("sample rates must be positive")
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or not len(x):
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, window=("kaiser", 5.0)).astype(np.float32)


def causal_prefix(raw: np.ndarray, sr: int, cut_time: float, seconds: float = WINDOW_SECONDS) -> np.ndarray:
    """The last ``seconds`` of ``raw`` ending at ``cut_time``, resampled *after* truncation.

    A symmetric FIR applied to the whole recording would leak samples after the cut into the
    prefix; truncating first makes the training waveform provably causal.
    """
    if not np.isfinite(cut_time) or cut_time < 0 or cut_time > len(raw) / sr:
        raise ValueError("cut outside observed recording")
    stop = int(cut_time * sr)
    start = max(0, stop - int(seconds * sr))
    return resample_antialiased(raw[start:stop], sr)


def to_16k(x, sr: int) -> np.ndarray:
    """Any PCM/float array at ``sr`` -> 16 kHz mono float32 (the package's one waveform convention)."""
    return resample(to_mono(to_float32(np.asarray(x))), int(sr))


def load_wav(path: str | os.PathLike) -> np.ndarray:
    """Read any soundfile-supported file as 16 kHz mono float32."""
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    return to_16k(x, sr)


def decode_payload(payload) -> np.ndarray:
    """Decode a Hugging Face ``Audio`` payload, an ``(array, sr)`` tuple or a bare 16 kHz array.

    Accepted dict forms: ``{"array", "sampling_rate"}``, ``{"bytes"}`` or ``{"path"}``.
    """
    if isinstance(payload, dict):
        if payload.get("array") is not None:
            return to_16k(payload["array"], payload["sampling_rate"])
        if payload.get("bytes") is not None:
            x, sr = load_audio_bytes(payload["bytes"])
            return resample(x, sr)
        if payload.get("path"):
            return load_wav(payload["path"])
        raise ValueError("unsupported audio payload")
    if isinstance(payload, tuple) and len(payload) == 2:
        return to_16k(*payload)
    return to_16k(payload, SAMPLE_RATE)


def last_window(x: np.ndarray, seconds: float = WINDOW_SECONDS, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Keep the last ``seconds`` of audio; left-pad with zeros if shorter."""
    n = int(seconds * sr)
    if len(x) >= n:
        return x[-n:]
    out = np.zeros(n, dtype=np.float32)
    if len(x):
        out[-len(x) :] = x
    return out


def log_mel(x: np.ndarray, sr: int = SAMPLE_RATE, *, normalize: bool = False) -> np.ndarray:
    """Waveform -> ``[80, 800]`` float32 log-mel, Whisper normalisation, last-8 s window."""
    x = last_window(to_16k(x, sr))
    if normalize:
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    # Vectorized STFT avoids the reference NumPy frontend's Python frame loop and never
    # imports a Torch runtime. Periodic Hann, centered reflect padding and the last-frame
    # removal reproduce WhisperFeatureExtractor's exact preprocessing convention.
    window, filters = _mel_constants()
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (200, 200), mode="reflect"), 400)[::160][:-1]
    spectrum = np.fft.rfft(frames * window, axis=-1)
    power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    mel = filters @ power.T
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)


@lru_cache(maxsize=1)
def _mel_constants():
    from transformers.audio_utils import mel_filter_bank

    filters = mel_filter_bank(201, N_MELS, 0.0, 8000.0, SAMPLE_RATE, norm="slaney", mel_scale="slaney").astype(np.float32).T
    return np.hanning(401)[:-1].astype(np.float32), filters
