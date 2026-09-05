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

import hashlib
import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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


def log_mel(x: np.ndarray, sr: int = SAMPLE_RATE, *, normalize: bool = False) -> np.ndarray:
    """Waveform -> ``[80, 800]`` float32 log-mel, Whisper normalisation, last-8 s window."""
    x = last_window(resample(to_mono(to_float32(x)), sr))
    if normalize:
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    # Vectorized STFT avoids the reference NumPy frontend's Python frame loop and never
    # imports a Torch runtime. Periodic Hann, centered reflect padding and the last-frame
    # removal reproduce WhisperFeatureExtractor's exact preprocessing convention.
    window, filters = _mel_constants()
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (200, 200), mode="reflect"), 400)[::160][:-1]
    spectrum = np.fft.rfft(frames * window, axis=-1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    mel = filters @ power.T
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)


@lru_cache(maxsize=1)
def _mel_constants():
    from transformers.audio_utils import mel_filter_bank
    filters = mel_filter_bank(201, N_MELS, 0.0, 8000.0, SAMPLE_RATE,
                              norm="slaney", mel_scale="slaney").astype(np.float32).T
    return np.hanning(401)[:-1].astype(np.float32), filters


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
# Robust pause detection for mining: adaptive energy + Silero VAD ensemble
# ---------------------------------------------------------------------------
#
# Why a second detector. ``silence_spans`` thresholds relative to the clip *peak*. One transient
# (a cough, a click) drags ordinary speech below the threshold and the cut lands mid-word;
# stationary noise at 12 dB SNR lifts the floor above the threshold and the clip yields no
# internal pause at all. Both failures are silent. The ensemble below keeps a pause only where a
# noise-floor-adaptive energy detector *and* a learned VAD agree, places the cut inside their
# intersection, and records how confident that agreement is so the training set can be audited
# and filtered instead of trusted.

SILERO_VAD_VERSION = "v6.2.1"
SILERO_VAD_COMMIT = "7e30209a3e901f9842f81b225f3e93d8199902b1"
SILERO_VAD_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    f"{SILERO_VAD_COMMIT}/src/silero_vad/data/silero_vad.onnx"
)
SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_CHUNK = 512  # 32 ms at 16 kHz
SILERO_CONTEXT = 64  # samples of previous chunk the v5+ model expects prepended
DIGITAL_SILENCE_DB = -80.0


def ensure_silero_vad(path: str | os.PathLike | None = None) -> Path:
    """Return a verified local copy of the pinned Silero VAD ONNX file, downloading if needed."""
    target = Path(path or os.environ.get("EOT_SILERO_VAD") or Path("models/vad") / f"silero_vad_{SILERO_VAD_VERSION}.onnx")
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SILERO_VAD_SHA256:
        return target
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial")
    with urllib.request.urlopen(SILERO_VAD_URL, timeout=60) as response, tmp.open("wb") as f:
        f.write(response.read())
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != SILERO_VAD_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Silero VAD checksum mismatch: {digest} != {SILERO_VAD_SHA256}")
    os.replace(tmp, target)
    return target


class SileroVAD:
    """Streaming Silero VAD (ONNX Runtime, CPU). ``probabilities`` -> one P(speech) per 32 ms chunk."""

    def __init__(self, path: str | os.PathLike | None = None, threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.path = ensure_silero_vad(path)
        self.session = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
        self.sha256 = SILERO_VAD_SHA256
        self.chunk_seconds = SILERO_CHUNK / SAMPLE_RATE

    def probabilities(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        x = resample(to_mono(to_float32(x)), sr)
        n_chunks = int(np.ceil(len(x) / SILERO_CHUNK))
        if n_chunks == 0:
            return np.zeros(0, dtype=np.float32)
        padded = np.zeros(n_chunks * SILERO_CHUNK, dtype=np.float32)
        padded[: len(x)] = x
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, SILERO_CONTEXT), dtype=np.float32)
        sr_in = np.array(SAMPLE_RATE, dtype=np.int64)
        out = np.empty(n_chunks, dtype=np.float32)
        for i in range(n_chunks):
            chunk = padded[i * SILERO_CHUNK : (i + 1) * SILERO_CHUNK][None, :]
            inp = np.concatenate([context, chunk], axis=1)
            prob, state = self.session.run(None, {"input": inp, "state": state, "sr": sr_in})
            out[i] = float(prob[0, 0])
            context = inp[:, -SILERO_CONTEXT:]
        return out


def frame_db(x: np.ndarray, sr: int = SAMPLE_RATE, frame_ms: float = 20.0) -> np.ndarray:
    return 20 * np.log10(frame_rms(x, sr, frame_ms) + 1e-9)


def hysteresis_mask(values: np.ndarray, on: float, off: float, *, start_active: bool = False) -> np.ndarray:
    """Boolean 'active' mask with two thresholds: activate above ``on``, deactivate below ``off``."""
    active = start_active
    out = np.zeros(len(values), dtype=bool)
    for i, v in enumerate(values):
        if active:
            active = v > off
        else:
            active = v > on
        out[i] = active
    return out


def remove_short_runs(mask: np.ndarray, value: bool, min_len: int) -> np.ndarray:
    """Flip runs of ``value`` shorter than ``min_len`` frames to the opposite value."""
    out = mask.copy()
    i = 0
    n = len(out)
    while i < n:
        if out[i] == value:
            j = i
            while j < n and out[j] == value:
                j += 1
            if j - i < min_len and (i > 0 or j < n):
                out[i:j] = not value
            i = j
        else:
            i += 1
    return out


def spans_from_quiet_mask(quiet: np.ndarray, hop: float, min_silence: float, total: float | None = None) -> list[Span]:
    spans: list[Span] = []
    i = 0
    n = len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            start, end = i * hop, j * hop
            if total is not None and j == n:
                end = max(end, total)
            if end - start + 1e-9 >= min_silence:
                spans.append(Span(start, end))
            i = j
        else:
            i += 1
    return spans


def silence_spans_adaptive(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: float = 20.0,
    min_silence: float = 0.2,
    margin_db: float = 9.0,
    floor_percentile: float = 10.0,
    hysteresis_db: float = 3.0,
    floor_db: float = -60.0,
    rel_db: float = -12.0,
    min_speech: float = 0.1,
) -> list[Span]:
    """Noise-floor-adaptive energy pauses.

    Threshold = clip(percentile_10(dB) + margin, floor_db, peak + rel_db): it follows the noise
    floor, unlike the legacy peak-relative rule, so stationary noise cannot hide every pause. The
    peak-relative cap is only a guard for degenerate clips. Soft speech that dips under the
    threshold in noisy clips is caught downstream by the VAD veto in ``ensemble_pauses``. Speech
    starts when a frame exceeds threshold + hysteresis and ends when it drops below the
    threshold; speech islands shorter than ``min_speech`` are folded into the pause.
    """
    db = frame_db(x, sr, frame_ms)
    if len(db) == 0:
        return []
    hop = frame_ms / 1000.0
    audible = db[db > DIGITAL_SILENCE_DB]
    noise_floor = float(np.percentile(audible if len(audible) else db, floor_percentile))
    upper = float(db.max() + rel_db)
    thr = float(np.clip(noise_floor + margin_db, floor_db, max(floor_db, upper)))
    speech = hysteresis_mask(db, on=thr + hysteresis_db, off=thr)
    speech = remove_short_runs(speech, True, max(1, int(round(min_speech / hop))))
    return spans_from_quiet_mask(~speech, hop, min_silence, total=len(x) / sr)


def vad_silence_spans(
    probs: np.ndarray,
    chunk_seconds: float = SILERO_CHUNK / SAMPLE_RATE,
    min_silence: float = 0.2,
    onset: float = 0.5,
    offset: float = 0.35,
    min_speech: float = 0.1,
    total: float | None = None,
) -> list[Span]:
    speech = hysteresis_mask(probs, on=onset, off=offset)
    speech = remove_short_runs(speech, True, max(1, int(round(min_speech / chunk_seconds))))
    return spans_from_quiet_mask(~speech, chunk_seconds, min_silence, total=total)


@dataclass(frozen=True)
class PauseSpan:
    """A pause with an agreement tier.

    ``start``/``end`` is the energy pause (``start`` is what a policy would see as pause onset);
    ``agree_from`` is the time from which the VAD is also quiet, so any cut at t >= agree_from is
    endorsed by both detectors; ``onset`` is the earliest plausible speech resumption (min of the
    two ends); ``iou`` is the fraction of the energy pause the VAD agrees with.
    """

    start: float
    end: float
    onset: float
    iou: float
    confidence: str  # "high" | "medium" | "low"
    energy: Span | None
    vad: Span | None
    agree_from: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


def vad_quiet_mask(
    probs: np.ndarray,
    onset: float = 0.5,
    offset: float = 0.35,
    min_speech: float = 0.1,
    chunk_seconds: float = SILERO_CHUNK / SAMPLE_RATE,
) -> np.ndarray:
    speech = hysteresis_mask(probs, on=onset, off=offset)
    speech = remove_short_runs(speech, True, max(1, int(round(min_speech / chunk_seconds))))
    return ~speech


def ensemble_pauses(
    energy_spans: list[Span],
    vad_quiet: np.ndarray,
    chunk_seconds: float = SILERO_CHUNK / SAMPLE_RATE,
    min_silence: float = 0.2,
    high_lag: float = 0.1,
    medium_lag: float | None = None,
    total: float | None = None,
) -> list[PauseSpan]:
    """Grade every energy pause by whether the VAD is quiet through it.

    Silero's posterior decays over a few chunks after speech ends, so requiring span IoU throws
    away most short, real pauses. What matters for a cut at ``start + g`` is that (a) the VAD is
    quiet at the cut and stays quiet until the energy pause ends and (b) its quiet run began soon
    after the energy pause did. ``lag`` = VAD quiet start - energy pause start:

    - high:   lag <= ``high_lag`` (both detectors saw the same boundary),
    - medium: lag <= ``medium_lag`` (default ``min_silence`` + one chunk: the 0.2 s cut is still
      inside the agreement region),
    - low:    the VAD still hears speech at the end of the energy pause (energy dipped inside a
      word: fricative, breath, low-energy syllable). Kept for audit only.

    VAD-only quiet runs (energy hears something: music, noise) are appended as ``low`` with
    ``energy=None``.
    """
    medium_lag = min_silence + chunk_seconds if medium_lag is None else medium_lag
    n = len(vad_quiet)
    out: list[PauseSpan] = []
    covered = np.zeros(n, dtype=bool)
    for e in energy_spans:
        i0 = max(0, min(n, int(np.floor(e.start / chunk_seconds))))
        # Last chunk fully inside the energy pause; the chunk containing the energy end may already
        # hold the next speech onset (the VAD often leads the hysteresis-delayed energy onset).
        i1 = max(i0 + 1, min(n, int(np.floor(e.end / chunk_seconds))))
        window = vad_quiet[i0:i1]
        if len(window) >= 2 and not window[-1] and window[-2]:
            window = window[:-1]  # VAD heard the onset one chunk (32 ms) before energy did
            i1 -= 1
        if len(window) == 0 or not window[-1]:
            out.append(PauseSpan(e.start, e.end, e.end, 0.0, "low", e, None, agree_from=e.end))
            covered[i0:i1] = True
            continue
        j = len(window) - 1
        while j > 0 and window[j - 1]:
            j -= 1
        agree_from = max(e.start, (i0 + j) * chunk_seconds)
        k = i1
        while k < n and vad_quiet[k]:
            k += 1
        vad_end = k * chunk_seconds if k < n else (total if total is not None else e.end)
        onset = min(e.end, max(vad_end, agree_from))
        lag = agree_from - e.start
        iou = max(0.0, (e.end - agree_from) / max(1e-9, e.end - e.start))
        conf = "high" if lag <= high_lag + 1e-9 else "medium" if lag <= medium_lag + 1e-9 else "low"
        out.append(PauseSpan(e.start, e.end, onset, round(iou, 4), conf, e, Span(agree_from, vad_end), agree_from=agree_from))
        covered[i0:k] = True
    i = 0
    while i < n:
        if vad_quiet[i] and not covered[i]:
            j = i
            while j < n and vad_quiet[j] and not covered[j]:
                j += 1
            if (j - i) * chunk_seconds + 1e-9 >= min_silence:
                s = Span(i * chunk_seconds, j * chunk_seconds)
                out.append(PauseSpan(s.start, s.end, s.end, 0.0, "low", None, s, agree_from=s.start))
            i = j
        else:
            i += 1
    return sorted(out, key=lambda p: p.start)


def estimate_snr_db(x: np.ndarray, sr: int = SAMPLE_RATE, frame_ms: float = 20.0) -> float:
    """Crude SNR: p90 of frame dB (speech) minus p10 of audible frames (floor). Digital silence excluded."""
    db = frame_db(x, sr, frame_ms)
    audible = db[db > DIGITAL_SILENCE_DB]
    if len(audible) < 5:
        return float("nan")
    return float(np.percentile(audible, 90) - np.percentile(audible, 10))


def digital_silence_fraction(x: np.ndarray, sr: int = SAMPLE_RATE, frame_ms: float = 20.0) -> float:
    db = frame_db(x, sr, frame_ms)
    return float((db <= DIGITAL_SILENCE_DB).mean()) if len(db) else 0.0


class PauseDetector:
    """Pause detector used by mining. ``legacy`` reproduces ``silence_spans`` (rev. 2); ``energy``
    is the adaptive detector alone; ``ensemble`` grades adaptive-energy pauses with Silero VAD."""

    def __init__(self, detector: str = "ensemble", vad: SileroVAD | None = None, min_silence: float = 0.2):
        if detector not in ("ensemble", "legacy", "energy"):
            raise ValueError(f"unknown detector {detector!r}")
        self.detector = detector
        self.min_silence = float(min_silence)
        self.vad = vad if (vad is not None or detector != "ensemble") else SileroVAD()

    def __call__(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> list[PauseSpan]:
        total = len(x) / sr
        if self.detector == "legacy":
            return [PauseSpan(s.start, s.end, s.end, 1.0, "high", s, None, agree_from=s.start) for s in silence_spans(x, sr, min_silence=self.min_silence)]
        energy = silence_spans_adaptive(x, sr, min_silence=self.min_silence)
        if self.detector == "energy":
            return [PauseSpan(s.start, s.end, s.end, 1.0, "high", s, None, agree_from=s.start) for s in energy]
        probs = self.vad.probabilities(x, sr)
        quiet = vad_quiet_mask(probs, chunk_seconds=self.vad.chunk_seconds)
        return ensemble_pauses(energy, quiet, chunk_seconds=self.vad.chunk_seconds, min_silence=self.min_silence, total=total)


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
