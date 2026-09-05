"""Pause detection for offline mining: legacy energy, noise-floor-adaptive energy and a
Silero VAD ensemble that grades every pause by detector agreement.

Why an ensemble. The legacy detector thresholds relative to the clip *peak*: one transient drags
ordinary speech below the threshold (cut lands mid-word) and stationary noise at telephony SNR
lifts the floor above it (no pause at all). Both failures are silent. The ensemble keeps a pause
only where the adaptive energy detector and a learned VAD agree, places the cut inside their
intersection and records how confident that agreement is so training data can be audited and
filtered instead of trusted.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eot.audio.frontend import SAMPLE_RATE, to_16k
from eot.io import atomic_replace, sha256_file
from eot.onnx import cpu_session


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
    if target.exists() and sha256_file(target) == SILERO_VAD_SHA256:
        return target
    import urllib.request

    def download(tmp: Path) -> None:
        with urllib.request.urlopen(SILERO_VAD_URL, timeout=60) as response, tmp.open("wb") as f:
            f.write(response.read())
        digest = sha256_file(tmp)
        if digest != SILERO_VAD_SHA256:
            raise RuntimeError(f"Silero VAD checksum mismatch: {digest} != {SILERO_VAD_SHA256}")

    atomic_replace(target, download, fsync=False)  # the temp file is removed on any failure
    return target


class SileroVAD:
    """Streaming Silero VAD (ONNX Runtime, CPU). ``probabilities`` -> one P(speech) per 32 ms chunk."""

    def __init__(self, path: str | os.PathLike | None = None, threads: int = 1):
        self.path = ensure_silero_vad(path)
        self.session = cpu_session(self.path, threads, log_severity=3)
        self.sha256 = SILERO_VAD_SHA256
        self.chunk_seconds = SILERO_CHUNK / SAMPLE_RATE

    def probabilities(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        x = to_16k(x, sr)
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
