"""Torch-free audio front-end: conventions, log-mel, pause detection, augmentation."""

from eot.audio.augment import add_room_tone, mu_law, pink_noise, telephony_augment
from eot.audio.frontend import (
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SECONDS,
    causal_prefix,
    decode_payload,
    last_window,
    load_audio_bytes,
    load_wav,
    log_mel,
    resample,
    resample_antialiased,
    to_float32,
    to_mono,
)
from eot.audio.pauses import (
    DIGITAL_SILENCE_DB,
    SILERO_VAD_SHA256,
    SILERO_VAD_VERSION,
    PauseDetector,
    PauseSpan,
    SileroVAD,
    Span,
    digital_silence_fraction,
    ensemble_pauses,
    ensure_silero_vad,
    estimate_snr_db,
    frame_db,
    frame_rms,
    hysteresis_mask,
    remove_short_runs,
    silence_spans,
    silence_spans_adaptive,
    spans_from_quiet_mask,
    vad_quiet_mask,
    vad_silence_spans,
)

__all__ = [
    "DIGITAL_SILENCE_DB", "N_FRAMES", "N_MELS", "SAMPLE_RATE", "SILERO_VAD_SHA256", "SILERO_VAD_VERSION",
    "WINDOW_SECONDS", "PauseDetector", "PauseSpan", "SileroVAD", "Span", "add_room_tone", "causal_prefix",
    "decode_payload", "digital_silence_fraction", "ensemble_pauses", "ensure_silero_vad", "estimate_snr_db", "frame_db",
    "frame_rms", "hysteresis_mask", "last_window", "load_audio_bytes", "load_wav", "log_mel", "mu_law",
    "pink_noise", "remove_short_runs", "resample", "resample_antialiased", "silence_spans", "silence_spans_adaptive", "spans_from_quiet_mask",
    "telephony_augment", "to_float32", "to_mono", "vad_quiet_mask", "vad_silence_spans",
]
