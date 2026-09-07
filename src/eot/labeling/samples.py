"""Sample schema shared by every labeler: clips in, pause-level causal samples out.

Every sample carries multi-horizon future-speech targets (``fvad``): "will the user speak again
within h seconds?" for h in ``HORIZONS``. Targets are derived from observed timestamps only; a
censored future masks the corresponding horizons instead of assuming silence.

Observation window (rev. 4)
---------------------------
Every sample records how long the speaker's own channel was observed after the cut without an
intervening event (``future_observed_until``, seconds, never negative) and which event ended the
observation (``observation_end_reason``): the clip ended, the other party (AppTek agent) took the
floor, the speaker resumed, or the labeler's maximum hold. A cut placed *after* the event (a tail
padded with room tone, an agent that already spoke) keeps the event's reason with a window of 0;
the row is kept and flagged, never silently dropped, so training can exclude or weight it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from eot.audio import SAMPLE_RATE

HORIZONS: tuple[float, ...] = (0.24, 0.64, 1.2, 2.0)
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
OBSERVATION_END_REASONS: tuple[str, ...] = ("clip_end", "agent_onset", "customer_resumed", "max_hold", "other")
# A HOLD cut may never lie past ``next_onset - ONSET_MARGIN_S``. The default keeps a pause that
# lasts exactly the score point a valid HOLD (the resumption is on the next sample, and the slice
# ``x[:cut]`` excludes it); a positive margin additionally drops cuts within the margin of the onset.
ONSET_MARGIN_S = 0.0
# Dual-channel labeling thresholds shared by the AppTek heuristic and the annotation-derived
# targets (pre-registered; do not tune after seeing test results).
EOT_CUSTOMER_SILENT_S = 1.0  # the floor has transferred only if the speaker stays silent this long after the listener starts
MAX_HOLD_S = 5.0  # a resumption later than this is not a hold decision but a new turn (or censored)


@dataclass
class Clip:
    id: str
    audio: np.ndarray  # float32 16 kHz mono
    label: int  # 1 = complete / EOT, 0 = incomplete / HOLD
    source: str = "unknown"
    agent_text: str = ""  # previous agent utterance if known (free context at runtime)
    sr: int = SAMPLE_RATE


@dataclass
class Sample:
    id: str
    clip_id: str
    audio: np.ndarray = field(repr=False)
    label: int
    kind: str  # "internal" | "final" | "whole" (prefix mining) | "oracle_eot" | "oracle_hold" (dual-channel labeling)
    cut_time: float
    pause_start: float
    time_to_onset: float | None  # seconds until next speech onset; None if EOT or censored
    fvad: list[int]
    fvad_mask: int | list[int]  # validity per horizon; scalar accepted for old manifests
    source: str
    agent_text: str
    # Labeling-quality metadata (rev. 3). Defaults keep older manifests loadable.
    cut_confidence: str = "high"  # high | medium | low (see audio.ensemble_pauses)
    pause_iou: float = 1.0  # IoU between the energy and VAD pause spans
    snr_est_db: float | None = None  # crude clip SNR estimate
    onset_margin_ms: float | None = None  # ms from the cut to the earliest plausible speech onset
    detector: str = "legacy"
    noise_fill: int = 0  # 1 -> add room tone at train time (digital-silence sources)
    # Observation window (rev. 4); see the module docstring.
    future_observed_until: float | None = None  # s after the cut observed without an intervening event
    observation_end_reason: str | None = None  # one of OBSERVATION_END_REASONS
    clip_duration_s: float | None = None  # length of the source clip / channel the cut was taken from
    weight: float = 1.0  # per-row loss weight (relative; 0 excludes the row from loss_eot)
    extra: dict = field(default_factory=dict)  # source-specific fields (accent, speaker_id, ...)

    def meta(self) -> dict:
        """Manifest row: every field except the waveform, with ``extra`` flattened (never overriding)."""
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("audio", "extra")}
        for key, value in (self.extra or {}).items():
            d.setdefault(key, value)
        return d


def fvad_targets(time_to_onset: float | None, observed_until: float = 0.0) -> tuple[list[int], list[int]]:
    """Future activity is observed independently of the conversational EOT label.

    With a known next onset every horizon is known. With right censoring only horizons
    within the observed silent future are valid. Clip completion alone proves nothing.
    Kept for callers that have no observation window; labelers use :func:`observed_fvad_targets`.
    """
    if time_to_onset is not None:
        if time_to_onset < -1e-9:
            raise ValueError("cut is after the next speech onset")
        return [int(time_to_onset <= h) for h in HORIZONS], [1] * len(HORIZONS)
    return [0] * len(HORIZONS), [int(h <= observed_until) for h in HORIZONS]


def observed_fvad_targets(
    time_to_onset: float | None,
    future_observed_until: float,
    tol: float = 1e-6,
    *,
    ended_by_resumption: bool | None = None,
) -> tuple[list[int], list[int]]:
    """Per-horizon future-speech targets and validity from the observation window.

    ``fvad[i]`` is 1 iff the speaker resumed on their own channel within ``HORIZONS[i]``. A target is
    asserted (``mask[i] = 1``) only where the observation window covers the horizon, or where the
    resumption itself ended the window before the horizon (the speaker did come back, so "speaks
    within h" is known for every longer h). A resumption *after* the window closed (the agent had
    taken the floor, or the clip ended first) proves nothing about continued waiting and stays masked.
    Float noise is absorbed: negative inputs count as 0.

    ``ended_by_resumption`` says whether the resumption is the event that closed the window
    (``observation_end_reason == "customer_resumed"``). Pass it whenever the reason is known: with a
    window of 0 the numbers alone cannot tell "resumed on the next sample" from "the agent started
    first and the speaker came back at the cut", and only the former is observed under waiting.
    """
    tto = None if time_to_onset is None else max(0.0, float(time_to_onset))
    window = max(0.0, float(future_observed_until))
    if ended_by_resumption is None:
        resumed_in_window = tto is not None and tto <= window + tol
    else:
        resumed_in_window = tto is not None and bool(ended_by_resumption)
    fvad = [int(tto is not None and tto <= h) for h in HORIZONS]
    mask = [int(h <= window + tol or (resumed_in_window and tto <= h)) for h in HORIZONS]
    return fvad, mask


def observation_window(
    cut: float,
    *,
    resumption: float | None = None,
    agent_onset: float | None = None,
    clip_end: float | None = None,
    max_hold: float | None = None,
) -> tuple[float, str]:
    """``(future_observed_until, observation_end_reason)`` for a cut at ``cut`` seconds.

    The earliest event after the cut ends the observation; ties resolve in the order listed
    (the speaker's own resumption is the most specific). A cut placed after the event yields a
    window of 0 with that event's reason, so the row can be flagged instead of dropped.
    """
    events = [
        (resumption, "customer_resumed"),
        (agent_onset, "agent_onset"),
        (clip_end, "clip_end"),
        (max_hold, "max_hold"),
    ]
    known = [(float(t), reason) for t, reason in events if t is not None]
    if not known:
        return 0.0, "other"
    end, reason = min(known, key=lambda e: e[0])
    return max(0.0, end - float(cut)), reason


def hold_cut(cut: float, onset: float, margin: float = ONSET_MARGIN_S, tol: float = 1e-9) -> float | None:
    """Admissible cut time for a HOLD sample, or ``None`` when the grid point lies past ``onset - margin``.

    ``tol`` absorbs float noise of grid arithmetic (``pause_start + g`` landing 1e-16 past the onset);
    the returned cut is never past ``onset - margin``, so ``onset - cut >= margin`` holds exactly and
    ``time_to_onset`` can never be negative at write time (fix D1 of the L006 ledger).
    """
    if margin < 0:
        raise ValueError("onset margin must be non-negative")
    limit = float(onset) - float(margin)
    if cut > limit + tol:
        return None
    return min(float(cut), limit)
