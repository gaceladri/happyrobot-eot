"""Sample schema shared by every labeler: clips in, pause-level causal samples out.

Every sample carries multi-horizon future-speech targets (``fvad``): "will the user speak again
within h seconds?" for h in ``HORIZONS``. Targets are derived from observed timestamps only; a
censored future masks the corresponding horizons instead of assuming silence.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field

import numpy as np

from eot.audio import SAMPLE_RATE


HORIZONS: tuple[float, ...] = (0.24, 0.64, 1.2, 2.0)


DEFAULT_SCORE_POINT = 0.2


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


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
    kind: str  # "internal" | "final" | "whole"
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
    extra: dict = field(default_factory=dict)  # source-specific fields (accent, speaker_id, ...)

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("audio")
        extra = d.pop("extra") or {}
        for key, value in extra.items():
            d.setdefault(key, value)
        return d


def fvad_targets(time_to_onset: float | None, is_eot: bool = False,
                 observed_until: float = 0.0) -> tuple[list[int], list[int]]:
    """Future activity is observed independently of the conversational EOT label.

    With a known next onset every horizon is known. With right censoring only horizons
    within the observed silent future are valid. Clip completion alone proves nothing.
    """
    if time_to_onset is not None:
        if time_to_onset < -1e-9:
            raise ValueError("cut is after the next speech onset")
        return [int(time_to_onset <= h) for h in HORIZONS], [1] * len(HORIZONS)
    return [0] * len(HORIZONS), [int(h <= observed_until) for h in HORIZONS]
