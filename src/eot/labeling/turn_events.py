"""Annotation-derived training targets from dual-channel conversational event labels.

Built for the otoSpeech full-duplex corpus (17 event categories per speaker channel). These are
conservative *offline* silver targets: no model ever receives future events, transcripts or
labels; SRT segment edges are not acoustic cuts (the caller establishes silence acoustically);
interruptions, overlap and channel bleed abstain instead of becoming forced negatives.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

BACKCHANNELS = {"Acknowledgement Backchannel", "Continuer Backchannel", "Reaction Backchannel"}
INTERRUPTIONS = {
    "Floor-taking Competitive Interruption", "Floor-taking Cooperative Interruption",
    "Non-floor Taking Competitive Interruption", "Non-floor Taking Cooperative Interruption", "Overlap",
}
NON_SPEECH = {"Awkward Silence", "Channel Bleed", "Laughter", "Non-Speech Noise", "Speech, Non-Linguistic"}
FLOOR = {"Normal Turn", "Bounded Response"}
HOLDS = {"Strong Floor Hold", "Filler"}
FLOOR_OR_HOLDS = FLOOR | HOLDS
KNOWN = BACKCHANNELS | INTERRUPTIONS | NON_SPEECH | FLOOR_OR_HOLDS
_TIMESTAMP_RE = re.compile(r"(\d+):([0-5]\d):([0-5]\d)[,.](\d{3})")
_EVENT_RE = re.compile(r"\[([^\]]+)\]\s*(.*)", re.DOTALL)


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    speaker: str
    kind: str
    text: str = ""


@dataclass(frozen=True)
class Target:
    label: int | None  # 1 EOT, 0 HOLD, None abstain
    reason: str
    label_confidence: str = "annotation_derived_unreviewed"


def timestamp(value: str) -> float:
    m = _TIMESTAMP_RE.fullmatch(value.strip())
    if not m:
        raise ValueError("invalid SRT timestamp")
    h, minute, s, ms = map(int, m.groups())
    return h * 3600 + minute * 60 + s + ms / 1000


def parse_srt(text: str, speaker: str) -> list[Event]:
    """Parse one speaker channel's SRT where every cue reads ``[Event Kind] transcript``."""
    events = []
    for block in re.split(r"\n\s*\n", text.lstrip("﻿").replace("\r\n", "\n").strip()):
        if not block.strip():
            continue
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise ValueError("malformed SRT block")
        times = lines[1].split("-->")
        if len(times) != 2:
            raise ValueError("malformed SRT interval")
        start, end = map(timestamp, times)
        if end <= start:
            raise ValueError("nonpositive SRT interval")
        m = _EVENT_RE.fullmatch(" ".join(lines[2:]))
        if not m or m[1] not in KNOWN:
            raise ValueError("unknown or missing conversational event")
        events.append(Event(start, end, speaker, m[1], m[2]))
    return sorted(events, key=lambda e: (e.start, e.end))


def label_pause(
    *,
    pause_start: float,
    cut_time: float,
    speaker: str,
    events: list[Event],
    observed_until: float,
    next_own_onset: float | None,
    boundary_tolerance: float = 0.25,
) -> Target:
    """Label one acoustically established pause of ``speaker`` using both channels' annotations.

    ``next_own_onset`` is the speaker's own acoustic resumption (None if none was observed before
    ``observed_until``). A ``Normal Turn`` segment ending does not by itself imply EOT: the floor
    must visibly transfer, with at least one second of observed future to confirm it.
    """
    finite = all(math.isfinite(v) for v in (pause_start, cut_time, observed_until, boundary_tolerance))
    if not finite or not 0 <= pause_start < cut_time <= observed_until:
        raise ValueError("invalid observed causal cut")
    if next_own_onset is not None and (
        not math.isfinite(next_own_onset) or next_own_onset < cut_time or next_own_onset > observed_until
    ):
        raise ValueError("cut after onset or onset beyond observed future")
    if boundary_tolerance < 0 or any(
        e.kind not in KNOWN or not e.speaker or not math.isfinite(e.start) or not math.isfinite(e.end)
        or not 0 <= e.start < e.end
        for e in events
    ):
        raise ValueError("invalid event annotation")

    own = [e for e in events if e.speaker == speaker and e.start <= pause_start and e.end >= pause_start - boundary_tolerance]
    if not own:
        return Target(None, "unannotated_boundary")
    if len(own) != 1:
        return Target(None, "ambiguous_own_annotation")
    prior = own[0]
    if prior.kind not in FLOOR_OR_HOLDS:
        return Target(None, "not_current_floor_holder")

    horizon = min(observed_until, next_own_onset if next_own_onset is not None else observed_until, pause_start + 5.0)
    listener = [e for e in events if e.speaker != speaker and e.end > pause_start and e.start < horizon]
    if any(e.start < pause_start or e.kind in INTERRUPTIONS or e.kind == "Channel Bleed" for e in listener):
        return Target(None, "overlap_interruption_or_bleed")
    if any(e.kind in HOLDS for e in listener):
        return Target(None, "uncertain_listener_floor")

    floor = sorted((e for e in listener if e.kind in FLOOR), key=lambda e: e.start)
    if floor:
        first = floor[0]
        if prior.kind in HOLDS:
            return Target(None, "hold_transfer_conflict")
        if observed_until < first.start + 1.0:
            return Target(None, "censored_transfer_confirmation")
        if next_own_onset is not None and next_own_onset < first.start + 1.0:
            return Target(None, "collision")
        return Target(1, "annotated_floor_transfer")
    if next_own_onset is not None and next_own_onset - pause_start <= 5.0:
        backchannel = any(e.kind in BACKCHANNELS for e in listener)
        return Target(0, "own_continuation_after_backchannel" if backchannel else "own_continuation")
    return Target(None, "unresolved_or_censored_future")
