"""Censoring-correct labeling (L009): observation window fields, per-horizon masks and the D1 cut
guard in the sample helpers, prefix mining and the AppTek labeler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eot.audio import SAMPLE_RATE, PauseSpan, Span
from eot.io import write_wav
from eot.labeling.apptek import Conversation, Segment, label_conversation
from eot.labeling.prefix_mining import mine_clip
from eot.labeling.samples import (
    OBSERVATION_END_REASONS,
    Clip,
    Sample,
    fvad_targets,
    hold_cut,
    observation_window,
    observed_fvad_targets,
)

SR = SAMPLE_RATE
NEW_FIELDS = ("future_observed_until", "observation_end_reason", "clip_duration_s", "weight")


class _Detector:
    """Fake pause detector returning fixed spans (the VAD onset may precede the energy end)."""

    detector = "fake"

    def __init__(self, spans: list[PauseSpan]):
        self.spans = spans

    def __call__(self, x, sr):
        return list(self.spans)


def _pause(start: float, end: float, onset: float | None = None) -> PauseSpan:
    onset = end if onset is None else onset
    return PauseSpan(start, end, onset, 0.9, "high", Span(start, end), Span(start, onset), agree_from=start)


def _tone(duration: float, rng: np.random.Generator, level: float = 0.3) -> np.ndarray:
    t = np.arange(int(duration * SR)) / SR
    env = 0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 4.0 * t))
    sig = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, np.pi)) for f in (140, 280, 420, 900, 1800))
    return (level * env * sig / 5).astype(np.float32)


# ----------------------------------------------------------------------------- pure helpers
@pytest.mark.parametrize(
    "tto,window,fvad,mask",
    [
        (0.3, 0.3, [0, 1, 1, 1], [1, 1, 1, 1]),  # resumed at 0.3 s: every horizon known
        (0.0, 0.0, [1, 1, 1, 1], [1, 1, 1, 1]),  # resumed on the next sample
        (None, 0.7, [0, 0, 0, 0], [1, 1, 0, 0]),  # silent for 0.7 s, then censored
        (None, 0.0, [0, 0, 0, 0], [0, 0, 0, 0]),  # nothing observed (cut past the clip end)
        (None, 2.0, [0, 0, 0, 0], [1, 1, 1, 1]),  # the exact horizon counts as covered
        (2.3, 0.3, [0, 0, 0, 0], [1, 0, 0, 0]),  # resumed only after the window closed (agent took the floor)
        (0.8, 0.1, [0, 0, 1, 1], [0, 0, 0, 0]),  # backchannel at 0.1 s: later resumption proves nothing
        (-1e-15, 0.5, [1, 1, 1, 1], [1, 1, 1, 1]),  # float noise counts as 0
    ],
)
def test_observed_fvad_targets_assert_only_within_the_window(tto, window, fvad, mask):
    assert observed_fvad_targets(tto, window) == (fvad, mask)


def test_future_speech_does_not_follow_eot_label():
    assert fvad_targets(None) == ([0] * 4, [0] * 4)
    assert fvad_targets(None, observed_until=0.7) == ([0] * 4, [1, 1, 0, 0])
    assert fvad_targets(0.9) == ([0, 0, 1, 1], [1] * 4)


def test_future_target_rejects_cut_after_onset():
    with pytest.raises(ValueError, match="after the next speech onset"):
        fvad_targets(-0.02)


def test_observation_window_takes_the_earliest_event_and_never_goes_negative():
    assert observation_window(1.0, resumption=1.5, clip_end=9.0) == (0.5, "customer_resumed")
    assert observation_window(1.0, resumption=5.0, agent_onset=1.3, clip_end=9.0) == pytest.approx((0.3, "agent_onset"))
    assert observation_window(1.0, clip_end=1.2) == pytest.approx((0.2, "clip_end"))
    assert observation_window(1.0, agent_onset=0.8, clip_end=9.0) == (0.0, "agent_onset")  # cut after the event: flagged
    assert observation_window(1.0, resumption=2.0, agent_onset=2.0) == (1.0, "customer_resumed")  # tie: own channel wins
    assert observation_window(1.0, max_hold=6.0, clip_end=9.0) == (5.0, "max_hold")
    assert observation_window(1.0) == (0.0, "other")
    for _, reason in (observation_window(0.0, clip_end=1.0), observation_window(0.0)):
        assert reason in OBSERVATION_END_REASONS


def test_hold_cut_guard_never_places_a_cut_past_onset_minus_margin():
    assert hold_cut(1.4, 1.45) == 1.4
    assert hold_cut(1.5, 1.45) is None
    assert hold_cut(1.45 + 1e-12, 1.45) == 1.45  # float noise absorbed, never past the onset
    assert 1.45 - hold_cut(1.45 + 1e-12, 1.45) >= 0.0
    assert hold_cut(1.4, 1.45, margin=0.1) is None
    assert hold_cut(1.3, 1.45, margin=0.1) == 1.3
    assert 1.45 - hold_cut(1.35 + 1e-12, 1.45, margin=0.1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        hold_cut(1.0, 2.0, margin=-0.1)


def test_sample_defaults_keep_old_constructors_working_and_emit_new_columns():
    s = Sample(
        id="x",
        clip_id="c",
        audio=np.zeros(10, np.float32),
        label=0,
        kind="internal",
        cut_time=1.0,
        pause_start=0.8,
        time_to_onset=0.1,
        fvad=[1, 1, 1, 1],
        fvad_mask=1,
        source="s",
        agent_text="",
    )
    meta = s.meta()
    assert meta["weight"] == 1.0 and meta["future_observed_until"] is None and meta["observation_end_reason"] is None
    assert set(NEW_FIELDS) <= set(meta)


# ----------------------------------------------------------------------------- prefix mining
def test_mine_clip_writes_observation_window_on_every_kind():
    total = 2.0
    audio = np.ones(int(total * SR), np.float32) * 0.1
    det = _Detector([_pause(0.5, 1.0, onset=0.95), _pause(1.5, 2.0)])  # internal pause + trailing pause
    rows = mine_clip(Clip("c", audio, 1), detector=det, grid=[0.2, 0.4, 0.6])
    for r in rows:
        assert r.clip_duration_s == total and r.weight == 1.0 and r.observation_end_reason in OBSERVATION_END_REASONS
        assert r.future_observed_until is not None and r.future_observed_until >= 0.0
    internal = [r for r in rows if r.kind == "internal"]
    assert [r.cut_time for r in internal] == [0.7, 0.9]  # 1.1 lies past the onset 0.95: grid stops
    for r in internal:
        assert r.observation_end_reason == "customer_resumed"
        assert r.future_observed_until == pytest.approx(r.time_to_onset)
        assert r.time_to_onset == pytest.approx(0.95 - r.cut_time) and r.time_to_onset >= 0.0
        assert r.fvad_mask == [1, 1, 1, 1]
    final = [r for r in rows if r.kind == "final"]
    assert [round(r.cut_time, 6) for r in final] == [1.7, 1.9, 2.1]
    assert [r.future_observed_until for r in final] == pytest.approx([0.3, 0.1, 0.0])
    assert [r.fvad_mask for r in final] == [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # silence observed only inside the clip
    assert all(r.observation_end_reason == "clip_end" and r.fvad == [0, 0, 0, 0] for r in final)
    whole = [r for r in mine_clip(Clip("h", audio, 0), detector=det, grid=[0.2]) if r.kind == "whole"]
    assert len(whole) == 1 and whole[0].observation_end_reason == "clip_end"
    assert whole[0].future_observed_until == pytest.approx(0.3) and whole[0].fvad_mask == [1, 0, 0, 0]


def test_mine_clip_onset_margin_drops_cuts_near_the_onset_and_counts_them():
    from collections import Counter

    audio = np.ones(2 * SR, np.float32) * 0.1
    det = _Detector([_pause(0.5, 1.0, onset=0.95)])
    stats = Counter()
    rows = [
        r for r in mine_clip(Clip("c", audio, 1), detector=det, grid=[0.2, 0.4, 0.6], onset_margin=0.1, stats=stats) if r.kind == "internal"
    ]
    assert [r.cut_time for r in rows] == [0.7]  # 0.9 is within 0.1 s of the onset
    assert stats["dropped_cut_past_onset_margin"] == 1
    assert all(r.time_to_onset >= 0.1 for r in rows)


def test_mine_clip_cut_landing_on_the_onset_by_float_noise_is_clamped_not_negative():
    audio = np.ones(2 * SR, np.float32) * 0.1
    start = 0.1 + 0.2 + 0.1  # 0.4000000000000001 in binary floating point
    det = _Detector([_pause(start, start + 0.2, onset=0.6)])
    rows = [r for r in mine_clip(Clip("c", audio, 1), detector=det, grid=[0.2]) if r.kind == "internal"]
    assert len(rows) == 1 and rows[0].time_to_onset == 0.0 and rows[0].cut_time <= 0.6


def test_mining_does_not_cut_after_earlier_vad_onset():
    class Detector:
        detector = "fake"

        def __call__(self, x, sr):
            return [PauseSpan(0.5, 0.72, 0.69, 0.9, "high", Span(0.5, 0.72), Span(0.52, 0.69), 0.52)]

    samples = mine_clip(Clip("boundary", np.ones(16000, np.float32) * 0.1, 1), detector=Detector())
    assert not [s for s in samples if s.kind == "internal"]


# ----------------------------------------------------------------------------- apptek
def _apptek_fixture(tmp_path: Path) -> tuple[Conversation, _Detector, float]:
    rng = np.random.default_rng(0)
    total = 9.5
    customer = rng.normal(0, 1e-4, int(total * SR)).astype(np.float32)
    for a, b in ((0.5, 1.5), (2.0, 3.0), (5.5, 6.0), (6.5, 7.5), (8.5, 9.0)):
        customer[int(a * SR) : int(b * SR)] += _tone(b - a, rng)
    agent = rng.normal(0, 1e-4, int(total * SR)).astype(np.float32)
    agent[int(3.5 * SR) : int(4.5 * SR)] += _tone(1.0, rng)  # turn-like reply -> EOT for the pause at 3.0
    agent[int(7.8 * SR) : int(8.1 * SR)] += _tone(0.3, rng)  # short backchannel inside the pause at 7.5
    write_wav(tmp_path / "customer.wav", customer, SR)
    write_wav(tmp_path / "agent.wav", agent, SR)
    segments = [
        Segment(0.0, 0.4, "agent", "a", "hello how can I help you"),
        Segment(0.5, 1.5, "customer", "c", "I would like to book a flight"),
        Segment(2.0, 3.0, "customer", "c", "to London next week"),
        Segment(3.5, 4.5, "agent", "a", "sure let me check that for you"),
        Segment(5.5, 6.0, "customer", "c", "thanks"),
        Segment(6.5, 7.5, "customer", "c", "and I also need"),
        Segment(7.8, 8.1, "agent", "a", "mm hm"),
        Segment(8.5, 9.0, "customer", "c", "a hotel"),
    ]
    conv = Conversation("conv", "en-AU", "aviation", total, segments, tmp_path / "agent.wav", tmp_path / "customer.wav")
    # Pause A: VAD hears the resumption 150 ms before the energy detector (D1-ext case).
    # Pause B: EOT, agent onset 3.5. Pause C: backchannel HOLD, agent onset 7.8, customer back at 8.5.
    det = _Detector([_pause(1.5, 2.0, onset=1.85), _pause(3.0, 5.5), _pause(7.5, 8.5)])
    return conv, det, total


def test_apptek_rows_carry_the_wait_window_and_hold_cuts_respect_the_vad_onset(tmp_path: Path):
    conv, det, total = _apptek_fixture(tmp_path)
    samples, decisions, summary = label_conversation(conv, det, grid=[0.2, 0.4, 0.6])
    assert [d.reason for d in decisions] == ["hold", "eot", "hold_backchannel"]
    metas = [s.meta() for s in samples]
    for m in metas:
        assert m["clip_duration_s"] == total and m["weight"] == 1.0 and "agent_onset_s" in m
        assert m["future_observed_until"] >= 0.0 and m["observation_end_reason"] in OBSERVATION_END_REASONS
        assert m["time_to_onset"] is None or m["time_to_onset"] >= 0.0
    hold = [m for m in metas if m["kind"] == "oracle_hold" and not m["backchannel"]]
    assert [m["cut_time"] for m in hold] == [1.7]  # 1.9 lies past the VAD onset 1.85 (energy end 2.0): dropped
    assert hold[0]["time_to_onset"] == pytest.approx(0.15) and hold[0]["future_observed_until"] == pytest.approx(0.15)
    assert hold[0]["observation_end_reason"] == "customer_resumed" and hold[0]["agent_onset_s"] is None
    eot = [m for m in metas if m["kind"] == "oracle_eot"]
    assert [m["cut_time"] for m in eot] == [3.2, 3.4, 3.6] and all(m["agent_onset_s"] == 3.5 for m in eot)
    assert [m["future_observed_until"] for m in eot] == pytest.approx([0.3, 0.1, 0.0])  # agent-first row kept, flagged
    assert all(m["observation_end_reason"] == "agent_onset" for m in eot)
    assert [m["fvad_mask"] for m in eot] == [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert all(m["fvad"][:3] == [0, 0, 0] for m in eot) and eot[2]["time_to_onset"] == pytest.approx(1.9)
    bc = [m for m in metas if m["backchannel"]]
    assert [m["cut_time"] for m in bc] == [7.7, 7.9, 8.1] and all(m["agent_onset_s"] == 7.8 for m in bc)
    assert [m["future_observed_until"] for m in bc] == pytest.approx([0.1, 0.0, 0.0])
    assert all(m["observation_end_reason"] == "agent_onset" and m["fvad_mask"] == [0, 0, 0, 0] for m in bc)
    assert [m["time_to_onset"] for m in bc] == pytest.approx([0.8, 0.6, 0.4])
    assert summary["n_pauses"] == 3 and summary["labeled_pauses_without_sample"] == {}


def test_apptek_onset_margin_applies_to_hold_rows(tmp_path: Path):
    conv, det, _ = _apptek_fixture(tmp_path)
    samples, _, summary = label_conversation(conv, det, grid=[0.2, 0.4, 0.6], onset_margin=0.2)
    hold = [s for s in samples if s.kind == "oracle_hold" and not s.extra["backchannel"]]
    assert hold == []  # 1.7 is within 0.2 s of the onset 1.85
    assert [s.cut_time for s in samples if s.kind == "oracle_eot"] == [3.2, 3.4, 3.6]  # EOT grid unchanged
    assert summary["labeled_pauses_without_sample"] == {"hold": 1}  # the silent exclusion is counted
