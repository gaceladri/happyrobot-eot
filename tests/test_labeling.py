"""Labeling pipeline tests: robust pause detector, dual-channel oracle rules, Krisp conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eot.labeling.apptek import Segment, decide, merge_intervals, speech_intervals, words_in
from eot.audio import (
    SAMPLE_RATE,
    PauseDetector,
    PauseSpan,
    Span,
    add_room_tone,
    digital_silence_fraction,
    ensemble_pauses,
    silence_spans,
    silence_spans_adaptive,
    vad_quiet_mask,
)
from eot.labeling.prefix_mining import mine_clip
from eot.labeling.samples import Clip

SR = SAMPLE_RATE


def _tone_speech(duration: float, rng: np.random.Generator, level: float = 0.3) -> np.ndarray:
    """Speech-like: amplitude-modulated harmonic burst with syllabic envelope."""
    t = np.arange(int(duration * SR)) / SR
    env = 0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 4.0 * t))  # 4 Hz syllables, never fully off
    sig = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, np.pi)) for f in (140, 280, 420, 900, 1800))
    return (level * env * sig / 5).astype(np.float32)


def _clip_with_pauses(rng, speech=(1.0, 0.8, 1.2), pauses=(0.4, 0.3), tail=0.5, noise_db=-70.0):
    parts, edges, t = [], [], 0.0
    for i, s in enumerate(speech):
        parts.append(_tone_speech(s, rng))
        t += s
        if i < len(pauses):
            edges.append((t, t + pauses[i]))
            parts.append(np.zeros(int(pauses[i] * SR), np.float32))
            t += pauses[i]
    parts.append(np.zeros(int(tail * SR), np.float32))
    x = np.concatenate(parts)
    x = x + rng.normal(0, 10 ** (noise_db / 20), size=len(x)).astype(np.float32)
    return x.astype(np.float32), edges


def test_adaptive_energy_finds_pauses_under_stationary_noise():
    rng = np.random.default_rng(0)
    clean, edges = _clip_with_pauses(rng)
    noisy, _ = _clip_with_pauses(np.random.default_rng(0), noise_db=-32.0)  # ~20 dB SNR
    for x in (clean, noisy):
        spans = [s for s in silence_spans_adaptive(x, SR, min_silence=0.2) if s.start > 0.1 and s.end < len(x) / SR - 0.1]
        assert len(spans) == len(edges), spans
        for s, (a, b) in zip(spans, edges):
            assert abs(s.start - a) < 0.08 and abs(s.end - b) < 0.08


def test_legacy_detector_loses_pauses_under_noise_the_adaptive_keeps():
    """Documents the failure that motivated the ensemble: peak-relative threshold vs noise floor."""
    noisy, edges = _clip_with_pauses(np.random.default_rng(1), noise_db=-32.0)  # ~12 dB SNR
    last_speech_end = edges[-1][1] + 1.2  # third speech segment ends here; after it is the trailing tail
    legacy = [s for s in silence_spans(noisy, SR, min_silence=0.2) if 0.1 < s.start < last_speech_end - 0.1]
    adaptive = [s for s in silence_spans_adaptive(noisy, SR, min_silence=0.2) if 0.1 < s.start < last_speech_end - 0.1]
    assert len(adaptive) == len(edges)
    assert len(legacy) < len(edges)


def test_ensemble_grades_by_vad_lag_and_vetoes_speech():
    chunk = 0.032
    n = int(3.0 / chunk)
    quiet = np.zeros(n, bool)
    quiet[int(1.0 / chunk) : int(1.5 / chunk)] = True  # VAD quiet 1.0-1.5
    quiet[int(2.3 / chunk) : int(2.6 / chunk)] = True  # VAD quiet 2.3-2.6, energy pause starts at 2.0
    energy = [Span(1.0, 1.5), Span(2.0, 2.6), Span(2.7, 2.9)]
    out = ensemble_pauses(energy, quiet, chunk_seconds=chunk, min_silence=0.2, total=3.0)
    by_start = {round(p.start, 2): p for p in out}
    assert by_start[1.0].confidence == "high" and by_start[1.0].iou > 0.9
    assert by_start[2.0].confidence == "low"  # VAD quiet only 0.3 s into a 0.6 s pause -> lag 0.3 > 0.232
    assert by_start[2.7].confidence == "low" and by_start[2.7].vad is None  # VAD hears speech throughout


def test_ensemble_tolerates_vad_leading_energy_by_one_chunk():
    chunk = 0.032
    n = int(3.0 / chunk)
    quiet = np.zeros(n, bool)
    quiet[int(1.0 / chunk) : int(1.5 / chunk) - 1] = True  # VAD flips to speech one chunk before energy does
    out = ensemble_pauses([Span(1.0, 1.5)], quiet, chunk_seconds=chunk, min_silence=0.2, total=3.0)
    assert out[0].confidence == "high"


def test_vad_quiet_mask_hysteresis():
    probs = np.array([0.1, 0.6, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1])
    q = vad_quiet_mask(probs, onset=0.5, offset=0.35, min_speech=0.1, chunk_seconds=0.032)
    assert q[0] and not q[1] and not q[2]  # stays speech between offset and onset
    assert q[4] and not q[9] and q[13]


@pytest.mark.parametrize("detector", ["legacy", "energy"])
def test_mine_clip_emits_confidence_metadata(detector):
    rng = np.random.default_rng(2)
    x, _ = _clip_with_pauses(rng)
    clip = Clip(id="c", audio=x, label=1, source="unit")
    samples = mine_clip(clip, grid=[0.2, 0.4], detector=PauseDetector(detector))
    internal = [s for s in samples if s.kind == "internal"]
    assert internal, "expected internal HOLD samples"
    for s in samples:
        meta = s.meta()
        assert meta["detector"] == detector
        assert meta["cut_confidence"] in ("high", "medium", "low")
        assert "extra" not in meta
    assert all(s.onset_margin_ms is not None and s.onset_margin_ms >= 0 for s in internal)


def test_mine_clip_drops_low_confidence_when_requested():
    class FakeDetector:
        detector = "fake"

        def __call__(self, x, sr):
            total = len(x) / sr
            return [
                PauseSpan(1.0, 1.4, 1.4, 0.1, "low", Span(1.0, 1.4), None, 1.4),
                PauseSpan(2.2, 2.5, 2.5, 0.9, "high", Span(2.2, 2.5), None, 2.2),
                PauseSpan(total - 0.5, total, total, 1.0, "high", Span(total - 0.5, total), None, total - 0.5),
            ]

    x = np.random.default_rng(0).normal(0, 0.1, size=4 * SR).astype(np.float32)
    from collections import Counter

    stats = Counter()
    samples = mine_clip(Clip("c", x, 1, "u"), grid=[0.2], detector=FakeDetector(), stats=stats)
    starts = {round(s.pause_start, 1) for s in samples if s.kind == "internal"}
    assert starts == {2.2}
    assert stats["dropped_internal_low"] == 1


def test_room_tone_removes_digital_silence():
    rng = np.random.default_rng(0)
    x = np.concatenate([_tone_speech(1.0, rng), np.zeros(SR, np.float32), _tone_speech(1.0, rng)])
    assert digital_silence_fraction(x) > 0.3
    y = add_room_tone(x, rng, snr_db=(20.0, 20.0))
    assert digital_silence_fraction(y) == 0.0
    assert len(y) == len(x) and np.abs(y).max() <= 1.0


# ---------------------------------------------------------------------------
# AppTek oracle rules
# ---------------------------------------------------------------------------

SEGS = [
    Segment(0.0, 2.0, "agent", "a", "Hi. How can I help you today?"),
    Segment(2.5, 8.0, "customer", "c", "I am calling about my bill (uh) it is too high."),
    Segment(9.0, 12.0, "agent", "a", "Sure, let me check that for you right now."),
    Segment(12.5, 14.0, "customer", "c", "okay thanks"),
]


def _pause(start, end, onset=None):
    onset = end if onset is None else onset
    return PauseSpan(start, end, onset, 1.0, "high", Span(start, end), Span(start, end), start)


def test_oracle_hold_when_customer_resumes_first():
    cust = [Span(2.5, 5.0), Span(5.4, 8.0), Span(12.5, 14.0)]
    agent = [Span(0.0, 2.0), Span(9.0, 12.0)]
    d = decide(_pause(5.0, 5.4), cust, agent, SEGS, 0.0, 15.0)
    assert d.label == 0 and d.reason == "hold" and d.onset == 5.4


def test_oracle_eot_when_agent_takes_the_floor():
    cust = [Span(2.5, 8.0), Span(12.5, 14.0)]
    agent = [Span(0.0, 2.0), Span(9.0, 12.0)]
    d = decide(_pause(8.0, 12.5), cust, agent, SEGS, 0.0, 15.0)
    assert d.label == 1 and d.reason == "eot"
    assert d.agent_onset == 9.0 and d.agent_words >= 3


def test_oracle_backchannel_is_hold():
    cust = [Span(2.5, 5.0), Span(6.0, 8.0), Span(12.5, 14.0)]
    agent = [Span(0.0, 2.0), Span(5.2, 5.5), Span(9.0, 12.0)]  # 0.3 s "mhm", unannotated -> 0 words
    d = decide(_pause(5.0, 6.0), cust, agent, SEGS, 0.0, 15.0)
    assert d.label == 0 and d.reason == "hold_backchannel"


def test_oracle_exclusions():
    agent = [Span(0.0, 2.0), Span(9.0, 12.0)]
    # collision: agent starts a real turn but the customer is back within 1 s
    cust = [Span(2.5, 8.0), Span(9.5, 10.5)]
    assert decide(_pause(8.0, 9.5), cust, agent, SEGS, 0.0, 15.0).reason == "collision"
    # agent already talking when the customer stopped
    cust = [Span(2.5, 9.5), Span(12.5, 14.0)]
    assert decide(_pause(9.5, 12.5), cust, agent, SEGS, 0.0, 15.0).reason == "overlap_at_pause"
    # nothing after: final pause
    cust = [Span(2.5, 8.0)]
    assert decide(_pause(8.0, 15.0), cust, [Span(0.0, 2.0)], SEGS, 0.0, 15.0).reason == "final_pause"
    # too little customer speech before
    cust = [Span(2.5, 2.6), Span(3.0, 8.0)]
    assert decide(_pause(2.6, 3.0), cust, agent, SEGS, 0.0, 15.0).reason == "no_speech_before"
    # before the agent greeting -> technical
    assert decide(_pause(5.0, 5.4), [Span(2.5, 5.0), Span(5.4, 8.0)], agent, SEGS, 6.0, 15.0).reason == "technical"
    # long pause with nothing from the agent
    cust = [Span(2.5, 8.0), Span(14.0, 15.0)]
    assert decide(_pause(8.0, 14.0), cust, [Span(0.0, 2.0)], SEGS, 0.0, 15.0).reason == "long_pause"


def test_helpers():
    assert merge_intervals([Span(0, 1), Span(1.2, 2), Span(3, 4)], 0.3) == [Span(0, 2), Span(3, 4)]
    pauses = [_pause(1.0, 1.5), PauseSpan(2.0, 2.4, 2.4, 0.0, "low", Span(2.0, 2.4), None, 2.4)]
    assert speech_intervals(pauses, 3.0) == [Span(0.0, 1.0), Span(1.5, 3.0)]
    assert words_in(SEGS, "agent", Span(9.0, 12.0)) == 9
    assert Segment(0, 1, "customer", "c", "(uh) okay so").words == 2


# ---------------------------------------------------------------------------
# Krisp conversion
# ---------------------------------------------------------------------------


def test_krisp_score_rows_and_metrics(tmp_path: Path):
    import soundfile as sf

    from eot.eval.krisp import metrics, score

    rng = np.random.default_rng(0)
    clips = tmp_path / "clips"
    (clips / "audio").mkdir(parents=True)
    rows = []
    for i, (label, tail) in enumerate([("shift", 0.8), ("hold", 0.5), ("hold", 0.1), ("shift", 1.3)]):
        x = np.concatenate([_tone_speech(1.0, rng), np.zeros(int(tail * SR), np.float32)])
        sf.write(clips / "audio" / f"{i}.wav", x, SR)
        rows.append({"id": str(i), "path": f"audio/{i}.wav", "label": label, "duration": 1 + tail, "last_silence_duration": tail, "speaker_id": str(i % 2)})
    (clips / "clips.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    class Adapter:
        adapter_id = "test"

        def predict_batch(self, batch):
            # confident only once >= 0.4 s of silence has elapsed
            return [1.0 if len(item["audio"]["array"]) / SR >= 1.4 else 0.0 for item in batch]

    out = tmp_path / "preds.jsonl"
    info = score(clips / "clips.jsonl", Adapter(), out)
    assert info["excluded_tail_lt_0.2"] == 1
    preds = [json.loads(l) for l in out.read_text().splitlines()]
    assert {p["id"] for p in preds} == {"0", "1", "3"}
    assert min(p["silence_dur"] for p in preds) == 0.2
    m = metrics(out, bootstrap_reps=20)
    assert m["n_hold"] == 1 and m["n_eot"] == 2
    ops = m["operating_points"]["best_latency_at_5pct"]
    assert ops is not None and ops["cutoff_rate"] == 0.0
    # Firing exactly when speech resumes is not a cutoff under the harness boundary rule.
    assert abs(ops["mean_latency"] - 0.5) < 1e-6
