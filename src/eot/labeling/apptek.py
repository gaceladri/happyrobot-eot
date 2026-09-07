"""AppTek Call-Center Dialogues: dual-channel heuristic labeling of customer pauses.

Why this source. Smart Turn labels are clip-level and single-channel: "did the same speaker say
more?" AppTek gives us the thing a production endpointer is actually trying to predict, on
separate, sample-aligned channels: *what did the other human do at this pause?* If the agent
took the floor, the pause was an end of turn; if the customer carried on, it was a hold. No
LLM judge or clip boundary decides the label. "Oracle" in historical artifact names means
access to future activity on both channels, not certain conversational intent. Human agents can
interrupt, and the overlap/collision exclusions select a subset of the real task.
The manual diarization (`segments`) is
used only for text (word counts, filler flags, technical-noise filter), never for timing.

Rules (fixed before training; every exclusion is counted and reported):

| after the customer pause                                          | label | flag              |
|-------------------------------------------------------------------|-------|-------------------|
| customer resumes before the agent starts                          | HOLD  |                   |
| agent utterance >= 0.6 s or >= 3 words, customer silent >= 1.0 s   | EOT   | eot_gap           |
| agent utterance < 0.6 s and <= 2 words, then customer continues   | HOLD  | backchannel=1     |
| agent turn-like but customer back within 1.0 s of agent onset     | drop  | collision         |
| agent speaking when the customer stopped                          | drop  | overlap_at_pause  |
| customer resumes > 5 s later and no agent activity                | drop  | long_pause        |
| agent replies < 0.2 s after the customer stopped                  | drop  | too_short         |
| last pause of the call / nothing after                            | drop  | final_pause       |
| technical role-play chatter nearby or before the agent's greeting | drop  | technical         |

Observation window (rev. 4). Every row records ``future_observed_until`` / ``observation_end_reason``:
the customer channel counts as observed under *continued waiting* only until the agent takes the
floor (``agent_onset``), the customer resumes (``customer_resumed``) or the file ends (``clip_end``),
whichever comes first. A HOLD cut is never placed past ``min(pause.end, pause.onset) - onset_margin``
(fix D1 of the L006 ledger); EOT cuts after the agent onset are kept with a window of 0 and
``agent_onset_s`` so training can exclude or weight them. A labeled pause whose grid points all lie
before the VAD agreement or past the resumption minus the margin yields no sample; such pauses are
counted by reason in ``labeled_pauses_without_sample`` (conversation summary and report.json).

Known distribution shifts (labeling plan, archived branch `archive/research`, output/LABELING_PIPELINE_PLAN.md §4): digital silence between
utterances (VoIP gating; every sample carries ``noise_fill=1`` so training adds room tone),
human-human role-play with slow turn exchanges, frequent overlap. AppTek is a *declared demo*:
its card says the corpus is not intended for training.

CLI
---
    uv run eot-apptek --root data/raw/apptek/hf --out data/mined/apptek --workers 8
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from eot.audio import (
    SAMPLE_RATE,
    SCORE_POINT,
    WINDOW_SECONDS,
    PauseDetector,
    PauseSpan,
    Span,
    digital_silence_fraction,
    estimate_snr_db,
    load_wav,
    silence_spans_adaptive,
)
from eot.io import atomic_write_json, read_jsonl_records, write_wav
from eot.labeling.prefix_mining import build_detector
from eot.labeling.samples import (
    CONFIDENCE_RANK,
    EOT_CUSTOMER_SILENT_S,
    MAX_HOLD_S,
    ONSET_MARGIN_S,
    Sample,
    hold_cut,
    observation_window,
    observed_fvad_targets,
)

HELDOUT_ACCENTS: frozenset[str] = frozenset({"en-IN", "en-SG", "en-GB_SCT"})  # CLI default; ``--heldout`` overrides


# Role-play technical chatter ("is your audio on?", "it keeps calibrating"). Deliberately narrow:
# business vocabulary such as "connection" or "line" must not trigger it in telecom domains.
TECH_RE = re.compile(
    r"\b(audio|mic|microphone|calibrat\w*|can you hear|hear me|you're (on )?mute|recording|"
    r"breaking up|echo)\b",
    re.I,
)
MIN_AGENT_UTT_S = 0.15  # shorter agent-channel activity is a click or breath, not speech
FILLER_RE = re.compile(r"\((uh|um|hmm|mm|er|ah|eh)\)|#\w+|\w+~", re.I)
WORD_RE = re.compile(r"[A-Za-z0-9']+")


# Oracle thresholds (pre-registered; do not tune after seeing test results). EOT_CUSTOMER_SILENT_S
# and MAX_HOLD_S are shared with :mod:`eot.labeling.turn_events` and live in :mod:`eot.labeling.samples`.
BACKCHANNEL_MAX_S = 0.6
BACKCHANNEL_MAX_WORDS = 2
MIN_PAUSE_S = 0.2
MIN_SPEECH_BEFORE_S = 0.3
AGENT_MERGE_GAP_S = 0.3
TECH_WINDOW_S = 10.0
GREETING_MIN_WORDS = 3


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    role: str
    speaker_id: str
    text: str

    @property
    def words(self) -> int:
        return len(WORD_RE.findall(re.sub(r"\([^)]*\)", " ", self.text)))


@dataclass
class Conversation:
    id: str
    accent: str
    domain: str
    duration: float
    segments: list[Segment]
    agent_path: Path
    customer_path: Path

    @property
    def customer_speaker(self) -> str:
        for s in self.segments:
            if s.role == "customer":
                return s.speaker_id
        return "unknown"


@dataclass
class Decision:
    pause_start: float
    pause_end: float  # energy pause end (customer's own next onset or file end)
    onset: float | None  # customer resumes (None if never)
    agent_onset: float | None
    agent_dur: float | None
    agent_words: int | None
    label: int | None  # 1 EOT, 0 HOLD, None excluded
    reason: str  # "hold" | "eot" | "hold_backchannel" | exclusion reason
    confidence: str
    iou: float


def load_conversations(root: Path, accents: Iterable[str] | None = None) -> list[Conversation]:
    root = Path(root)
    out: list[Conversation] = []
    for meta in sorted((root / "diarization").glob("*/metadata.jsonl")):
        accent = meta.parent.name
        if accents is not None and accent not in accents:
            continue
        for row in read_jsonl_records(meta):
            stem = Path(row["file_name"]).stem
            audio_dir = root / "test" / accent / "audio"
            agent, customer = audio_dir / f"{stem}_channel1.wav", audio_dir / f"{stem}_channel2.wav"
            if not (agent.exists() and customer.exists()):
                continue
            segments = [
                Segment(float(s["start"]), float(s["end"]), str(s["role"]), str(s.get("speaker_id", "")), str(s.get("text", "")))
                for s in row["segments"]
            ]
            out.append(Conversation(stem, accent, str(row.get("domain", "")), float(row["duration"]), segments, agent, customer))
    return out


def speech_intervals(pauses: list[PauseSpan], total: float, min_confidence: str = "medium") -> list[Span]:
    """Complement of confident pauses. Low-confidence pauses count as speech (VAD heard speech)."""
    rank = CONFIDENCE_RANK[min_confidence]
    t = 0.0
    out: list[Span] = []
    for p in sorted(pauses, key=lambda p: p.start):
        if CONFIDENCE_RANK[p.confidence] < rank:
            continue
        if p.start > t + 1e-6:
            out.append(Span(t, p.start))
        t = max(t, p.end)
    if total > t + 1e-6:
        out.append(Span(t, total))
    return out


def merge_intervals(spans: list[Span], gap: float) -> list[Span]:
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: s.start):
        if out and s.start - out[-1].end <= gap:
            out[-1] = Span(out[-1].start, max(out[-1].end, s.end))
        else:
            out.append(s)
    return out


def words_in(segments: list[Segment], role: str, span: Span) -> int:
    """Words of ``role`` segments overlapping ``span``, pro-rated by temporal overlap."""
    total = 0.0
    for s in segments:
        if s.role != role or s.end <= span.start or s.start >= span.end:
            continue
        overlap = min(s.end, span.end) - max(s.start, span.start)
        total += s.words * overlap / max(1e-6, s.end - s.start)
    return int(round(total))


def technical_nearby(segments: list[Segment], t: float, window: float = TECH_WINDOW_S) -> bool:
    return any(abs((s.start + s.end) / 2 - t) <= window and TECH_RE.search(s.text) for s in segments)


def first_agent_greeting(segments: list[Segment]) -> float:
    for s in segments:
        if s.role == "agent" and s.words >= GREETING_MIN_WORDS and not TECH_RE.search(s.text):
            return s.start
    return 0.0


def last_text_before(segments: list[Segment], role: str, t: float, within: float = 1.5) -> Segment | None:
    best = None
    for s in segments:
        if s.role == role and s.end <= t + 0.05 and t - s.end <= within:
            if best is None or s.end > best.end:
                best = s
    return best


def decide(
    pause: PauseSpan,
    customer_speech: list[Span],
    agent_utts: list[Span],
    segments: list[Segment],
    greeting_t: float,
    total: float,
) -> Decision:
    p0 = pause.start
    resumes = [s for s in customer_speech if s.start >= pause.end - 1e-6]
    c1 = resumes[0].start if resumes else None
    before = [s for s in customer_speech if s.end <= p0 + 1e-6]
    prev = before[-1] if before else None
    base = dict(pause_start=p0, pause_end=pause.end, onset=c1, confidence=pause.confidence, iou=pause.iou)

    def drop(reason: str, a0=None, ad=None, aw=None) -> Decision:
        return Decision(agent_onset=a0, agent_dur=ad, agent_words=aw, label=None, reason=reason, **base)

    if p0 < greeting_t:
        return drop("technical")
    if technical_nearby(segments, p0):
        return drop("technical")
    if prev is None or prev.end - prev.start < MIN_SPEECH_BEFORE_S:
        return drop("no_speech_before")
    if any(a.start < p0 - 0.05 < a.end for a in agent_utts) or any(a.start <= p0 <= a.end for a in agent_utts):
        return drop("overlap_at_pause")
    # Customer's previous stretch mostly under agent speech -> the customer was the backchanneler.
    overlap = sum(max(0.0, min(prev.end, a.end) - max(prev.start, a.start)) for a in agent_utts)
    if overlap > 0.5 * (prev.end - prev.start):
        return drop("customer_backchannel")
    nxt = [a for a in agent_utts if a.start > p0]
    a = nxt[0] if nxt else None
    if a is not None and c1 is not None and a.start >= c1:
        a = None  # customer resumed first
    if a is None:
        if c1 is None:
            return drop("final_pause")
        if c1 - p0 > MAX_HOLD_S:
            return drop("long_pause")
        return Decision(agent_onset=None, agent_dur=None, agent_words=None, label=0, reason="hold", **base)
    gap = a.start - p0
    dur = a.end - a.start
    words = words_in(segments, "agent", a)
    if gap < MIN_PAUSE_S:
        return drop("too_short", a.start, dur, words)
    turn_like = dur >= BACKCHANNEL_MAX_S or words >= BACKCHANNEL_MAX_WORDS + 1
    if not turn_like:
        if c1 is None:
            return drop("final_pause", a.start, dur, words)
        if c1 - p0 > MAX_HOLD_S:
            return drop("long_pause", a.start, dur, words)
        return Decision(agent_onset=a.start, agent_dur=dur, agent_words=words, label=0, reason="hold_backchannel", **base)
    if c1 is not None and c1 - a.start < EOT_CUSTOMER_SILENT_S:
        return drop("collision", a.start, dur, words)
    return Decision(agent_onset=a.start, agent_dur=dur, agent_words=words, label=1, reason="eot", **base)


def _agent_speech(agent, total: float) -> list[Span]:
    """Agent utterances: complement of the agent channel's energy pauses, merged across short gaps."""
    pauses = [PauseSpan.trusted(s) for s in silence_spans_adaptive(agent, SAMPLE_RATE, min_silence=0.1)]
    utterances = [s for s in speech_intervals(pauses, total) if s.end - s.start >= MIN_AGENT_UTT_S]
    return merge_intervals(utterances, AGENT_MERGE_GAP_S)


def _make_sample(
    sample_id: str, customer, cut: float, pause: PauseSpan, decision: Decision, resumption: float | None, fields: dict
) -> Sample:
    """One causal sample cut at ``cut`` on the customer channel; ``fields`` are the per-conversation/per-pause constants."""
    start = max(0, int((cut - WINDOW_SECONDS) * SAMPLE_RATE))
    tto = max(0.0, resumption - cut) if resumption is not None else None
    observed, reason = observation_window(cut, resumption=resumption, agent_onset=decision.agent_onset, clip_end=fields["clip_duration_s"])
    fv, m = observed_fvad_targets(tto, observed, ended_by_resumption=reason == "customer_resumed")
    return Sample(
        id=sample_id,
        audio=customer[start : int(cut * SAMPLE_RATE)].copy(),
        label=decision.label,
        kind="oracle_eot" if decision.label == 1 else "oracle_hold",
        cut_time=round(cut, 3),
        pause_start=round(pause.start, 3),
        time_to_onset=tto,
        fvad=fv,
        fvad_mask=m,
        cut_confidence=pause.confidence,
        pause_iou=round(pause.iou, 3),
        onset_margin_ms=round(tto * 1000.0, 1) if tto is not None else None,
        future_observed_until=observed,
        observation_end_reason=reason,
        **fields,
    )


def label_conversation(
    conv: Conversation,
    detector: PauseDetector,
    grid: Iterable[float] = (0.2, 0.4, 0.6),
    min_confidence: str = "medium",
    rule_version: str = "v1",
    onset_margin: float = ONSET_MARGIN_S,
    heldout: frozenset[str] = HELDOUT_ACCENTS,
) -> tuple[list[Sample], list[Decision], dict]:
    """Label every customer pause of one conversation; ``heldout`` accents get ``split="heldout"``."""
    if rule_version not in RULES:
        raise ValueError(f"unknown labeling rule version {rule_version!r}; choose from {sorted(RULES)}")
    post_process = RULES[rule_version]
    grid = sorted(set(float(g) for g in grid) | {SCORE_POINT})
    customer = load_wav(conv.customer_path)
    agent = load_wav(conv.agent_path)
    n = min(len(customer), len(agent))
    customer, agent = customer[:n], agent[:n]
    total = n / SAMPLE_RATE
    pauses = detector(customer, SAMPLE_RATE)
    cust_speech = speech_intervals(pauses, total, min_confidence)
    agent_speech = _agent_speech(agent, total)
    greeting_t = first_agent_greeting(conv.segments)
    snr = estimate_snr_db(customer)
    snr = None if snr != snr else round(float(snr), 2)
    digital = round(digital_silence_fraction(customer), 3)
    common = dict(
        clip_id=conv.id,
        source=f"apptek:{conv.accent}",
        snr_est_db=snr,
        detector=detector.detector,
        noise_fill=1,
        clip_duration_s=total,
    )
    common_extra = {
        "accent": conv.accent,
        "speaker_id": conv.customer_speaker,
        "domain": conv.domain,
        "label_method": "dual_channel_heuristic",
        "label_rule_version": rule_version,
        "label_confidence": "heuristic_unreviewed",
        "digital_silence_fraction": digital,
        "split": "heldout" if conv.accent in heldout else "train",
    }
    samples: list[Sample] = []
    decisions: list[Decision] = []
    no_cut: Counter = Counter()  # labeled pauses with no admissible grid point, by reason
    rank = CONFIDENCE_RANK[min_confidence]
    for p in pauses:
        if CONFIDENCE_RANK[p.confidence] < rank:
            decisions.append(Decision(p.start, p.end, None, None, None, None, None, "low_confidence", p.confidence, p.iou))
            continue
        d = post_process(decide(p, cust_speech, agent_speech, conv.segments, greeting_t, total), agent_speech, conv.segments)
        decisions.append(d)
        if d.label is None:
            continue
        prev_text = last_text_before(conv.segments, "customer", p.start)
        agent_text_seg = last_text_before(conv.segments, "agent", p.start, within=60.0)
        fields = {
            **common,
            "agent_text": agent_text_seg.text if agent_text_seg else "",
            "extra": {
                **common_extra,
                "backchannel": int(d.reason == "hold_backchannel"),
                "eot_gap": round(d.agent_onset - p.start, 3) if d.label == 1 else None,
                "agent_onset_s": round(d.agent_onset, 3) if d.agent_onset is not None else None,
                "agent_words": d.agent_words,
                "disfluent_before_pause": int(bool(prev_text and FILLER_RE.search(prev_text.text))),
            },
        }
        # HOLD: the customer's earliest plausible resumption is the decision's onset (energy pause
        # end) capped by the VAD-informed onset of the recorded pause; the cut guard and the stored
        # time_to_onset use the same value (fix D1-ext). EOT: the confident resumption after the
        # agent's turn, if any, only bounds the grid as before.
        resumption = min(d.onset, p.onset, p.end) if d.label == 0 else d.onset
        limit = resumption if resumption is not None else total  # never crop past the speaker's own resumption
        n_before = len(samples)
        for g in grid:
            cut = p.start + g
            if cut + 1e-9 < p.agree_from:
                continue
            if cut > total:
                break
            if d.label == 0:
                cut = hold_cut(cut, limit, margin=onset_margin, tol=1e-9)
                if cut is None:
                    break
            elif cut > limit + 1e-9:
                break
            samples.append(_make_sample(f"{conv.id}_{len(samples):03d}", customer, cut, p, d, resumption, fields))
        if len(samples) == n_before:
            no_cut[d.reason] += 1  # every grid point lay before the VAD agreement or past the resumption
    summary = {
        "conversation": conv.id,
        "accent": conv.accent,
        "duration_s": round(total, 1),
        "n_pauses": len(pauses),
        "reasons": dict(Counter(d.reason for d in decisions)),
        "n_samples": len(samples),
        "labeled_pauses_without_sample": dict(no_cut),
        "digital_silence_fraction": digital,
        "snr_est_db": snr,
        "customer_pauses": [
            {"start": round(p.start, 3), "end": round(p.end, 3), "onset": round(p.onset, 3), "confidence": p.confidence, "iou": p.iou}
            for p in pauses
        ],
        "decisions": [{**{key: (round(v, 3) if isinstance(v, float) else v) for key, v in d.__dict__.items()}} for d in decisions],
    }
    return samples, decisions, summary


def conservative_decision(d: Decision, agent_utts: list[Span], segments: list[Segment]) -> Decision:
    """Rule v2: abstain where v1 inferred backchannel intent from duration/word count alone.

    A brief "yes" can complete an answer, and a brief acknowledgment can precede a longer agent
    turn before the customer returns. Neither warrants a confident HOLD without event annotation
    or independent review. This trades coverage for label caution; it is not a claim of accuracy.
    """
    if d.reason != "hold_backchannel":
        return d
    follows = [a for a in agent_utts if a.start > d.agent_onset and (d.onset is None or a.start < d.onset)]
    has_turn = any(a.end - a.start >= BACKCHANNEL_MAX_S or words_in(segments, "agent", a) > BACKCHANNEL_MAX_WORDS for a in follows)
    reason = "ambiguous_listener_event_chain" if has_turn else "ambiguous_short_listener_event"
    return replace(d, label=None, reason=reason)


# Decision post-processors by rule version. v1 is the pre-registered rule the shipped model was
# trained with; v2 abstains where v1 asserted backchannel intent from duration alone.
RULES: dict[str, Callable[[Decision, list[Span], list[Segment]], Decision]] = {
    "v1": lambda d, _agent_utts, _segments: d,
    "v2": conservative_decision,
}


# ---------------------------------------------------------------------------
# CLI with multiprocessing
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(
    detector_name: str,
    threads: int,
    dirs: dict[str, str],
    heldout: list[str],
    rule_version: str,
    onset_margin: float = ONSET_MARGIN_S,
) -> None:
    _WORKER["detector"] = build_detector(detector_name, MIN_PAUSE_S, threads)
    _WORKER["dirs"] = {k: Path(v) for k, v in dirs.items()}
    _WORKER["heldout"] = frozenset(heldout)
    _WORKER["rule_version"] = rule_version
    _WORKER["onset_margin"] = onset_margin


def _work(payload: tuple[Conversation, list[float], str]):
    """Worker: label one conversation and write its wavs; return manifest rows only."""
    conv, grid, min_conf = payload
    samples, _, summary = label_conversation(
        conv,
        _WORKER["detector"],
        grid,
        min_conf,
        _WORKER["rule_version"],
        _WORKER["onset_margin"],
        _WORKER["heldout"],
    )
    split = "heldout" if conv.accent in _WORKER["heldout"] else "train"
    out_dir: Path = _WORKER["dirs"][split]
    metas = []
    for s in samples:
        path = out_dir / "audio" / f"{s.id}.wav"
        meta = s.meta()
        meta["path"] = path.relative_to(out_dir).as_posix()
        meta["sha256"] = write_wav(path, s.audio, SAMPLE_RATE)
        metas.append(meta)
    return conv, split, metas, summary


def _limit_per_accent(convs: list[Conversation], limit: int | None) -> list[Conversation]:
    """The first ``limit`` conversations of every accent (debug runs); all of them when ``limit`` is falsy."""
    if not limit:
        return convs
    per: Counter = Counter()
    kept = []
    for c in convs:
        if per[c.accent] < limit:
            kept.append(c)
            per[c.accent] += 1
    return kept


def _report(args: argparse.Namespace, heldout: frozenset[str], counts: dict, elapsed_s: float) -> dict:
    return {
        "root": str(args.root),
        "revision_note": "apptek-com/apptek_callcenter_dialogues b98967d9",
        "detector": args.detector,
        "min_confidence": args.min_confidence,
        "grid": args.grid,
        "rule_version": args.rule_version,
        "onset_margin": float(args.onset_margin),
        "heldout_accents": sorted(heldout),
        "n_conversations": counts["n_conversations"],
        "n_pauses": counts["n_pauses"],  # denominator of decision_reasons (every candidate pause has one reason)
        "samples_written": dict(counts["n_written"]),
        "decision_reasons": dict(counts["reasons"]),
        # labeled pauses that produced no sample: every grid point lay before the VAD agreement or
        # past the resumption minus the onset margin (a silent exclusion otherwise)
        "labeled_pauses_without_sample": dict(counts["no_cut"]),
        "per_accent": {k: dict(v) for k, v in sorted(counts["per_accent"].items())},
        "thresholds": {
            "backchannel_max_s": BACKCHANNEL_MAX_S,
            "backchannel_max_words": BACKCHANNEL_MAX_WORDS,
            "eot_customer_silent_s": EOT_CUSTOMER_SILENT_S,
            "max_hold_s": MAX_HOLD_S,
            "min_pause_s": MIN_PAUSE_S,
            "min_speech_before_s": MIN_SPEECH_BEFORE_S,
            "agent_merge_gap_s": AGENT_MERGE_GAP_S,
            "tech_window_s": TECH_WINDOW_S,
        },
        "elapsed_s": round(elapsed_s, 1),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="snapshot root with test/ and diarization/")
    ap.add_argument("--out", type=Path, required=True, help="fresh run directory (train/, heldout/, conversations.jsonl, report.json)")
    ap.add_argument("--accents", nargs="*", default=None, help="default: all accents present")
    ap.add_argument("--heldout", nargs="*", default=sorted(HELDOUT_ACCENTS), help="accents routed to heldout/ (also the split column)")
    ap.add_argument("--grid", type=float, nargs="+", default=[0.2, 0.4, 0.6], help="cut points after the pause start (seconds)")
    ap.add_argument("--detector", choices=("ensemble", "legacy", "energy"), default="ensemble", help="customer pause detector")
    ap.add_argument("--min-confidence", choices=("low", "medium", "high"), default="medium", help="drop pauses below this agreement tier")
    ap.add_argument("--rule-version", choices=sorted(RULES), default="v1", help="v2 abstains on unreviewed short listener events")
    ap.add_argument("--workers", type=int, default=8, help="labeling processes")
    ap.add_argument(
        "--onset-margin",
        type=float,
        default=ONSET_MARGIN_S,
        help="a HOLD cut is never placed past next_onset - margin (seconds; default keeps cuts up to the onset)",
    )
    ap.add_argument("--limit", type=int, default=None, help="conversations per accent (debug)")
    args = ap.parse_args(argv)
    import multiprocessing as mp

    convs = _limit_per_accent(load_conversations(args.root, args.accents), args.limit)
    heldout = frozenset(args.heldout)
    dirs = {"train": args.out / "train", "heldout": args.out / "heldout"}
    if (args.out / "conversations.jsonl").exists() or any((d / "samples.jsonl").exists() for d in dirs.values()):
        raise FileExistsError("labeling output already exists; choose a fresh run directory to preserve prior data")
    for d in dirs.values():
        (d / "audio").mkdir(parents=True, exist_ok=True)
    counts: dict = {
        "n_conversations": len(convs),
        "n_pauses": 0,
        "n_written": Counter(),
        "reasons": Counter(),
        "no_cut": Counter(),
        "per_accent": {},
    }
    t0 = time.time()
    payloads = [(c, args.grid, args.min_confidence) for c in convs]
    init = (args.detector, 1, {k: str(v) for k, v in dirs.items()}, sorted(heldout), args.rule_version, args.onset_margin)
    with contextlib.ExitStack() as stack:
        manifests = {k: stack.enter_context((d / "samples.jsonl").open("w")) for k, d in dirs.items()}
        summaries = stack.enter_context((args.out / "conversations.jsonl").open("w"))
        pool = stack.enter_context(mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=init))
        for i, (conv, split, metas, summary) in enumerate(pool.imap_unordered(_work, payloads), 1):
            for meta in metas:
                manifests[split].write(json.dumps(meta, sort_keys=True, default=str) + "\n")
            counts["n_written"][split] += len(metas)
            counts["reasons"].update(summary["reasons"])
            counts["n_pauses"] += int(summary["n_pauses"])
            counts["no_cut"].update(summary["labeled_pauses_without_sample"])
            counts["per_accent"].setdefault(conv.accent, Counter()).update(summary["reasons"])
            summaries.write(json.dumps(summary, sort_keys=True, default=str) + "\n")
            if i % 20 == 0 or i == len(convs):
                for m in manifests.values():
                    m.flush()
                summaries.flush()
                print(f"{i}/{len(convs)} conversations, samples={dict(counts['n_written'])}, {time.time() - t0:.0f}s", flush=True)
    report = _report(args, heldout, counts, time.time() - t0)
    atomic_write_json(args.out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_accent"}, indent=2))


if __name__ == "__main__":
    main()
