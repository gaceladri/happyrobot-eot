"""Prefix mining: turn clip-level EOT/HOLD data into pause-level causal training samples.

Why
---
Smart Turn v3.2 reaches ~94 % clip accuracy but 35 % false cutoffs @300 ms on LiveKit EoT Bench.
The gap is train/eval mismatch: training asks "is this clip complete?", deployment asks
"at *this* 200 ms pause, has the user yielded the floor?". EoT Bench scores every silence
>= 100 ms inside a real turn; only the last one is EOT, all earlier ones are HOLD.

Prefix mining rebuilds that decision distribution from ordinary clips, with no new data:

- every internal pause of >= ``min_pause`` seconds followed by more speech yields HOLD samples,
  cut at ``score_point`` (and optionally a grid of later points) into the pause;
- the clip's own trailing pause yields the clip label (EOT for complete clips);
- incomplete clips that end mid-thought are kept whole as HOLD (label 0), but their
  "time to next onset" is unknown (censored) so the auxiliary targets are masked.

Every sample records its observation window (``future_observed_until`` / ``observation_end_reason``
/ ``clip_duration_s``, see :mod:`eot.labeling.samples`): internal cuts were observed until the
speaker resumed, clip-end cuts until the clip ended. A HOLD cut is never placed past
``next_onset - onset_margin`` (``--onset-margin``, default 0): the grid stops at the first
inadmissible point instead of clamping past the onset, so ``time_to_onset`` is never negative.

Every sample also carries multi-horizon future-speech targets (``fvad``): "will the user speak
again within h seconds?" for h in ``HORIZONS``. For an internal pause cut at time t with speech
resuming at t_on, the target is 1 for horizons >= (t_on - t). For a true EOT it is all zeros.
This is the label-free duration signal of Next-Turn / DualTurn FVAD, derived from timestamps.

Pause detection (rev. 3)
------------------------
By default pauses come from :class:`eot.audio.PauseDetector` in ``ensemble`` mode: a
noise-floor-adaptive energy detector and Silero VAD v6.2.1 must agree; the cut is placed inside
their intersection; every sample records ``cut_confidence`` (high/medium/low), ``pause_iou``,
``snr_est_db`` and ``onset_margin_ms``. Pauses only one detector sees ("low") are counted and
dropped from training by default. ``--detector legacy`` reproduces the peak-relative energy
detector of rev. 2 so its cost can be measured (run R2').

CLI
---
    uv run eot-mine --manifest clips.jsonl --out mined/  [--grid 0.2 0.4 0.6] [--min-pause 0.2]
                    [--detector ensemble|legacy|energy] [--min-confidence medium]

``clips.jsonl`` rows: {"id", "path", "label" (1=EOT complete, 0=incomplete), "source", "agent_text"?}
Output: ``mined/<id>_<k>.wav`` + ``mined/samples.jsonl`` with per-sample metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from eot.audio import (
    SAMPLE_RATE,
    SCORE_POINT,
    SILERO_VAD_SHA256,
    SILERO_VAD_VERSION,
    WINDOW_SECONDS,
    PauseDetector,
    PauseSpan,
    SileroVAD,
    Span,
    digital_silence_fraction,
    estimate_snr_db,
    load_wav,
)
from eot.io import (
    append_jsonl_record,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_wav,
    read_jsonl_records,
    read_samples,
    resolve_record_path,
    safe_audio_filename,
    seed_from_key,
    sha256_file,
    write_wav,
)
from eot.labeling.samples import (
    CONFIDENCE_RANK,
    ONSET_MARGIN_S,
    Clip,
    Sample,
    hold_cut,
    observation_window,
    observed_fvad_targets,
)

MAX_STORED_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)  # only the model window is written to disk
# Clips with more digital silence than this were gated (VoIP/AEC): training adds room tone (``noise_fill``).
DIGITAL_SILENCE_FILL_FRACTION = 0.2


def _speech_bounds(x: np.ndarray, spans: list[Span] | list[PauseSpan], sr: int) -> tuple[float, float]:
    """(first_speech, last_speech_end) in seconds using detected silence spans."""
    total = len(x) / sr
    first = spans[0].end if spans and spans[0].start <= 1e-6 else 0.0
    last = spans[-1].start if spans and abs(spans[-1].end - total) < 0.05 else total
    return first, last


def _trailing_pause(spans: list[PauseSpan], total: float) -> PauseSpan | None:
    if spans and abs(spans[-1].end - total) < 0.05:
        return spans[-1]
    return None


def _matched_noise_pad(x: np.ndarray, samples: int, sr: int, key: str) -> np.ndarray:
    """Extend a pause with deterministic clip-matched room tone, never digital zero.

    The quietest 20 ms frame is a useful local estimate of the recording floor. Repeating it
    preserves its spectrum; low-level deterministic dither avoids a periodic/exact-zero cue.
    """
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    frame = max(1, int(round(0.02 * sr)))
    usable = len(x) // frame * frame
    if usable:
        frames = np.asarray(x[:usable], dtype=np.float32).reshape(-1, frame)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        quiet = frames[int(np.argmin(rms))]
        floor = float(rms.min())
    else:
        quiet = np.zeros(frame, dtype=np.float32)
        floor = 0.0
    repeated = np.resize(quiet, samples).astype(np.float32, copy=False)
    rng = np.random.default_rng(seed_from_key(key))
    dither = rng.normal(0.0, max(1e-6, floor * 0.02), size=samples).astype(np.float32)
    return np.clip(repeated + dither, -1.0, 1.0).astype(np.float32)


def _sample_fields(
    kind: str,
    label: int,
    audio: np.ndarray,
    cut: float,
    pause_start: float,
    tto: float | None,
    observed: float,
    reason: str,
    confidence: str,
    iou: float,
) -> dict:
    """``Sample`` fields that differ per cut; the future-speech targets follow from the observation window."""
    fv, m = observed_fvad_targets(tto, observed, ended_by_resumption=reason == "customer_resumed")
    return dict(
        audio=audio,
        label=label,
        kind=kind,
        cut_time=cut,
        pause_start=pause_start,
        time_to_onset=tto,
        fvad=fv,
        fvad_mask=m,
        cut_confidence=confidence,
        pause_iou=iou,
        onset_margin_ms=round(tto * 1000.0, 1) if tto is not None else None,
        future_observed_until=observed,
        observation_end_reason=reason,
    )


def _internal_samples(
    x: np.ndarray,
    sr: int,
    pauses: list[PauseSpan],
    first_speech: float,
    last_speech_end: float,
    grid: list[float],
    min_speech_before: float,
    min_rank: int,
    onset_margin: float,
    stats: Counter,
) -> list[dict]:
    """Internal pauses (speech before AND after) -> HOLD cuts with a known time-to-onset."""
    total = len(x) / sr
    out = []
    for sp in pauses:
        if sp.start <= first_speech + 1e-6 or sp.end >= last_speech_end - 1e-6:
            continue  # leading / trailing silence
        if sp.start - first_speech < min_speech_before:
            continue
        if CONFIDENCE_RANK[sp.confidence] < min_rank:
            stats[f"dropped_internal_{sp.confidence}"] += 1
            continue
        onset = min(sp.end, sp.onset)  # earliest plausible resumption on the speaker's own channel
        for g in grid:
            cut = sp.start + g
            if cut + 1e-9 < sp.agree_from:
                stats["dropped_cut_before_vad_silence"] += 1
                continue
            # A pause that survives to the score point is a valid HOLD example even when speech
            # resumes on the immediately following sample; a grid point past onset - margin ends
            # the grid for this pause (D1 guard: the cut is never moved past the onset).
            cut = hold_cut(cut, onset, margin=onset_margin, tol=max(1e-9, 0.5 / sr))
            if cut is None:
                stats["dropped_cut_past_onset_margin"] += 1
                break
            tto = max(0.0, onset - cut)
            observed, reason = observation_window(cut, resumption=onset, clip_end=total)
            out.append(
                _sample_fields(
                    "internal", 0, x[: int(cut * sr)].copy(), cut, sp.start, tto, observed, reason, sp.confidence, round(sp.iou, 3)
                )
            )
    return out


def _final_samples(clip: Clip, x: np.ndarray, sr: int, last_speech_end: float, grid: list[float], tail: dict) -> list[dict]:
    """Complete turn: EOT cuts into the trailing pause, padded with matched noise when the clip is shorter than the cut."""
    total = len(x) / sr
    trailing = total - last_speech_end
    out = []
    for g in grid:
        cut = last_speech_end + g
        seg = x[: int(min(cut, total) * sr)]
        if cut > total:
            seg = np.concatenate([seg, _matched_noise_pad(x, int(round((cut - total) * sr)), sr, f"{clip.id}:final:{g:.6f}")])
        # The tail after the cut was observed silent until the clip ended (0 s when the cut is
        # beyond the clip end and the tail is matched noise; the row is kept and flagged).
        observed, reason = observation_window(cut, clip_end=total)
        out.append(_sample_fields("final", 1, seg, cut, last_speech_end, None, observed, reason, **tail))
        if trailing < g:  # do not fabricate long silences beyond the first grid point
            break
    return out


def _whole_sample(clip: Clip, x: np.ndarray, sr: int, last_speech_end: float, tail_silence: float, tail: dict) -> dict:
    """Incomplete clip that ends mid-thought: kept whole as HOLD with a censored time-to-onset."""
    total = len(x) / sr
    cut = last_speech_end + min(tail_silence, max(0.0, total - last_speech_end))
    seg = x[: int(cut * sr)] if cut < total else x
    if cut - last_speech_end < tail_silence:
        missing = int(round((tail_silence - (cut - last_speech_end)) * sr))
        seg = np.concatenate([seg, _matched_noise_pad(x, missing, sr, f"{clip.id}:whole:{tail_silence:.6f}")])
    cut_time = last_speech_end + tail_silence
    observed, reason = observation_window(cut_time, clip_end=total)
    return _sample_fields("whole", 0, seg, cut_time, last_speech_end, None, observed, reason, **tail)


def mine_clip(
    clip: Clip,
    score_point: float = SCORE_POINT,
    grid: Iterable[float] = (SCORE_POINT,),
    min_pause: float = 0.2,
    min_speech_before: float = 0.3,
    tail_silence: float = SCORE_POINT,
    detector: PauseDetector | None = None,
    min_confidence: str = "medium",
    stats: Counter | None = None,
    onset_margin: float = ONSET_MARGIN_S,
) -> list[Sample]:
    """Produce pause-level samples from one clip. See module docstring.

    ``detector`` defaults to the legacy peak-relative energy detector for backwards compatibility
    of callers; the CLI defaults to the ensemble. ``stats`` (optional Counter) receives counts of
    pauses dropped for low confidence so the mining report can show them.
    """
    stats = Counter() if stats is None else stats
    x, sr = clip.audio, clip.sr
    total = len(x) / sr
    detector = detector or PauseDetector("legacy", min_silence=min_pause)
    pauses = detector(x, sr)
    first_speech, last_speech_end = _speech_bounds(x, pauses, sr)
    grid = sorted(set(float(g) for g in grid) | {float(score_point)})
    snr = estimate_snr_db(x, sr)
    common = dict(
        clip_id=clip.id,
        source=clip.source,
        agent_text=clip.agent_text,
        snr_est_db=None if snr != snr else round(float(snr), 2),  # NaN -> None
        detector=detector.detector,
        noise_fill=int(digital_silence_fraction(x, sr) > DIGITAL_SILENCE_FILL_FRACTION),
        clip_duration_s=total,
    )
    fields = _internal_samples(
        x, sr, pauses, first_speech, last_speech_end, grid, min_speech_before, CONFIDENCE_RANK[min_confidence], onset_margin, stats
    )
    trailing_pause = _trailing_pause(pauses, total)
    tail = (
        dict(confidence=trailing_pause.confidence, iou=round(trailing_pause.iou, 3))
        if trailing_pause
        else dict(confidence="medium", iou=0.0)
    )
    if clip.label == 1:
        fields += _final_samples(clip, x, sr, last_speech_end, grid, tail)
    else:
        fields.append(_whole_sample(clip, x, sr, last_speech_end, tail_silence, tail))
    return [Sample(id=f"{clip.id}_{k:02d}", **f, **common) for k, f in enumerate(fields)]


def mine(clips: Iterable[Clip], **kw) -> Iterator[Sample]:
    for c in clips:
        yield from mine_clip(c, **kw)


def clips_from_manifest(path: Path) -> Iterator[Clip]:
    """Stream ``Clip`` objects from a ``clips.jsonl`` manifest (paths resolved relative to it)."""
    for row in read_samples(path):
        audio = load_wav(row["path"])
        yield Clip(
            id=str(row["id"]),
            audio=audio,
            label=int(row["label"]),
            source=str(row.get("source", "unknown")),
            agent_text=str(row.get("agent_text", "")),
        )


def _persist(sample: Sample, out_dir: Path, write) -> dict:
    """Write the last ``WINDOW_SECONDS`` of a sample under ``out_dir/audio`` with ``write`` and return its manifest row."""
    wav = out_dir / "audio" / safe_audio_filename(str(sample.id))
    stored = sample.audio[-MAX_STORED_SAMPLES:]
    meta = sample.meta()
    meta.update(
        {
            "path": wav.relative_to(out_dir).as_posix(),
            "stored_duration_s": round(len(stored) / SAMPLE_RATE, 6),
            "sha256": write(wav, stored, SAMPLE_RATE),
        }
    )
    return meta


def _count(stats: dict, meta: dict) -> None:
    """Add one manifest row to the kind / label / confidence tallies."""
    stats[meta["kind"]] = stats.get(meta["kind"], 0) + 1
    stats[f"label_{int(meta['label'])}"] += 1
    conf = str(meta.get("cut_confidence", "high"))
    stats["confidence"][conf] = stats["confidence"].get(conf, 0) + 1


def _initial_stats(existing: Iterable[dict] = ()) -> dict:
    """Mining statistics seeded with the rows a resumed run already holds."""
    stats = {"internal": 0, "final": 0, "whole": 0, "label_1": 0, "label_0": 0, "written": 0, "skipped": 0, "confidence": {}}
    for row in existing:
        _count(stats, row)
    return stats


def write_samples(samples: Iterable[Sample], out_dir: Path, *, resume: bool = False) -> dict:
    """Persist mined samples safely; ``resume`` skips complete records and repairs torn tails."""
    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "samples.jsonl"
    if resume:
        existing_rows = read_jsonl_records(manifest, repair_trailing=True)
    else:
        existing_rows = []
        atomic_write_jsonl(manifest, [])
    existing = {str(row["id"]): row for row in existing_rows}
    if len(existing) != len(existing_rows):
        if not resume:
            raise ValueError("new samples unexpectedly contain duplicate ids")
        atomic_write_jsonl(manifest, existing.values())
    stats = _initial_stats(existing.values())
    for s in samples:
        sample_id = str(s.id)
        prior = existing.get(sample_id)
        if prior is not None:
            prior_path = resolve_record_path(manifest, prior["path"])
            if not prior_path.exists():
                prior["sha256"] = atomic_write_wav(prior_path, s.audio[-MAX_STORED_SAMPLES:], SAMPLE_RATE)
                prior["stored_duration_s"] = round(min(len(s.audio), MAX_STORED_SAMPLES) / SAMPLE_RATE, 6)
                atomic_write_jsonl(manifest, existing.values())
            elif prior.get("sha256") and sha256_file(prior_path) != prior["sha256"]:
                raise ValueError(f"mined audio failed checksum validation: {prior_path}")
            elif not prior.get("sha256"):
                prior["sha256"] = sha256_file(prior_path)
                atomic_write_jsonl(manifest, existing.values())
            stats["skipped"] += 1
            continue
        meta = _persist(s, out_dir, atomic_write_wav)
        append_jsonl_record(manifest, meta)
        existing[sample_id] = meta
        _count(stats, meta)
        stats["written"] += 1
    return stats


def build_detector(name: str, min_pause: float, vad_threads: int = 1) -> PauseDetector:
    vad = SileroVAD(threads=vad_threads) if name == "ensemble" else None
    return PauseDetector(name, vad=vad, min_silence=min_pause)


_WORKER: dict = {}


def _init_mine_worker(
    detector: str,
    min_pause: float,
    grid: list[float],
    min_confidence: str,
    out_dir: str,
    onset_margin: float = ONSET_MARGIN_S,
) -> None:
    _WORKER["detector"] = build_detector(detector, min_pause, 1)
    _WORKER.update(
        grid=grid,
        min_pause=min_pause,
        min_confidence=min_confidence,
        out_dir=Path(out_dir),
        onset_margin=onset_margin,
    )


def _mine_one(row: dict) -> tuple[list[dict], dict]:
    """Worker: mine one clip, write its wavs, return manifest rows + drop counts."""
    audio = load_wav(row["_audio_path"])
    clip = Clip(
        id=str(row["id"]),
        audio=audio,
        label=int(row["label"]),
        source=str(row.get("source", "unknown")),
        agent_text=str(row.get("agent_text", "")),
    )
    dropped: Counter = Counter()
    samples = mine_clip(
        clip,
        grid=_WORKER["grid"],
        min_pause=_WORKER["min_pause"],
        detector=_WORKER["detector"],
        min_confidence=_WORKER["min_confidence"],
        stats=dropped,
        onset_margin=_WORKER["onset_margin"],
    )
    # Plain write: the parallel path has no resume.
    return [_persist(s, _WORKER["out_dir"], write_wav) for s in samples], dict(dropped)


def write_samples_parallel(
    manifest_in: Path,
    out_dir: Path,
    *,
    detector: str,
    min_pause: float,
    grid: list[float],
    min_confidence: str,
    workers: int,
    onset_margin: float = ONSET_MARGIN_S,
) -> dict:
    """Multiprocess mining (no resume). Workers write wavs; the parent appends manifest rows."""
    import multiprocessing as mp

    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "samples.jsonl"
    rows = read_jsonl_records(manifest_in)
    for row in rows:
        row["_audio_path"] = str(resolve_record_path(manifest_in, row["path"]))
    stats = {**_initial_stats(), "dropped": {}}
    seen: set[str] = set()
    with (
        manifest.open("w") as f,
        mp.get_context("spawn").Pool(
            workers,
            initializer=_init_mine_worker,
            initargs=(detector, min_pause, list(grid), min_confidence, str(out_dir), onset_margin),
        ) as pool,
    ):
        for i, (metas, dropped) in enumerate(pool.imap_unordered(_mine_one, rows, chunksize=4), 1):
            for meta in metas:
                if meta["id"] in seen:
                    raise ValueError(f"duplicate sample id {meta['id']}")
                seen.add(meta["id"])
                f.write(json.dumps(meta, sort_keys=True, default=str) + "\n")
                _count(stats, meta)
                stats["written"] += 1
            for key, value in dropped.items():
                stats["dropped"][key] = stats["dropped"].get(key, 0) + value
            if i % 500 == 0 or i == len(rows):
                f.flush()
                print(f"mined {i}/{len(rows)} clips -> {stats['written']} samples", flush=True)
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    ap.add_argument("--min-pause", type=float, default=0.2)
    ap.add_argument("--detector", choices=("ensemble", "legacy", "energy"), default="ensemble")
    ap.add_argument("--min-confidence", choices=("low", "medium", "high"), default="medium")
    ap.add_argument("--vad-threads", type=int, default=1)
    ap.add_argument(
        "--onset-margin",
        type=float,
        default=ONSET_MARGIN_S,
        help="a HOLD cut is never placed past next_onset - margin (seconds; default keeps cuts up to the onset)",
    )
    ap.add_argument("--workers", type=int, default=1, help=">1 enables multiprocess mining (no --resume)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted compatible run")
    args = ap.parse_args(argv)
    metadata_path = args.out / "metadata.json"
    config = {
        "source_manifest": os.path.relpath(args.manifest.resolve(), start=args.out.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest),
        "grid": sorted(set(float(value) for value in args.grid) | {SCORE_POINT}),
        "min_pause": float(args.min_pause),
        "window_seconds": WINDOW_SECONDS,
        "detector": args.detector,
        "min_confidence": args.min_confidence,
        "onset_margin": float(args.onset_margin),
        "silero_vad": {"version": SILERO_VAD_VERSION, "sha256": SILERO_VAD_SHA256} if args.detector == "ensemble" else None,
    }
    if args.resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        mismatches = {key: (previous.get(key), value) for key, value in config.items() if previous.get(key) != value}
        if mismatches:
            raise ValueError(f"cannot resume mining with changed inputs: {mismatches}")
    atomic_write_json(metadata_path, {**config, "stats": None, "completed": False})
    if args.workers > 1:
        if args.resume:
            raise ValueError("--resume is only supported with --workers 1")
        stats = write_samples_parallel(
            args.manifest,
            args.out,
            detector=args.detector,
            min_pause=args.min_pause,
            grid=args.grid,
            min_confidence=args.min_confidence,
            workers=args.workers,
            onset_margin=args.onset_margin,
        )
    else:
        detector = build_detector(args.detector, args.min_pause, args.vad_threads)
        dropped: Counter = Counter()
        stats = write_samples(
            mine(
                clips_from_manifest(args.manifest),
                grid=args.grid,
                min_pause=args.min_pause,
                detector=detector,
                min_confidence=args.min_confidence,
                stats=dropped,
                onset_margin=args.onset_margin,
            ),
            args.out,
            resume=args.resume,
        )
        stats["dropped"] = dict(dropped)
    atomic_write_json(metadata_path, {**config, "stats": stats, "completed": True})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
