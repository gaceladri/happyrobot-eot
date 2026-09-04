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

Every sample also carries multi-horizon future-speech targets (``fvad``): "will the user speak
again within h seconds?" for h in ``HORIZONS``. For an internal pause cut at time t with speech
resuming at t_on, the target is 1 for horizons >= (t_on - t). For a true EOT it is all zeros.
This is the label-free duration signal of Next-Turn / DualTurn FVAD, derived from timestamps.

CLI
---
    uv run eot-mine --manifest clips.jsonl --out mined/  [--grid 0.2 0.4 0.6] [--min-pause 0.2]

``clips.jsonl`` rows: {"id", "path", "label" (1=EOT complete, 0=incomplete), "source", "agent_text"?}
Output: ``mined/<id>_<k>.wav`` + ``mined/samples.jsonl`` with per-sample metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .audio import SAMPLE_RATE, WINDOW_SECONDS, Span, silence_spans, to_float32, to_mono

HORIZONS: tuple[float, ...] = (0.24, 0.64, 1.2, 2.0)
DEFAULT_SCORE_POINT = 0.2


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
    fvad_mask: int  # 1 if fvad targets are valid
    source: str
    agent_text: str

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("audio")
        return d


def fvad_targets(time_to_onset: float | None, is_eot: bool) -> tuple[list[int], int]:
    if is_eot:
        return [0] * len(HORIZONS), 1
    if time_to_onset is None:
        return [0] * len(HORIZONS), 0
    return [int(time_to_onset <= h) for h in HORIZONS], 1


def _speech_bounds(x: np.ndarray, spans: list[Span], sr: int) -> tuple[float, float]:
    """(first_speech, last_speech_end) in seconds using detected silence spans."""
    total = len(x) / sr
    first = spans[0].end if spans and spans[0].start <= 1e-6 else 0.0
    last = spans[-1].start if spans and abs(spans[-1].end - total) < 0.05 else total
    return first, last


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
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    dither = rng.normal(0.0, max(1e-6, floor * 0.02), size=samples).astype(np.float32)
    return np.clip(repeated + dither, -1.0, 1.0).astype(np.float32)


def mine_clip(
    clip: Clip,
    score_point: float = DEFAULT_SCORE_POINT,
    grid: Iterable[float] = (DEFAULT_SCORE_POINT,),
    min_pause: float = 0.2,
    min_speech_before: float = 0.3,
    tail_silence: float = DEFAULT_SCORE_POINT,
) -> list[Sample]:
    """Produce pause-level samples from one clip. See module docstring."""
    x = clip.audio
    sr = clip.sr
    total = len(x) / sr
    spans = silence_spans(x, sr, min_silence=min_pause)
    first_speech, last_speech_end = _speech_bounds(x, spans, sr)
    grid = sorted(set(float(g) for g in grid) | {float(score_point)})
    out: list[Sample] = []
    k = 0

    # Internal pauses: speech before AND after -> HOLD, with known time-to-onset.
    for sp in spans:
        if sp.start <= first_speech + 1e-6 or sp.end >= last_speech_end - 1e-6:
            continue  # leading / trailing silence
        if sp.start - first_speech < min_speech_before:
            continue
        for g in grid:
            cut = sp.start + g
            boundary_eps = max(1e-9, 0.5 / sr)
            if cut > sp.end + boundary_eps:
                break
            # A pause that survives to the score point is a valid HOLD example even
            # when speech resumes on the immediately following sample/frame.
            cut = min(cut, sp.end)
            tto = sp.end - cut
            fv, m = fvad_targets(tto, is_eot=False)
            out.append(
                Sample(
                    id=f"{clip.id}_{k:02d}", clip_id=clip.id,
                    audio=x[: int(cut * sr)].copy(), label=0, kind="internal",
                    cut_time=cut, pause_start=sp.start, time_to_onset=tto,
                    fvad=fv, fvad_mask=m, source=clip.source, agent_text=clip.agent_text,
                )
            )
            k += 1

    # Final region.
    if clip.label == 1:
        # Complete turn: cut into the trailing pause; pad if the clip has < tail_silence of tail.
        trailing = total - last_speech_end
        for g in grid:
            cut = last_speech_end + g
            seg = x[: int(min(cut, total) * sr)]
            if cut > total:
                missing = int(round((cut - total) * sr))
                seg = np.concatenate([
                    seg,
                    _matched_noise_pad(x, missing, sr, f"{clip.id}:final:{g:.6f}"),
                ])
            fv, m = fvad_targets(None, is_eot=True)
            out.append(
                Sample(
                    id=f"{clip.id}_{k:02d}", clip_id=clip.id, audio=seg, label=1, kind="final",
                    cut_time=cut, pause_start=last_speech_end, time_to_onset=None,
                    fvad=fv, fvad_mask=m, source=clip.source, agent_text=clip.agent_text,
                )
            )
            k += 1
            if trailing < g:  # do not fabricate long silences beyond the first grid point
                break
    else:
        # Incomplete clip ends naturally mid-thought: keep whole, HOLD, censored duration.
        cut = last_speech_end + min(tail_silence, max(0.0, total - last_speech_end))
        seg = x[: int(cut * sr)] if cut < total else x
        if cut - last_speech_end < tail_silence:
            missing = int(round((tail_silence - (cut - last_speech_end)) * sr))
            seg = np.concatenate([
                seg,
                _matched_noise_pad(x, missing, sr, f"{clip.id}:whole:{tail_silence:.6f}"),
            ])
        fv, m = fvad_targets(None, is_eot=False)
        out.append(
            Sample(
                id=f"{clip.id}_{k:02d}", clip_id=clip.id, audio=seg, label=0, kind="whole",
                cut_time=last_speech_end + tail_silence, pause_start=last_speech_end, time_to_onset=None,
                fvad=fv, fvad_mask=m, source=clip.source, agent_text=clip.agent_text,
            )
        )
    return out


def mine(clips: Iterable[Clip], **kw) -> Iterator[Sample]:
    for c in clips:
        yield from mine_clip(c, **kw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_manifest(path: Path) -> Iterator[Clip]:
    import soundfile as sf

    from .audio import resample, to_float32, to_mono
    from .data import read_jsonl_records, resolve_record_path

    path = Path(path)
    for row in read_jsonl_records(path, repair_trailing=False):
        audio_path = resolve_record_path(path, row["path"])
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        audio = resample(to_mono(to_float32(np.asarray(audio))), sr)
        yield Clip(
            id=str(row["id"]), audio=audio, label=int(row["label"]),
            source=str(row.get("source", "unknown")), agent_text=str(row.get("agent_text", "")),
        )


def _atomic_write_wav(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.partial.wav")
    try:
        sf.write(tmp, to_mono(to_float32(np.asarray(audio))), SAMPLE_RATE, subtype="PCM_16")
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_samples(samples: Iterable[Sample], out_dir: Path, *, resume: bool = False) -> dict:
    """Persist mined samples safely; ``resume`` skips complete records and repairs torn tails."""
    from .data import (
        append_jsonl_record,
        atomic_write_jsonl,
        read_jsonl_records,
        resolve_record_path,
        safe_audio_filename,
        sha256_file,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
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

    stats = {
        "internal": sum(row.get("kind") == "internal" for row in existing.values()),
        "final": sum(row.get("kind") == "final" for row in existing.values()),
        "whole": sum(row.get("kind") == "whole" for row in existing.values()),
        "label_1": sum(int(row["label"]) == 1 for row in existing.values()),
        "label_0": sum(int(row["label"]) == 0 for row in existing.values()),
        "written": 0,
        "skipped": 0,
    }
    max_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    for s in samples:
        sample_id = str(s.id)
        prior = existing.get(sample_id)
        if prior is not None:
            prior_path = resolve_record_path(manifest, prior["path"])
            if not prior_path.exists():
                _atomic_write_wav(prior_path, s.audio[-max_samples:])
                prior["sha256"] = sha256_file(prior_path)
                prior["stored_duration_s"] = round(min(len(s.audio), max_samples) / SAMPLE_RATE, 6)
                atomic_write_jsonl(manifest, existing.values())
            elif prior.get("sha256") and sha256_file(prior_path) != prior["sha256"]:
                raise ValueError(f"mined audio failed checksum validation: {prior_path}")
            elif not prior.get("sha256"):
                prior["sha256"] = sha256_file(prior_path)
                atomic_write_jsonl(manifest, existing.values())
            stats["skipped"] += 1
            continue
        wav = audio_dir / safe_audio_filename(sample_id)
        stored_audio = s.audio[-max_samples:]
        _atomic_write_wav(wav, stored_audio)
        meta = s.meta()
        meta.update({
            "path": wav.relative_to(out_dir).as_posix(),
            "stored_duration_s": round(len(stored_audio) / SAMPLE_RATE, 6),
            "sha256": sha256_file(wav),
        })
        append_jsonl_record(manifest, meta)
        existing[sample_id] = meta
        stats[s.kind] += 1
        stats[f"label_{s.label}"] += 1
        stats["written"] += 1
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    ap.add_argument("--min-pause", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted compatible run")
    args = ap.parse_args(argv)
    from .data import atomic_write_json, sha256_file

    metadata_path = args.out / "metadata.json"
    config = {
        "source_manifest": os.path.relpath(args.manifest.resolve(), start=args.out.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest),
        "grid": sorted(set(float(value) for value in args.grid) | {DEFAULT_SCORE_POINT}),
        "min_pause": float(args.min_pause),
        "window_seconds": WINDOW_SECONDS,
    }
    if args.resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        mismatches = {key: (previous.get(key), value) for key, value in config.items() if previous.get(key) != value}
        if mismatches:
            raise ValueError(f"cannot resume mining with changed inputs: {mismatches}")
    atomic_write_json(metadata_path, {**config, "stats": None, "completed": False})
    stats = write_samples(
        mine(_read_manifest(args.manifest), grid=args.grid, min_pause=args.min_pause),
        args.out,
        resume=args.resume,
    )
    atomic_write_json(metadata_path, {**config, "stats": stats, "completed": True})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
