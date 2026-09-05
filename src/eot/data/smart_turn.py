"""Smart Turn v3.2 corpus: streaming loader and resumable, relocation-safe acquisition.

Columns used: ``audio``, ``endpoint_bool``, ``language`` and ``dataset`` (the contributing source,
used for grouped splits because the corpus has no speaker ids). ``acquire_clips`` streams a pinned
revision, writes each WAV atomically, fsyncs one manifest record per clip and can resume until
both class quotas are met.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from eot.audio import decode_payload
from eot.io import (
    append_jsonl_record, atomic_write_json, atomic_write_jsonl, atomic_write_wav, read_jsonl_records, resolve_record_path,
    safe_audio_filename, sha256_file,
)
from eot.labeling.samples import Clip


SMART_TURN_TRAIN = "pipecat-ai/smart-turn-data-v3.2-train"
SMART_TURN_TEST = "pipecat-ai/smart-turn-data-v3.2-test"


def load_smart_turn_clips(
    name: str = SMART_TURN_TRAIN,
    language: str = "eng",
    per_label: int = 8_000,
    seed: int = 0,
    streaming: bool = True,
    revision: str | None = None,
    max_rows: int | None = None,
) -> Iterator[Clip]:
    """Yield up to ``per_label`` clips of each label for ``language`` from the Smart Turn corpus."""
    from datasets import Audio, load_dataset

    ds = load_dataset(name, split="train", streaming=streaming, revision=revision)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=seed, buffer_size=2_000) if streaming else ds.shuffle(seed=seed)
    taken = {0: 0, 1: 0}
    for i, row in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        if str(row.get("language", "")) != language:
            continue
        label = int(bool(row["endpoint_bool"]))
        if taken[label] >= per_label:
            if all(v >= per_label for v in taken.values()):
                break
            continue
        x = decode_payload(row["audio"])
        taken[label] += 1
        yield Clip(
            id=str(row.get("id", i)), audio=x, label=label,
            source=str(row.get("dataset", "unknown")), agent_text=str(row.get("agent_text", "") or ""),
        )


def _validate_resume_metadata(path: Path, expected: dict) -> None:
    if not path.exists():
        return
    current = json.loads(path.read_text())
    mismatches = {
        key: (current.get(key), value)
        for key, value in expected.items()
        if current.get(key) != value
    }
    if mismatches:
        details = ", ".join(f"{key}={old!r} (requested {new!r})" for key, (old, new) in mismatches.items())
        raise ValueError(f"cannot resume acquisition with changed provenance: {details}")


def acquire_clips(
    clips: Iterable[Clip],
    out_dir: Path,
    *,
    dataset: str,
    revision: str,
    language: str,
    per_label: int,
    seed: int,
    max_rows: int | None = None,
    resume: bool = False,
    progress_every: int = 100,
) -> dict:
    """Persist an iterable of already-filtered clips and return reproducibility metadata."""
    if per_label <= 0:
        raise ValueError("per_label must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when set")
    if not revision:
        raise ValueError("an immutable dataset revision is required")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    manifest = out_dir / "clips.jsonl"
    metadata_path = out_dir / "metadata.json"

    if manifest.exists() and manifest.stat().st_size and not resume:
        raise FileExistsError(f"{manifest} already exists; pass --resume to continue")

    provenance = {
        "dataset": dataset,
        "revision": revision,
        "language": language,
        "seed": int(seed),
    }
    if resume:
        _validate_resume_metadata(metadata_path, provenance)
    existing_rows = read_jsonl_records(manifest, repair_trailing=resume)
    if not resume and existing_rows:
        raise FileExistsError(f"{manifest} already contains records")

    # Deduplicate a pre-existing manifest deterministically before appending more data.
    existing: dict[str, dict] = {}
    for row in existing_rows:
        clip_id = str(row["id"])
        if clip_id not in existing:
            existing[clip_id] = row
    if len(existing) != len(existing_rows):
        if not resume:
            raise ValueError("manifest contains duplicate clip ids")
        atomic_write_jsonl(manifest, existing.values())

    if resume and existing and not metadata_path.exists():
        for key, expected in provenance.items():
            found = {row.get(key) for row in existing.values()}
            if found != {expected}:
                raise ValueError(
                    f"cannot prove resume provenance for {key}: found {sorted(map(str, found))}, "
                    f"requested {expected!r}"
                )

    # Validate the whole completed prefix before trusting quota counts. Without
    # this pass a fully-counted manifest could return before noticing a deleted
    # or corrupted WAV. Missing files are removed from the manifest so the
    # deterministic stream can reconstruct them; checksum mismatches fail hard.
    manifest_changed = False
    for clip_id, row in list(existing.items()):
        audio_path = resolve_record_path(manifest, row["path"])
        if not audio_path.exists():
            del existing[clip_id]
            manifest_changed = True
            continue
        digest = sha256_file(audio_path)
        if row.get("sha256") and digest != row["sha256"]:
            raise ValueError(f"acquired audio failed checksum validation: {audio_path}")
        if not row.get("sha256"):
            row["sha256"] = digest
            manifest_changed = True
    if manifest_changed:
        atomic_write_jsonl(manifest, existing.values())

    counts = Counter(int(row["label"]) for row in existing.values())
    if set(counts) - {0, 1}:
        raise ValueError(f"manifest contains unsupported labels: {sorted(counts)}")
    clips_seen = 0
    clips_added = 0

    def metadata(completed: bool) -> dict:
        return {
            **provenance,
            "per_label": int(per_label),
            "max_rows": max_rows,
            "clips_seen_this_run": clips_seen,
            "clips_added_this_run": clips_added,
            "counts": {"0": counts[0], "1": counts[1]},
            "manifest": manifest.name,
            "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
            "completed": completed,
        }

    # Publish provenance before the first potentially expensive decode/write. If interrupted,
    # the next --resume invocation can reject a different dataset/revision/language/seed.
    atomic_write_json(metadata_path, metadata(completed=False))

    for clip in clips:
        if counts[0] >= per_label and counts[1] >= per_label:
            break
        clips_seen += 1
        label = int(clip.label)
        if label not in (0, 1):
            raise ValueError(f"clip {clip.id!r} has invalid label {label}")
        clip_id = str(clip.id)
        prior = existing.get(clip_id)
        if prior is not None:
            if int(prior["label"]) != label:
                raise ValueError(
                    f"clip id {clip_id!r} changed label from {prior['label']} to {label}"
                )
            prior_path = resolve_record_path(manifest, prior["path"])
            if not prior_path.exists():
                prior["sha256"] = atomic_write_wav(prior_path, clip.audio, clip.sr)
                atomic_write_jsonl(manifest, existing.values())
            elif prior.get("sha256") and sha256_file(prior_path) != prior["sha256"]:
                raise ValueError(f"acquired audio failed checksum validation: {prior_path}")
            continue
        if counts[label] >= per_label:
            continue

        wav = audio_dir / safe_audio_filename(clip_id)
        row = {
            "id": clip_id,
            "path": wav.relative_to(out_dir).as_posix(),
            "label": label,
            "source": str(clip.source),
            "agent_text": str(clip.agent_text or ""),
            **provenance,
            "duration_s": round(len(clip.audio) / float(clip.sr), 6),
            "sha256": atomic_write_wav(wav, clip.audio, clip.sr),
        }
        append_jsonl_record(manifest, row)
        existing[clip_id] = row
        counts[label] += 1
        clips_added += 1
        if progress_every > 0 and clips_added % progress_every == 0:
            progress = metadata(completed=False)
            atomic_write_json(metadata_path, progress)
            print(json.dumps({"progress": progress["counts"], "clips_added": clips_added}))

    completed = counts[0] >= per_label and counts[1] >= per_label
    result = metadata(completed=completed)
    atomic_write_json(metadata_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=SMART_TURN_TRAIN)
    parser.add_argument("--revision", required=True, help="immutable Hugging Face dataset revision")
    parser.add_argument("--language", default="eng", help="exact language value to retain")
    parser.add_argument("--per-label", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None, help="maximum source rows to scan")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    clips = load_smart_turn_clips(
        name=args.dataset,
        language=args.language,
        per_label=args.per_label,
        seed=args.seed,
        streaming=True,
        revision=args.revision,
        max_rows=args.max_rows,
    )
    result = acquire_clips(
        clips,
        args.out,
        dataset=args.dataset,
        revision=args.revision,
        language=args.language,
        per_label=args.per_label,
        seed=args.seed,
        max_rows=args.max_rows,
        resume=args.resume,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["completed"]:
        raise RuntimeError(
            "dataset stream ended before both class quotas were met; resume with a larger --max-rows"
        )


if __name__ == "__main__":
    main()
