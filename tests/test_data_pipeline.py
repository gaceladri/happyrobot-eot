from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eot.acquire import acquire_clips
from eot.audio import SAMPLE_RATE, WINDOW_SECONDS, silence_spans
from eot.data import MinedDataset, grouped_split, read_jsonl_records, read_samples
from eot.prefix_mining import Clip, Sample, _read_manifest, mine_clip, write_samples


def _rows(source: str, label: int, count: int) -> list[dict]:
    return [
        {"source": source, "clip_id": f"{source}-{index}", "label": label, "path": "unused.wav"}
        for index in range(count)
    ]


def test_grouped_split_avoids_dominant_source_and_keeps_two_classes() -> None:
    rows = _rows("zero", 0, 10) + _rows("one", 1, 10)
    rows += [
        {"source": "dominant", "clip_id": f"dominant-{index}", "label": index % 2, "path": "unused.wav"}
        for index in range(80)
    ]
    train, dev, diagnostics = grouped_split(rows, dev_frac=0.2, seed=7, return_diagnostics=True)
    assert {row["source"] for row in dev} == {"zero", "one"}
    assert {int(row["label"]) for row in train} == {0, 1}
    assert {int(row["label"]) for row in dev} == {0, 1}
    assert diagnostics["group_overlap"] == []
    assert len(dev) == 20


def test_grouped_split_falls_back_to_clip_without_prefix_leakage() -> None:
    rows = []
    for label in (0, 1):
        for clip_index in range(3):
            clip_id = f"clip-{label}-{clip_index}"
            rows.extend(
                {"source": "only-source", "clip_id": clip_id, "label": label, "path": "unused.wav"}
                for _ in range(2)
            )
    train, dev, diagnostics = grouped_split(rows, dev_frac=0.34, seed=3, return_diagnostics=True)
    assert diagnostics["group_key"] == "clip_id"
    assert {row["clip_id"] for row in train}.isdisjoint({row["clip_id"] for row in dev})
    assert {int(row["label"]) for row in train} == {0, 1}
    assert {int(row["label"]) for row in dev} == {0, 1}


def test_grouped_split_uses_clip_guardrail_when_sources_are_too_large() -> None:
    rows = []
    for source, count in (("source-a", 8), ("source-b", 6)):
        rows.extend(
            {
                "source": source,
                "clip_id": f"{source}-{index}",
                "label": index % 2,
                "path": "unused.wav",
            }
            for index in range(count)
        )
    train, dev, diagnostics = grouped_split(rows, dev_frac=0.15, seed=2, return_diagnostics=True)
    assert diagnostics["group_key"] == "clip_id"
    assert diagnostics["requested_group_key"] == "source"
    assert "fallback_reason" in diagnostics
    assert len(dev) / len(rows) <= 0.30
    assert {int(row["label"]) for row in train} == {0, 1}
    assert {int(row["label"]) for row in dev} == {0, 1}


def test_mine_clip_uses_matched_noise_instead_of_label_correlated_zeros() -> None:
    rng = np.random.default_rng(4)
    speech = 0.1 * np.sin(2 * np.pi * 180 * np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE)
    short_room_tail = rng.normal(0.0, 0.001, size=int(0.05 * SAMPLE_RATE))
    audio = np.concatenate([speech, short_room_tail]).astype(np.float32)
    samples = mine_clip(Clip(id="room", audio=audio, label=1), grid=[0.2], min_pause=0.2)
    assert len(samples) == 1
    assert samples[0].label == 1
    fabricated_tail = samples[0].audio[len(audio):]
    assert len(fabricated_tail) > 0
    assert not np.all(fabricated_tail == 0.0)
    assert float(np.sqrt(np.mean(fabricated_tail**2))) > 1e-5
    assert samples[0].fvad == [0, 0, 0, 0]
    assert samples[0].fvad_mask == [0, 0, 0, 0]


def test_exactly_200ms_internal_pause_is_mined_at_score_point() -> None:
    speech = (0.1 * np.sin(2 * np.pi * 180 * np.arange(int(0.4 * SAMPLE_RATE)) / SAMPLE_RATE)).astype(np.float32)
    pause = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([speech, pause, speech])
    spans = silence_spans(audio, min_silence=0.2)
    assert any(abs(span.duration - 0.2) < 1e-8 for span in spans)

    samples = mine_clip(
        Clip(id="exact-boundary", audio=audio, label=1),
        score_point=0.2,
        grid=[0.2],
        min_pause=0.2,
        min_speech_before=0.3,
    )
    internal = [sample for sample in samples if sample.kind == "internal"]
    assert len(internal) == 1
    assert internal[0].label == 0
    assert internal[0].time_to_onset == pytest.approx(0.0, abs=1e-8)
    assert internal[0].cut_time == pytest.approx(0.6, abs=1e-8)


def _clip(identifier: str, label: int) -> Clip:
    audio = np.linspace(-0.1, 0.1, SAMPLE_RATE // 20, dtype=np.float32)
    return Clip(id=identifier, audio=audio, label=label, source="mock", agent_text="")


def test_acquisition_resume_is_append_safe_relative_and_relocatable(tmp_path: Path) -> None:
    out = tmp_path / "raw"
    common = {
        "dataset": "mock/smart-turn",
        "revision": "deadbeef",
        "language": "eng",
        "seed": 11,
        "progress_every": 0,
    }
    first = acquire_clips([_clip("../hold", 0), _clip("eot", 1)], out, per_label=1, **common)
    assert first["completed"]
    rows = read_jsonl_records(out / "clips.jsonl")
    assert len(rows) == 2
    assert all(not Path(row["path"]).is_absolute() for row in rows)
    assert all(".." not in Path(row["path"]).parts for row in rows)
    # Simulate a crash after a valid final JSON object but before its newline was flushed.
    manifest = out / "clips.jsonl"
    manifest.write_text(manifest.read_text().rstrip("\n"))

    second = acquire_clips(
        [_clip("../hold", 0), _clip("eot", 1), _clip("hold-2", 0), _clip("eot-2", 1)],
        out,
        per_label=2,
        resume=True,
        **common,
    )
    assert second["counts"] == {"0": 2, "1": 2}
    assert len(read_jsonl_records(out / "clips.jsonl")) == 4

    missing_row = read_jsonl_records(out / "clips.jsonl")[0]
    missing_path = out / missing_row["path"]
    missing_path.unlink()
    repaired = acquire_clips(
        [_clip("../hold", 0), _clip("eot", 1), _clip("hold-2", 0), _clip("eot-2", 1)],
        out,
        per_label=2,
        resume=True,
        **common,
    )
    assert repaired["completed"]
    assert missing_path.is_file()

    relocated = tmp_path / "relocated"
    out.rename(relocated)
    loaded = list(_read_manifest(relocated / "clips.jsonl"))
    assert len(loaded) == 4
    assert {clip.label for clip in loaded} == {0, 1}


def test_mining_resume_stores_only_last_window_and_relative_paths(tmp_path: Path) -> None:
    samples = tmp_path / "mined"
    audio = np.linspace(-0.2, 0.2, SAMPLE_RATE * 10, dtype=np.float32)
    sample = Sample(
        id="../../unsafe/sample",
        clip_id="clip",
        audio=audio,
        label=1,
        kind="final",
        cut_time=10.0,
        pause_start=9.8,
        time_to_onset=None,
        fvad=[0, 0, 0, 0],
        fvad_mask=1,
        source="mock",
        agent_text="",
    )
    first = write_samples([sample], samples)
    second = write_samples([sample], samples, resume=True)
    assert first["written"] == 1
    assert second["written"] == 0
    assert second["skipped"] == 1

    row = read_jsonl_records(samples / "samples.jsonl")[0]
    assert not Path(row["path"]).is_absolute()
    assert ".." not in Path(row["path"]).parts
    stored, sr = sf.read(samples / row["path"], dtype="float32")
    assert sr == SAMPLE_RATE
    assert len(stored) == int(WINDOW_SECONDS * SAMPLE_RATE)

    stored_path = samples / row["path"]
    stored_path.unlink()
    repaired = write_samples([sample], samples, resume=True)
    repaired_row = read_jsonl_records(samples / "samples.jsonl")[0]
    assert repaired["skipped"] == 1
    assert stored_path.is_file()
    assert repaired_row["sha256"]

    relocated = tmp_path / "moved-mined"
    samples.rename(relocated)
    resolved = read_samples(relocated / "samples.jsonl")
    assert Path(resolved[0]["path"]).is_file()


def test_telephony_augmentation_is_worker_order_independent_per_epoch(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    signal = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32) * 0.1
    sf.write(wav, signal, SAMPLE_RATE)
    rows = [{"id": "stable-id", "path": str(wav), "label": 0, "fvad": [0, 0, 0, 0], "fvad_mask": 0}]
    dataset = MinedDataset(rows, telephony_prob=1.0, seed=9)

    dataset.set_epoch(2)
    first = dataset[0]["input_features"]
    repeated = dataset[0]["input_features"]
    assert np.array_equal(first.numpy(), repeated.numpy())

    dataset.set_epoch(3)
    next_epoch = dataset[0]["input_features"]
    assert not np.array_equal(first.numpy(), next_epoch.numpy())
