"""Censoring-correct training (L009): per-row targets/weights derived from the observation window,
backward compatibility of old manifests and the weighted loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import tiny_model

from eot.audio import SAMPLE_RATE
from eot.data.dataset import MinedDataset, row_targets, row_weight
from eot.io import write_wav
from eot.labeling.samples import observed_fvad_targets
from eot.modeling.train import filter_time_to_onset

SR = SAMPLE_RATE


def test_observed_fvad_targets_uses_the_reason_to_settle_a_zero_window_tie():
    # Resumed on the next sample (the resumption closed the window): every horizon is known.
    assert observed_fvad_targets(0.0, 0.0, ended_by_resumption=True) == ([1, 1, 1, 1], [1, 1, 1, 1])
    # Agent started before the cut and the speaker came back exactly at the cut: the window is 0
    # and was closed by the agent, so nothing was observed under waiting (the numbers alone tie).
    assert observed_fvad_targets(0.0, 0.0, ended_by_resumption=False) == ([1, 1, 1, 1], [0, 0, 0, 0])
    assert observed_fvad_targets(0.0, 0.0) == ([1, 1, 1, 1], [1, 1, 1, 1])  # without the reason: the tie counts
    # Resumption after an agent onset that closed a positive window: only the window is asserted.
    assert observed_fvad_targets(0.9, 0.3, ended_by_resumption=False) == ([0, 0, 1, 1], [1, 0, 0, 0])
    assert observed_fvad_targets(None, 0.3, ended_by_resumption=False) == ([0, 0, 0, 0], [1, 0, 0, 0])
    tie = {
        "label": 0,
        "time_to_onset": 0.0,
        "future_observed_until": 0.0,
        "observation_end_reason": "agent_onset",
        "fvad": [1, 1, 1, 1],
        "fvad_mask": [1, 1, 1, 1],
    }
    assert row_targets(tie) == ([1, 1, 1, 1], [0, 0, 0, 0], True)  # the dataset applies the same rule


# ----------------------------------------------------------------------------- dataset / loss
def test_row_targets_derive_the_mask_from_the_window_and_fall_back_for_legacy_rows():
    derived = {
        "label": 1,
        "time_to_onset": None,
        "future_observed_until": 0.7,
        "observation_end_reason": "clip_end",
        "fvad": [0, 0, 0, 0],
        "fvad_mask": 1,
    }
    assert row_targets(derived) == ([0, 0, 0, 0], [1, 1, 0, 0], True)
    hold = {
        "label": 0,
        "time_to_onset": 0.5,
        "future_observed_until": 0.5,
        "observation_end_reason": "customer_resumed",
        "fvad": [9, 9, 9, 9],
        "fvad_mask": 0,
    }
    assert row_targets(hold) == ([0, 1, 1, 1], [1, 1, 1, 1], True)  # stored values are ignored when the window is present
    # Intermediate format (apptek.py before L009): future_observed_until was the raw span to the file
    # end, with no observation_end_reason. Deriving from it would assert silence while the agent
    # spoke, so such rows keep today's rule (the L006 guard).
    intermediate = {"label": 1, "time_to_onset": None, "future_observed_until": 7.3, "fvad": [0, 0, 0, 0], "fvad_mask": 1}
    assert row_targets(intermediate) == ([0, 0, 0, 0], [0, 0, 0, 0], False)
    legacy_eot = {"label": 1, "time_to_onset": None, "fvad": [0, 0, 0, 0], "fvad_mask": 1}
    assert row_targets(legacy_eot) == ([0, 0, 0, 0], [0, 0, 0, 0], False)  # today's guard: EOT clip label proves nothing
    legacy_hold = {"label": 0, "time_to_onset": 0.28, "fvad": [0, 1, 1, 1], "fvad_mask": 1}
    assert row_targets(legacy_hold) == ([0, 1, 1, 1], [1, 1, 1, 1], False)
    legacy_whole = {"label": 0, "time_to_onset": None, "fvad": [0, 0, 0, 0], "fvad_mask": 0}
    assert row_targets(legacy_whole) == ([0, 0, 0, 0], [0, 0, 0, 0], False)
    null_window = {"label": 1, "future_observed_until": None, "fvad": [0, 0, 0, 0], "fvad_mask": [1, 1, 1, 1]}
    assert row_targets(null_window)[1:] == ([0, 0, 0, 0], False)


def test_row_weight_defaults_to_one_and_rejects_invalid_values():
    assert row_weight({}) == 1.0 and row_weight({"weight": None}) == 1.0 and row_weight({"weight": 0}) == 0.0
    assert row_weight({"weight": "0.25"}) == 0.25
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="invalid weight"):
            row_weight({"id": "r", "weight": bad})


def test_dataset_emits_weight_and_counts_legacy_rows(tmp_path: Path):
    wav = tmp_path / "s.wav"
    write_wav(wav, (0.05 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)).astype(np.float32), SR)
    legacy = {"id": "old", "path": str(wav), "label": 1, "fvad": [0, 0, 0, 0], "fvad_mask": 1}
    new = {
        "id": "new",
        "path": str(wav),
        "label": 1,
        "fvad": [0, 0, 0, 0],
        "fvad_mask": [1, 1, 1, 1],
        "time_to_onset": None,
        "future_observed_until": 0.7,
        "observation_end_reason": "clip_end",
        "weight": 0.25,
    }
    ds = MinedDataset([legacy, new])
    assert ds.legacy_counts == {"rows": 2, "rows_without_observation_window": 1, "rows_without_weight": 1}
    old_item, new_item = ds[0], ds[1]
    assert old_item["fvad_mask"].tolist() == [0, 0, 0, 0] and old_item["weight"].item() == 1.0
    assert new_item["fvad_mask"].tolist() == [1, 1, 0, 0] and new_item["weight"].item() == 0.25
    assert new_item["fvad"].dtype == torch.float32 and new_item["weight"].dtype == torch.float32
    with pytest.raises(ValueError, match="invalid weight"):
        MinedDataset([{**legacy, "weight": -2}])


def test_loss_eot_is_unchanged_with_unit_weights_and_zero_weight_excludes_rows():
    model = tiny_model(max_source_positions=100)
    x = torch.randn(4, 80, 200)
    labels = torch.tensor([0, 1, 0, 1])
    with torch.no_grad():
        plain = model(x, None, labels)["loss_eot"]
        ones = model(x, None, labels, weight=torch.ones(4))["loss_eot"]
        scaled = model(x, None, labels, weight=torch.full((4,), 0.3))["loss_eot"]
        partial = model(x, None, labels, weight=torch.tensor([1.0, 1.0, 0.0, 0.0]))["loss_eot"]
        subset = model(x[:2], None, labels[:2])["loss_eot"]
        zeros = model(x, None, labels, weight=torch.zeros(4))["loss"]
    torch.testing.assert_close(ones, plain)
    torch.testing.assert_close(scaled, plain)  # weights are relative
    torch.testing.assert_close(partial, subset, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(zeros) and zeros.item() == 0.0


def test_filter_time_to_onset_clamps_float_noise_and_excludes_real_overshoots():
    rows = [
        {"id": "ok", "time_to_onset": 0.1},
        {"id": "eot", "time_to_onset": None},
        {"id": "noise", "time_to_onset": -9e-16},
        {"id": "overshoot", "time_to_onset": -0.02},
    ]
    kept, counts = filter_time_to_onset(rows)
    assert [r["id"] for r in kept] == ["ok", "eot", "noise"]
    assert kept[2]["time_to_onset"] == 0.0
    assert counts == {"excluded_cuts_after_speech_onset": 1, "clamped_float_noise_time_to_onset": 1}


def test_old_manifest_rows_still_load_with_todays_semantics(tmp_path: Path):
    """A row shaped like the stored smart-turn-ensemble / apptek-oracle manifests (scalar mask, no window, no weight)."""
    wav = tmp_path / "s.wav"
    write_wav(wav, np.zeros(SR, np.float32), SR)
    old_final = {
        "id": "f",
        "path": str(wav),
        "label": 1,
        "kind": "final",
        "cut_time": 6.2,
        "pause_start": 6.0,
        "time_to_onset": None,
        "fvad": [0, 0, 0, 0],
        "fvad_mask": 1,
        "noise_fill": 0,
    }
    old_internal = {
        "id": "i",
        "path": str(wav),
        "label": 0,
        "kind": "internal",
        "cut_time": 6.2,
        "pause_start": 6.0,
        "time_to_onset": 0.28,
        "fvad": [0, 1, 1, 1],
        "fvad_mask": 1,
        "noise_fill": 0,
    }
    old_whole = {**old_internal, "id": "w", "kind": "whole", "time_to_onset": None, "fvad": [0, 0, 0, 0], "fvad_mask": 0}
    ds = MinedDataset([old_final, old_internal, old_whole])
    assert ds.legacy_counts["rows_without_observation_window"] == 3
    assert ds[0]["fvad_mask"].tolist() == [0, 0, 0, 0]
    assert ds[1]["fvad_mask"].tolist() == [1, 1, 1, 1] and ds[1]["fvad"].tolist() == [0, 1, 1, 1]
    assert ds[2]["fvad_mask"].tolist() == [0, 0, 0, 0]
    assert all(ds[i]["weight"].item() == 1.0 for i in range(3))
