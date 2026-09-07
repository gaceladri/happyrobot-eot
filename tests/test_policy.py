"""Endpointing policy: the harness fire rule, prediction grouping and the causal endpointer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eot.eval.policy import Endpointer, Policy, fire_time, group_predictions, main


def test_fire_time_latches_the_first_positive_score_until_the_action_delay():
    points = [(0.2, 0.9), (0.3, 0.3), (0.4, 0.3)]
    assert fire_time(points, 1.0, 0.5, delay=0.4, timeout=2.0) == (0.4, True)
    assert fire_time(points, 1.0, 0.5, delay=0.2, timeout=2.0) == (0.2, True)


def test_fire_time_threshold_is_strict():
    assert fire_time([(0.2, 0.5)], 1.0, 0.5, delay=0.2, timeout=2.0) == (None, False)
    assert fire_time([(0.2, 0.5 + 1e-6)], 1.0, 0.5, delay=0.2, timeout=2.0) == (0.2, True)


def test_fire_time_timeout_needs_the_span_to_last_that_long():
    assert fire_time([(0.2, 0.0)], 2.0, 0.5, delay=0.2, timeout=1.0) == (1.0, False)
    assert fire_time([(0.2, 0.0)], 0.5, 0.5, delay=0.2, timeout=1.0) == (None, False)
    # A crossing after the timeout is the timeout, not a model decision.
    assert fire_time([(0.2, 0.0), (1.5, 0.9)], 2.0, 0.5, delay=0.2, timeout=1.0) == (1.0, False)


def _rows():
    return [
        {"id": "t", "span_index": 1, "silence_dur": 0.3, "p_eot": 0.2, "label": "eot"},
        {"id": "t", "span_index": 0, "silence_dur": 0.2, "p_eot": 0.1, "label": "hold"},
        {"id": "t", "span_index": 1, "silence_dur": 0.2, "p_eot": 0.9, "label": "eot"},
    ]


def test_group_predictions_emits_sorted_engine_spans():
    spans = group_predictions(_rows())
    assert spans == [
        {"id": "t", "span_index": 0, "label": "hold", "span_len": 0.2, "points": [(0.2, 0.1)]},
        {"id": "t", "span_index": 1, "label": "eot", "span_len": 0.3, "points": [(0.2, 0.9), (0.3, 0.2)]},
    ]


def test_sweep_cli_reports_harness_metric_names(tmp_path: Path, capsys):
    preds = tmp_path / "preds.jsonl"
    preds.write_text("".join(json.dumps(r) + "\n" for r in _rows()))
    main(["--predictions", str(preds), "--out", str(tmp_path / "sweep.json")])
    report = json.loads((tmp_path / "sweep.json").read_text())
    assert report["n_hold"] == 1 and report["n_eot"] == 1
    assert set(report["model"]) == {"best_cutoff_at_0.3s", "best_cutoff_at_0.6s", "best_latency_at_5pct", "best_latency_at_10pct"}
    best = report["model"]["best_latency_at_5pct"]
    assert best["cutoff_rate"] == 0.0 and best["mean_latency"] == pytest.approx(0.2)
    assert {"cutoff_rate", "mean_latency"} <= set(report["pareto"][0])
    assert "n_hold" in capsys.readouterr().out


def test_endpointer_timeout_without_inference_and_stale_generation():
    e = Endpointer(Policy(timeout=1.0))
    e.on_vad(-0.1, True)
    e.on_vad(0.0, False)
    generation = e.generation
    e.on_vad(0.3, True)
    e.on_vad(0.4, False)
    assert not e.on_score(0.8, 1.0, generation)
    assert not e.on_tick(1.39)
    assert e.on_tick(1.41)
    assert not e.on_tick(2.0)


def test_endpointer_latches_early_score_until_action_delay():
    e = Endpointer(Policy(threshold=0.5, action_delay=0.5, timeout=1.0))
    e.on_vad(0.0, True)
    e.on_vad(0.1, False)
    g = e.request_score(0.31)
    assert g is not None
    assert e.request_score(0.32) is None
    assert not e.on_score(0.35, 0.8, g)
    assert not e.should_score(0.5)
    assert not e.on_tick(0.59)
    assert e.on_tick(0.61)
    e.reset()
    e.on_vad(1.0, False)
    assert not e.should_score(2.0)
    assert not e.on_tick(3.0)
