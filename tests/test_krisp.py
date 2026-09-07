"""Krisp policy engine: scalar/vectorised agreement, boundary rules and loud failures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eot.eval.krisp import evaluate, metrics, sweep


def test_krisp_counts_timeout_cutoff_and_excludes_boundary():
    holds = [{"label": "hold", "span_len": 1.5, "points": [(0.2, 0.0)]}]
    eots = [{"label": "eot", "span_len": 2.0, "points": [(0.2, 0.0)]}]
    assert evaluate(holds + eots, 0.5, 0.2, 1.0)["cutoff_rate"] == 1.0
    assert evaluate(holds + eots, 0.5, 0.2, 1.5)["cutoff_rate"] == 0.0


def test_evaluate_applies_the_latching_rule_to_holds_and_eots():
    early = [(0.2, 0.9), (0.3, 0.3), (0.4, 0.3), (0.5, 0.3)]
    hold = {"label": "hold", "span_len": 0.5, "points": early}
    eot = {"label": "eot", "span_len": 2.0, "points": early}
    r = evaluate([hold, eot], 0.5, 0.4, 2.0)
    assert r["cutoff_rate"] == 1.0 and r["mean_latency"] == pytest.approx(0.4) and r["detect_rate"] == 1.0
    r = evaluate([hold, eot], 0.5, 0.5, 2.0)  # firing exactly when speech resumes is not a cutoff
    assert r["cutoff_rate"] == 0.0 and r["mean_latency"] == pytest.approx(0.5)
    r = evaluate([hold, eot], 0.9, 0.2, 2.0)  # p == threshold does not fire
    assert r["cutoff_rate"] == 0.0 and r["detect_rate"] == 0.0 and r["mean_latency"] == pytest.approx(2.0)


def test_vectorized_krisp_sweep_matches_scalar_replay():
    rng = np.random.default_rng(7)
    spans = []
    for i in range(30):
        duration = float(rng.choice([0.3, 0.5, 1.0, 1.2, 2.5]))
        points = [(round(float(t), 1), float(rng.uniform())) for t in np.arange(0.2, duration + 1e-6, 0.1)]
        spans.append(dict(label="hold" if i % 2 else "eot", span_len=duration, points=points))
    fast = sweep(spans)
    for row in fast[::23]:
        ref = evaluate(spans, row["threshold"], row["action_delay"], row["timeout"])
        for key in ["cutoff_rate", "mean_latency", "median_latency", "detect_rate", "timeout_rate"]:
            assert row[key] == pytest.approx(ref[key])


def test_krisp_metrics_fails_loudly_on_missing_predictions(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        metrics(tmp_path / "missing.jsonl", bootstrap_reps=1)
