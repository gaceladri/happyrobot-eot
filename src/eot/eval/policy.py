"""Endpointing policy and a lightweight local diagnostic sweep.

Three clocks, never mixed:
- model compute (ms per forward) -> measured by ``load_test``;
- endpointing delay / dead air (silence from last speech to commit) -> measured here;
- end-to-first-audio -> system property, out of scope for this module.

A raw P(EOT) is not a decision. This local replay has three useful knobs inspired by EoT Bench:
``threshold`` (confidence to commit), ``action_delay`` (minimum silence before acting) and
``timeout`` (commit anyway). ``replay_turn`` applies them causally to the scores a model
produced on the silence grid of one user turn (every silence >= 100 ms is a decision point; the
last one is EOT, earlier ones HOLD) and returns whether the turn was falsely cut and the dead
air at the true end. ``sweep`` grids the knobs for fast development feedback.

This module is deliberately not the authoritative benchmark: its turn-level aggregation and
timeout behavior are not identical to the current official harness. Submission metrics must
come from pinned ``eot-harness predict`` + ``compute-metrics``, as used by ``eot-runpod evaluate``.

Input rows follow eot-bench ``predictions.parquet``: ``id, span_index, timestamp, silence_dur,
p_eot, label`` (label in {"hold", "eot"}). The CLI accepts jsonl or parquet.

    uv run eot-sweep --predictions runs/base/eotbench_predictions.jsonl
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from eot.eval.metrics import pareto_front


GRID_STEP = 0.1  # eot-bench scores every 100 ms inside a silence span


@dataclass(frozen=True)
class Policy:
    threshold: float = 0.5
    action_delay: float = 0.2
    timeout: float = 2.5


@dataclass
class TurnOutcome:
    turn_id: str
    false_cutoff: bool
    eot_latency: float | None  # dead air at the true end (None if the eot span was never reached)
    detected_by_model: bool
    span_latencies: list[float]


def _fire_time(points: list[tuple[float, float]], span_len: float, pol: Policy) -> tuple[float | None, bool]:
    """First (silence_dur) at which the policy commits inside one span, and whether the model did it."""
    for sil, p in sorted(points):
        if sil >= pol.timeout:
            return pol.timeout, False
        if sil >= pol.action_delay and p >= pol.threshold:
            return sil, True
    if span_len >= pol.timeout:
        return pol.timeout, False
    return None, False


def replay_turn(spans: list[dict], pol: Policy) -> TurnOutcome:
    """``spans``: ordered list of {"label", "silence_len", "points": [(silence_dur, p_eot), ...]}."""
    lat: list[float] = []
    for i, sp in enumerate(spans):
        t, by_model = _fire_time(sp["points"], sp["silence_len"], pol)
        is_last = i == len(spans) - 1
        if sp["label"] == "hold" and t is not None:
            return TurnOutcome(sp.get("id", "?"), True, None, False, lat)
        if is_last:
            if t is None:  # eot span shorter than any commit point: dead air = its whole length + nothing fired
                t = max(sp["silence_len"], pol.timeout)
            return TurnOutcome(sp.get("id", "?"), False, t, by_model, lat)
    return TurnOutcome("?", False, None, False, lat)


def group_predictions(rows: Iterable[dict]) -> list[list[dict]]:
    """Group flat prediction rows into per-turn ordered spans."""
    turns: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        sp = turns[str(r["id"])].setdefault(int(r["span_index"]), {"id": str(r["id"]), "label": r["label"], "points": [], "silence_len": 0.0})
        sp["points"].append((float(r["silence_dur"]), float(r["p_eot"])))
        sp["silence_len"] = max(sp["silence_len"], float(r["silence_dur"]))
    out = []
    for tid, spans in turns.items():
        ordered = [spans[k] for k in sorted(spans)]
        out.append(ordered)
    return out


def evaluate_policy(turns: list[list[dict]], pol: Policy) -> dict:
    outcomes = [replay_turn(t, pol) for t in turns]
    n = len(outcomes)
    fc = sum(o.false_cutoff for o in outcomes) / max(1, n)
    lats = [o.eot_latency for o in outcomes if o.eot_latency is not None]
    return {
        "threshold": pol.threshold, "action_delay": pol.action_delay, "timeout": pol.timeout,
        "false_cutoff_rate": fc,
        "mean_latency_s": float(np.mean(lats)) if lats else float("nan"),
        "median_latency_s": float(np.median(lats)) if lats else float("nan"),
        "detect_rate": sum(o.detected_by_model for o in outcomes) / max(1, n),
        "n_turns": n,
    }


def sweep(
    turns: list[list[dict]],
    thresholds=np.round(np.arange(0.05, 1.0, 0.05), 2),
    action_delays=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
    timeouts=(1.5, 2.0, 2.5, 3.0),
) -> list[dict]:
    return [evaluate_policy(turns, Policy(float(t), float(a), float(o))) for t in thresholds for a in action_delays for o in timeouts]


def operating_points(results: list[dict]) -> dict:
    """Local diagnostic summary from a sweep; not an official leaderboard result."""
    def best_fc_at(budget):
        c = [r for r in results if r["mean_latency_s"] <= budget]
        return min(c, key=lambda r: r["false_cutoff_rate"]) if c else None

    def best_lat_at(budget):
        c = [r for r in results if r["false_cutoff_rate"] <= budget]
        return min(c, key=lambda r: r["mean_latency_s"]) if c else None

    def fmt(r, key):
        return None if r is None else {key: r[key], "policy": {k: r[k] for k in ("threshold", "action_delay", "timeout")}, "detect_rate": r["detect_rate"]}

    return {
        "fc@300ms": fmt(best_fc_at(0.3), "false_cutoff_rate"),
        "fc@600ms": fmt(best_fc_at(0.6), "false_cutoff_rate"),
        "latency@5%": fmt(best_lat_at(0.05), "mean_latency_s"),
        "latency@10%": fmt(best_lat_at(0.10), "mean_latency_s"),
    }


def pareto(results: list[dict]) -> list[dict]:
    return pareto_front(results, latency_key="mean_latency_s", cutoff_key="false_cutoff_rate")


def vad_baseline_rows(turns: list[list[dict]]) -> list[list[dict]]:
    """Silence-only baseline: p_eot = 1 everywhere, so only action_delay/timeout matter."""
    return [[{**sp, "points": [(s, 1.0) for s, _ in sp["points"]]} for sp in t] for t in turns]


class Endpointer:
    """Stateful causal endpointer: feed VAD frames and model scores, get commit decisions.

    Generation ids make stale inference results harmless: if speech resumes while a forward
    pass is in flight, the result for the old generation is discarded.
    """

    def __init__(self, policy: Policy, score_point: float = 0.2, score_mode: str = "score_point"):
        if score_mode not in ("score_point", "max"):
            raise ValueError("score_mode must be score_point or max")
        self.policy = policy
        self.score_point = score_point
        self.score_mode = score_mode
        self.generation = 0
        self.silence_start: float | None = None
        self.committed = False
        self.has_speech = False
        self.scored = False
        self.positive = False
        self.inflight = False
        self.last_request = float("-inf")

    def reset(self) -> None:
        """Start a new user turn and invalidate every in-flight score."""
        self.generation += 1
        self.silence_start = None
        self.committed = False
        self.has_speech = False
        self.scored = self.positive = self.inflight = False
        self.last_request = float("-inf")

    def on_vad(self, t: float, is_speech: bool) -> None:
        if is_speech:
            self.has_speech = True
            if self.silence_start is not None:
                self.generation += 1
                self.scored = self.positive = self.inflight = False
                self.last_request = float("-inf")
            self.silence_start = None
            self.committed = False
        elif self.has_speech and self.silence_start is None:
            self.silence_start = t

    def should_score(self, t: float) -> bool:
        return (self.silence_start is not None and not self.committed and not self.inflight
                and (t - self.silence_start) >= self.score_point
                and (not self.scored or self.score_mode == "max")
                and t - self.last_request >= GRID_STEP - 1e-9)

    def request_score(self, t: float) -> int | None:
        """Reserve one inference; callers pass its generation back with the result."""
        if not self.should_score(t):
            return None
        self.inflight = True
        self.last_request = t
        return self.generation

    def on_error(self, generation: int) -> None:
        if generation == self.generation:
            self.inflight = False
            if self.score_mode == "score_point":
                # A retry with later audio would change the calibrated scoring protocol.
                self.scored = True

    def on_tick(self, t: float) -> bool:
        """A timeout must work even if inference failed or no score arrived."""
        if (self.silence_start is not None and not self.committed
                and (t - self.silence_start >= self.policy.timeout or
                     (self.positive and t - self.silence_start >= self.policy.action_delay))):
            self.committed = True
            return True
        return False

    def on_score(self, t: float, p_eot: float, generation: int) -> bool:
        if generation != self.generation or self.silence_start is None or self.committed:
            return False
        self.inflight = False
        if self.scored and self.score_mode == "score_point":
            return self.on_tick(t)
        self.scored = True
        # Match the pinned harness's strict threshold and first-positive-score semantics.
        self.positive = self.positive or p_eot > self.policy.threshold
        return self.on_tick(t)


def _read_predictions(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    turns = group_predictions(_read_predictions(args.predictions))
    res = sweep(turns)
    vad = sweep(vad_baseline_rows(turns))
    report = {"model": operating_points(res), "vad_baseline": operating_points(vad), "pareto": pareto(res), "n_turns": len(turns)}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
