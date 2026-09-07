"""Endpointing policy: the harness fire rule, a causal endpointer and the ``eot-sweep`` CLI.

Three clocks, never mixed:
- model compute (ms per forward) -> measured by ``load_test``;
- endpointing delay / dead air (silence from last speech to commit) -> measured here;
- end-to-first-audio -> system property, out of scope for this module.

A raw P(EOT) is not a decision. A policy has three knobs, the same as EoT Bench: ``threshold``
(confidence to commit), ``action_delay`` (minimum silence before acting) and ``timeout`` (commit
anyway). :func:`fire_time` applies them to the scores a model produced on the silence grid of one
span exactly like the pinned harness (``third_party/eot-bench/eot_harness/metrics.py``): the first
score strictly above the threshold *latches* and the policy fires at
``max(action_delay, crossing)``; the timeout fires when the span lasts that long. Earlier local
readouts (the archived research) used a non-latching ``p >= threshold`` rule, so they are not
comparable with the numbers this rule produces. :class:`Endpointer` is the streaming form of the
same rule; :mod:`eot.eval.krisp` is the batch engine (``sweep`` / ``evaluate`` / ``operating_points``).

The local sweep scores every silence >= 100 ms inside a turn (the last one is EOT, earlier ones
HOLD) and aggregates per span, as the harness does. It is a development diagnostic, not an
official result: submission metrics come from pinned ``eot-harness predict`` + ``compute-metrics``.

Input rows follow eot-bench ``predictions.parquet``: ``id, span_index, timestamp, silence_dur,
p_eot, label`` (label in {"hold", "eot"}). The CLI accepts jsonl or parquet and prints
``{"model": operating_points, "vad_baseline": operating_points, "pareto": [...], "n_hold", "n_eot"}``
with the harness metric names (``cutoff_rate``, ``mean_latency``).

    uv run eot-sweep --predictions runs/base/eotbench_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from eot.audio import GRID_STEP

EPS = 1e-9  # the harness's boundary tolerance


@dataclass(frozen=True)
class Policy:
    threshold: float = 0.5
    action_delay: float = 0.2
    timeout: float = 2.5


def fire_time(
    points: Iterable[tuple[float, float]], span_len: float, thr: float, delay: float, timeout: float
) -> tuple[float | None, bool]:
    """``(fire_time, fired_by_model)`` for one silence span under the harness rule.

    ``points`` are ``(silence_dur, p_eot)`` pairs. The first point with ``p_eot > thr`` (strict)
    latches; the policy fires at ``max(delay, crossing)`` when that is within the timeout. Otherwise
    the timeout fires, but only if the span lasts that long: silence beyond the clip is
    unobservable, so a shorter span that never fires returns ``None`` (censored, no decision).
    """
    crossing = min((sil for sil, p in points if p > thr), default=math.inf)
    fire = max(delay, crossing)
    if fire <= timeout + EPS:
        return fire, True
    if span_len >= timeout:
        return timeout, False
    return None, False


def group_predictions(rows: Iterable[dict]) -> list[dict]:
    """Group flat prediction rows into engine spans ``{"id", "span_index", "label", "span_len", "points"}``.

    ``span_len`` is the longest scored silence (the scoring grid stops at the span end), so a
    fire at the last grid point is never a cutoff, as in :func:`eot.eval.krisp.evaluate`.
    """
    spans: dict[tuple[str, int], dict] = {}
    for r in rows:
        key = (str(r["id"]), int(r["span_index"]))
        sp = spans.setdefault(key, {"id": key[0], "span_index": key[1], "label": r["label"], "span_len": 0.0, "points": []})
        sp["points"].append((float(r["silence_dur"]), float(r["p_eot"])))
        sp["span_len"] = max(sp["span_len"], float(r["silence_dur"]))
    for sp in spans.values():
        sp["points"].sort()
    return [spans[k] for k in sorted(spans)]


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
        return (
            self.silence_start is not None
            and not self.committed
            and not self.inflight
            and (t - self.silence_start) >= self.score_point
            and (not self.scored or self.score_mode == "max")
            and t - self.last_request >= GRID_STEP - EPS
        )

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
        if (
            self.silence_start is not None
            and not self.committed
            and (t - self.silence_start >= self.policy.timeout or (self.positive and t - self.silence_start >= self.policy.action_delay))
        ):
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
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> None:
    # krisp imports fire_time from this module, so the engine is imported here, not at module level.
    from eot.eval.krisp import operating_points, pareto, sweep, vad_baseline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True, help="eot-bench prediction rows (.jsonl or .parquet)")
    ap.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    args = ap.parse_args(argv)
    spans = group_predictions(_read_predictions(args.predictions))
    results = sweep(spans)
    report = {
        "model": operating_points(results),
        "vad_baseline": operating_points(sweep(vad_baseline(spans))),
        "pareto": pareto(results),
        "n_hold": sum(s["label"] == "hold" for s in spans),
        "n_eot": sum(s["label"] == "eot" for s in spans),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
