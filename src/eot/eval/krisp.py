"""Krisp Turn-Taking Test v1 as a *second external test set* (benchmark-only licence).

The dataset LICENSE forbids training, fine-tuning and derivative models; nothing here ever
writes Krisp audio into a training manifest. Each clip ends with a labelled trailing silence
(``last_silence_duration``) and a human turn label (``shift`` = end of turn, ``hold``). We turn
that into one causal decision per clip on the same grid the EoT Bench harness uses (score every
100 ms of silence, official score point 0.2 s) so the policy sweep, false-cutoff and latency
numbers are comparable in spirit, and we bootstrap by ``speaker_id`` (30 speakers) because clips
from one speaker are not independent.

Subcommands
-----------
    uv run eot-krisp ingest  --parquet data/raw/krisp/hf/data/test.parquet --out data/raw/krisp/clips
    uv run eot-krisp score   --clips data/raw/krisp/clips/clips.jsonl --onnx exports/r2/eot.onnx --out eval/krisp/r2.jsonl
    uv run eot-krisp metrics --predictions eval/krisp/r2.jsonl --out eval/krisp/r2.metrics.json

Exclusions (reported, pre-registered): hold clips whose tail is < 0.2 s cannot be scored at the
score point and are dropped, mirroring the harness ``--min-hold-span-duration 0.2``; tails
> 5 s are clipped to 5 s like ``--max-hold-span-duration``.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eot.audio import GRID_STEP, SAMPLE_RATE, SCORE_POINT, decode_payload, load_wav
from eot.io import atomic_write_json, atomic_write_jsonl, read_jsonl_records, sha256_file, write_wav
from eot.metrics import pareto_front, roc_auc
from eot.onnx import cpu_session




MIN_TAIL = 0.2
MAX_TAIL = 5.0
KRISP_REVISION = "ea19b2743a49b2c2452bf3ced948712020b962bc"


def ingest(parquet: Path, out: Path) -> Path:
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    rows = []
    table = pq.ParquetFile(parquet)
    for batch in table.iter_batches(batch_size=64):
        for r in batch.to_pylist():
            x = decode_payload({"bytes": r["audio"]["bytes"]})
            cid = Path(r["filename"]).stem
            path = out / "audio" / f"{cid}.wav"
            rows.append({
                "id": cid, "path": path.relative_to(out).as_posix(), "label": r["label"],
                "duration": float(r["duration"]), "measured_duration": round(len(x) / SAMPLE_RATE, 3),
                "last_silence_duration": float(r["last_silence_duration"]),
                "speaker_id": str(r["speaker_id"]), "age": r.get("age"), "gender": r.get("gender"),
                "sha256": write_wav(path, x, SAMPLE_RATE),
            })
    manifest = out / "clips.jsonl"
    atomic_write_jsonl(manifest, rows)
    atomic_write_json(out / "metadata.json", {
        "dataset": "Krisp-AI/turn-taking-test-v1", "revision": KRISP_REVISION, "n": len(rows),
        "labels": {k: sum(r["label"] == k for r in rows) for k in ("shift", "hold")},
        "licence": "benchmark only; never used for training",
    })
    return manifest


def score(clips_manifest: Path, adapter, out: Path, batch_size: int = 32) -> dict:
    """Write harness-style prediction rows: one span per clip, scored every 100 ms from 0.2 s."""
    rows = read_jsonl_records(clips_manifest)
    root = clips_manifest.parent
    n_excluded_short = 0
    pending, meta = [], []
    written = 0
    partial = out.with_suffix(out.suffix + ".partial")
    with partial.open("w") as f:

        def flush():
            nonlocal written
            if not pending:
                return
            for p, m in zip(adapter.predict_batch(pending), meta):
                f.write(json.dumps({**m, "p_eot": float(p)}) + "\n")
                written += 1
            pending.clear()
            meta.clear()

        for r in rows:
            tail = float(r["last_silence_duration"])
            if tail + 1e-9 < MIN_TAIL:
                n_excluded_short += 1
                continue
            x = load_wav(root / r["path"])
            total = len(x) / SAMPLE_RATE
            speech_end = total - tail
            span_len = min(tail, MAX_TAIL)
            t = SCORE_POINT
            while t <= span_len + 1e-9:
                cut = speech_end + t
                pending.append({"audio": {"array": x[: int(cut * SAMPLE_RATE)], "sampling_rate": SAMPLE_RATE}, "messages": []})
                meta.append({
                    "id": r["id"], "span_index": 0, "timestamp": round(cut, 3), "silence_dur": round(t, 3),
                    "label": "eot" if r["label"] == "shift" else "hold", "speaker_id": r["speaker_id"],
                    "span_len": round(span_len, 3), "tail": round(tail, 3),
                })
                if len(pending) >= batch_size:
                    flush()
                t += GRID_STEP
        flush()
    partial.replace(out)
    report = {"rows_written": written, "clips": len(rows), "excluded_tail_lt_0.2": n_excluded_short,
              "adapter_id": adapter.adapter_id, "clips_sha256": sha256_file(clips_manifest),
              "model_sha256": getattr(adapter, "model_sha", None)}
    out.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _load_spans(predictions: Path) -> list[dict]:
    if not predictions.is_file():
        raise FileNotFoundError(f"predictions file not found: {predictions}")
    spans: dict[str, dict] = {}
    for r in read_jsonl_records(predictions):
        s = spans.setdefault(r["id"], {"id": r["id"], "label": r["label"], "speaker_id": r.get("speaker_id", "?"), "span_len": float(r["span_len"]), "points": []})
        s["points"].append((float(r["silence_dur"]), float(r["p_eot"])))
    for s in spans.values():
        s["points"].sort()
    return list(spans.values())


def _fire(points, span_len, thr, delay, timeout):
    """(fire_time, fired_by_model, timed_out). Silence beyond the clip is unobservable: a span shorter
    than the timeout that never fires is censored (no decision inside the clip)."""
    for sil, p in points:
        if sil >= timeout:
            return timeout, False, True
        if sil >= delay - 1e-9 and p >= thr:
            return sil, True, False
    if span_len >= timeout:
        return timeout, False, True
    return None, False, False


def evaluate(spans: list[dict], thr: float, delay: float, timeout: float) -> dict:
    holds = [s for s in spans if s["label"] == "hold"]
    eots = [s for s in spans if s["label"] == "eot"]
    cut = 0
    for s in holds:
        t, by_model, _ = _fire(s["points"], s["span_len"], thr, delay, timeout)
        cut += int(t is not None and t < s["span_len"] - 1e-9)
    lat, detected, timed_out = [], 0, 0
    for s in eots:
        t, by_model, to = _fire(s["points"], s["span_len"], thr, delay, timeout)
        if t is None:
            t = max(s["span_len"], timeout)  # nothing fired inside the clip: at best the timeout
            to = True
        lat.append(t)
        detected += int(by_model)
        timed_out += int(to)
    return {
        "threshold": thr, "action_delay": delay, "timeout": timeout,
        "cutoff_rate": cut / max(1, len(holds)), "mean_latency": float(np.mean(lat)) if lat else float("nan"),
        "median_latency": float(np.median(lat)) if lat else float("nan"),
        "detect_rate": detected / max(1, len(eots)), "timeout_rate": timed_out / max(1, len(eots)),
        "n_hold": len(holds), "n_eot": len(eots),
    }


def sweep(spans: list[dict]) -> list[dict]:
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    delays = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    timeouts = (1.0, 1.5, 2.0, 2.5, 3.0)
    width = max((len(s["points"]) for s in spans), default=0)
    if not spans or width == 0:
        return []
    times = np.full((len(spans), width), np.inf)
    probs = np.full_like(times, -np.inf)
    for i, s in enumerate(spans):
        for j, (t, prob) in enumerate(s["points"]):
            times[i, j], probs[i, j] = t, prob
    hold = np.array([s["label"] == "hold" for s in spans])
    lengths = np.array([s["span_len"] for s in spans])
    results = []
    for threshold in thresholds:
        positive = probs >= threshold
        for delay in delays:
            crossing = np.min(np.where(positive & (times >= delay - 1e-9), times, np.inf), axis=1)
            for timeout in timeouts:
                fire = np.minimum(crossing, np.where(lengths >= timeout, timeout, np.inf))
                latency = np.minimum(crossing[~hold], timeout)
                detected = crossing[~hold] < timeout
                results.append({
                    "threshold": float(threshold), "action_delay": delay, "timeout": timeout,
                    "cutoff_rate": float(np.mean(fire[hold] < lengths[hold] - 1e-9)) if hold.any() else 0.,
                    "mean_latency": float(np.mean(latency)) if len(latency) else float("nan"),
                    "median_latency": float(np.median(latency)) if len(latency) else float("nan"),
                    "detect_rate": float(np.mean(detected)) if len(detected) else 0.,
                    "timeout_rate": float(np.mean(~detected)) if len(detected) else 0.,
                    "n_hold": int(hold.sum()), "n_eot": int((~hold).sum()),
                })
    return results


def operating_points(results: list[dict]) -> dict:
    def best_cut_at(budget):
        c = [r for r in results if r["mean_latency"] <= budget + 1e-9]
        return min(c, key=lambda r: (r["cutoff_rate"], r["mean_latency"])) if c else None

    def best_lat_at(budget):
        c = [r for r in results if r["cutoff_rate"] <= budget + 1e-9]
        return min(c, key=lambda r: (r["mean_latency"], r["cutoff_rate"])) if c else None

    return {
        "best_cutoff_at_0.3s": best_cut_at(0.3), "best_cutoff_at_0.6s": best_cut_at(0.6),
        "best_latency_at_5pct": best_lat_at(0.05), "best_latency_at_10pct": best_lat_at(0.10),
    }


def auc_at_score_point(spans: list[dict], score_point: float = SCORE_POINT) -> float:
    ys, ps = [], []
    for s in spans:
        for sil, p in s["points"]:
            if abs(sil - score_point) < 1e-6:
                ys.append(1 if s["label"] == "eot" else 0)
                ps.append(p)
                break
    return roc_auc(np.asarray(ys), np.asarray(ps)) if ys else float("nan")


def grouped_bootstrap(spans: list[dict], policy: dict, reps: int = 1000, seed: int = 0) -> dict:
    """Resample speakers with replacement; report 95 % CIs for cutoff rate and mean latency."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        groups[s["speaker_id"]].append(s)
    keys = list(groups)
    cuts, lats = [], []
    for _ in range(reps):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        sample = [s for k in picked for s in groups[keys[k]]]
        r = evaluate(sample, policy["threshold"], policy["action_delay"], policy["timeout"])
        cuts.append(r["cutoff_rate"])
        lats.append(r["mean_latency"])
    return {
        "n_groups": len(keys), "reps": reps,
        "cutoff_rate_ci95": [float(np.percentile(cuts, 2.5)), float(np.percentile(cuts, 97.5))],
        "mean_latency_ci95": [float(np.nanpercentile(lats, 2.5)), float(np.nanpercentile(lats, 97.5))],
    }


def metrics(predictions: Path, bootstrap_reps: int = 1000) -> dict:
    spans = _load_spans(predictions)
    results = sweep(spans)
    ops = operating_points(results)
    vad = [{**s, "points": [(t, 1.0) for t, _ in s["points"]]} for s in spans]
    vad_ops = operating_points(sweep(vad))
    ci = {k: grouped_bootstrap(spans, v, bootstrap_reps) for k, v in ops.items() if v is not None}
    return {
        "n_clips": len(spans), "n_hold": sum(s["label"] == "hold" for s in spans), "n_eot": sum(s["label"] == "eot" for s in spans),
        "metric_version": 3,
        "bootstrap_interpretation": "conditional on policies selected on this benchmark; descriptive, not independent policy validation",
        "auc_at_0.2s": auc_at_score_point(spans), "operating_points": ops, "bootstrap_by_speaker": ci,
        "vad_baseline": vad_ops,
        "pareto": sorted(
            ({"cutoff_rate": r["cutoff_rate"], "mean_latency": r["mean_latency"], "threshold": r["threshold"], "action_delay": r["action_delay"], "timeout": r["timeout"]}
             for r in pareto_front(results)), key=lambda r: r["mean_latency"]),
    }


class SmartTurnPublicAdapter:
    """Public Smart Turn v3.2 ONNX (pipecat-ai/smart-turn-v3), mirroring the harness adapter's
    preprocessing (left-pad/truncate to 8 s, WhisperFeatureExtractor chunk_length=8, do_normalize)."""

    adapter_id = "pipecat-ai/smart-turn-v3.2-public"

    def __init__(self, filename: str = "smart-turn-v3.2-gpu.onnx"):
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        path = hf_hub_download("pipecat-ai/smart-turn-v3", filename,
                               revision="f766f81d3cfdf7737ac64aad813d91bbfd56bf93")
        self.model_sha = sha256_file(Path(path))
        self.session = cpu_session(path, threads=2)
        self.fe = WhisperFeatureExtractor(chunk_length=8)
        self.max_samples = 8 * SAMPLE_RATE

    def predict_batch(self, batch):
        feats = []
        for item in batch:
            arr = np.asarray(item["audio"]["array"], dtype=np.float32)
            arr = arr[-self.max_samples :] if len(arr) > self.max_samples else np.pad(arr, (self.max_samples - len(arr), 0))
            f = self.fe(arr, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length", max_length=self.max_samples, truncation=True, do_normalize=True)
            feats.append(f.input_features[0])
        probs = self.session.run(None, {"input_features": np.stack(feats).astype(np.float32)})[0].reshape(-1)
        return [float(p) for p in probs]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("--parquet", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("score")
    b.add_argument("--clips", type=Path, required=True)
    b.add_argument("--checkpoint", default=None)
    b.add_argument("--onnx", default=None)
    b.add_argument("--vad-baseline", action="store_true", help="p_eot = 1 everywhere (no model)")
    b.add_argument("--smart-turn-public", action="store_true", help="score with the public Smart Turn v3.2 ONNX (R1)")
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("metrics")
    c.add_argument("--predictions", type=Path, required=True)
    c.add_argument("--out", type=Path, default=None)
    c.add_argument("--bootstrap-reps", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.cmd == "ingest":
        print(ingest(args.parquet, args.out))
    elif args.cmd == "score":
        if args.vad_baseline:
            class _VAD:
                adapter_id = "vad-baseline"

                def predict_batch(self, batch):
                    return [1.0] * len(batch)

            adapter = _VAD()
        elif args.smart_turn_public:
            adapter = SmartTurnPublicAdapter()
        else:
            from eot.eval.eotbench import EOTAdapter

            adapter = EOTAdapter(checkpoint=args.checkpoint, onnx=args.onnx)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(score(args.clips, adapter, args.out), indent=2))
    else:
        report = metrics(args.predictions, args.bootstrap_reps)
        text = json.dumps(report, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
        print(json.dumps({k: v for k, v in report.items() if k != "pareto"}, indent=2))


if __name__ == "__main__":
    main()
