"""Numerical QA of the labeling pipeline. Run *before* training; results go in the write-up.

Checks (pre-registered thresholds in output/LABELING_PIPELINE_PLAN.md §2.4):

a. boundaries    detected customer pauses vs AppTek manual segment boundaries (recall @100/200 ms,
                 median |error|); manual boundaries are the only human timing we have.
b. pstn          pause-span Jaccard clean vs telephony_augment at 12/20/30 dB SNR for the legacy,
                 adaptive-energy and ensemble detectors (300 Smart Turn clips).
c. single_vs_dual on AppTek: the single-channel rule "same speaker resumes within T s -> HOLD"
                 against the dual-channel oracle. Contamination = oracle-EOT pauses the
                 single-channel rule calls HOLD. This bounds the bias of clip-level labeling.
d. listening     prepares a blind listening audit (random order, hidden key) and computes an
                 *automatic proxy*: Silero speech probability in the 160 ms before each cut
                 (a cut inside a word shows P(speech) > 0.5). The human audit stays the truth.
e. distribution  label ratio, kinds, source share, confidence tiers, pause lengths, accents.
f. krisp_tail    our detectors' trailing-silence estimate vs Krisp's human ``last_silence_duration``
                 (independent check of boundary accuracy on real conversational audio).

    uv run eot-qa run --smart-turn-clips data/raw/smart-turn-v3.2-eng/clips.jsonl \
        --mined data/mined/smart-turn-ensemble/samples.jsonl data/mined/smart-turn-legacy/samples.jsonl \
        --apptek data/mined/apptek-oracle --apptek-root data/raw/apptek/hf \
        --krisp-clips data/raw/krisp/clips/clips.jsonl --out eval/labeling_qa
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, PauseDetector, SileroVAD, resample, telephony_augment, to_float32, to_mono
from .data import read_jsonl_records, resolve_record_path


def _read(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    return resample(to_mono(to_float32(np.asarray(x))), sr)


def _pct(values, qs=(10, 50, 90)) -> dict:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return {"n": 0}
    return {"n": int(len(values)), **{f"p{q}": round(float(np.percentile(values, q)), 4) for q in qs}}


# ---------------------------------------------------------------------------
# a. boundaries vs manual AppTek segments
# ---------------------------------------------------------------------------


def check_boundaries(apptek_dir: Path, apptek_root: Path, tolerances=(0.1, 0.2)) -> dict:
    from .apptek import load_conversations

    convs = {c.id: c for c in load_conversations(apptek_root)}
    summaries = read_jsonl_records(apptek_dir / "conversations.jsonl")
    errors, matched, total, eot_boundary_total, eot_boundary_matched = [], Counter(), 0, 0, Counter()
    miss_reasons: Counter = Counter()
    for s in summaries:
        conv = convs.get(s["conversation"])
        if conv is None:
            continue
        kept = [p for p in s["customer_pauses"] if p["confidence"] != "low"]
        starts = np.array([p["start"] for p in kept] or [np.nan])
        ends = np.array([p["end"] for p in kept] or [np.nan])
        starts_all = np.array([p["start"] for p in s["customer_pauses"]] or [np.nan])
        segs = conv.segments
        for i, seg in enumerate(segs):
            if seg.role != "customer":
                continue
            nxt = next((t for t in segs[i + 1 :]), None)
            if nxt is None:
                continue
            gap = nxt.start - seg.end
            if gap < 0.3:
                continue  # manual boundaries with no audible pause are not testable
            total += 1
            is_eot_boundary = nxt.role == "agent"
            eot_boundary_total += int(is_eot_boundary)
            err = float(np.nanmin(np.abs(starts - seg.end))) if np.isfinite(starts).any() else float("inf")
            errors.append(err)
            for tol in tolerances:
                if err <= tol:
                    matched[tol] += 1
                    if is_eot_boundary:
                        eot_boundary_matched[tol] += 1
            if err > max(tolerances):
                # why was the manual boundary not matched?
                if np.isfinite(starts).any() and bool(np.any((starts <= seg.end) & (ends >= seg.end))):
                    miss_reasons["segment_end_inside_detected_pause (annotator padding; pause found earlier)"] += 1
                elif np.isfinite(starts_all).any() and float(np.nanmin(np.abs(starts_all - seg.end))) <= max(tolerances):
                    miss_reasons["only_low_confidence_pause_there (VAD veto)"] += 1
                else:
                    miss_reasons["no_pause_detected_next_is_" + nxt.role] += 1
    finite = [e for e in errors if np.isfinite(e)]
    return {
        "n_manual_gaps_ge_0.3s": total,
        "misses_at_200ms_by_reason": dict(miss_reasons),
        "recall": {f"@{int(t * 1000)}ms": round(matched[t] / max(1, total), 4) for t in tolerances},
        "recall_turn_boundaries": {f"@{int(t * 1000)}ms": round(eot_boundary_matched[t] / max(1, eot_boundary_total), 4) for t in tolerances},
        "n_turn_boundaries": eot_boundary_total,
        "abs_error_s_of_matched_200ms": _pct([e for e in finite if e <= 0.2]),
        "abs_error_s_all": _pct(finite),
        "threshold": "recall@100ms >= 0.9 and median error <= 60 ms",
        "passed": matched[0.1] / max(1, total) >= 0.9 and (float(np.median([e for e in finite if e <= 0.2])) <= 0.06 if finite else False),
    }


# ---------------------------------------------------------------------------
# b. PSTN robustness (Jaccard of pause spans clean vs augmented)
# ---------------------------------------------------------------------------


def _jaccard(a, b, total: float, hop: float = 0.01) -> float:
    g = np.arange(0, total, hop)
    ma = np.zeros(len(g), bool)
    mb = np.zeros(len(g), bool)
    for s in a:
        ma[(g >= s.start) & (g < s.end)] = True
    for s in b:
        mb[(g >= s.start) & (g < s.end)] = True
    u = (ma | mb).sum()
    return float((ma & mb).sum() / u) if u else float("nan")


def check_pstn(clips_manifest: Path, detectors: dict[str, PauseDetector], n_clips: int = 300, snrs=(12, 20, 30), seed: int = 0) -> dict:
    rows = read_jsonl_records(clips_manifest)
    rng = random.Random(seed)
    rows = rng.sample(rows, min(n_clips, len(rows)))
    noise_rng = np.random.default_rng(seed)
    out: dict = {"n_clips": len(rows), "snr_db": list(snrs), "jaccard_median": {}, "jaccard_mean": {}, "pauses_per_clip_clean": {}, "pauses_per_clip_noisy": {}}
    J = {(name, snr): [] for name in detectors for snr in snrs}
    n_clean = Counter()
    n_noisy = Counter()
    for row in rows:
        x = _read(resolve_record_path(clips_manifest, row["path"]))
        total = len(x) / SAMPLE_RATE
        clean = {name: [p for p in det(x) if p.confidence != "low"] for name, det in detectors.items()}
        for name in detectors:
            n_clean[name] += len(clean[name])
        for snr in snrs:
            y = telephony_augment(x, SAMPLE_RATE, noise_rng, snr_db=(snr, snr), drop_prob=0.0)
            for name, det in detectors.items():
                noisy = [p for p in det(y) if p.confidence != "low"]
                # Only clips where the clean detector found something are informative: 'nothing vs
                # nothing' is not agreement, it is absence of evidence (and it hid the legacy collapse).
                if clean[name]:
                    J[(name, snr)].append(_jaccard(clean[name], noisy, total))
                if snr == 20:
                    n_noisy[name] += len(noisy)
    out["clips_with_clean_pauses"] = {}
    for name in detectors:
        out["jaccard_median"][name] = {f"{snr}dB": round(float(np.nanmedian(J[(name, snr)])), 3) for snr in snrs}
        out["jaccard_mean"][name] = {f"{snr}dB": round(float(np.nanmean(J[(name, snr)])), 3) for snr in snrs}
        out["clips_with_clean_pauses"][name] = len(J[(name, snrs[0])])
        out["pauses_per_clip_clean"][name] = round(n_clean[name] / len(rows), 2)
        out["pauses_per_clip_noisy"][name] = round(n_noisy[name] / len(rows), 2)
    out["threshold"] = "median Jaccard >= 0.8 at 20 dB over clips with >=1 clean pause"
    out["passed"] = {name: out["jaccard_median"][name]["20dB"] >= 0.8 for name in detectors}
    return out


# ---------------------------------------------------------------------------
# c. single-channel rule vs dual-channel oracle (AppTek)
# ---------------------------------------------------------------------------


def check_single_vs_dual(apptek_dir: Path, horizons=(1.0, 2.0, 3.0, 5.0, float("inf"))) -> dict:
    summaries = read_jsonl_records(apptek_dir / "conversations.jsonl")
    decided = [d for s in summaries for d in s["decisions"] if d["label"] is not None]
    n_eot = sum(d["label"] == 1 for d in decided)
    n_hold = sum(d["label"] == 0 for d in decided)
    out = {"n_labeled_pauses": len(decided), "n_oracle_eot": n_eot, "n_oracle_hold": n_hold, "by_horizon": {}}
    eot_onset_gaps = [d["onset"] - d["pause_start"] for d in decided if d["label"] == 1 and d["onset"] is not None]
    out["oracle_eot_customer_returns_after_s"] = _pct(eot_onset_gaps)
    out["oracle_eot_agent_response_gap_s"] = _pct([d["agent_onset"] - d["pause_start"] for d in decided if d["label"] == 1 and d["agent_onset"] is not None])
    for T in horizons:
        is_single_hold = [d["onset"] is not None and d["onset"] - d["pause_start"] <= T for d in decided]
        single_hold = [d for d, h in zip(decided, is_single_hold) if h]
        single_eot = [d for d, h in zip(decided, is_single_hold) if not h]
        contaminated = sum(d["label"] == 1 for d in single_hold)
        missed_holds = sum(d["label"] == 0 for d in single_eot)
        out["by_horizon"][("inf" if T == float("inf") else f"{T:g}s")] = {
            "single_channel_hold": len(single_hold),
            "of_which_oracle_eot (contamination)": contaminated,
            "contamination_rate": round(contaminated / max(1, len(single_hold)), 4),
            "single_channel_eot": len(single_eot),
            "of_which_oracle_hold": missed_holds,
            "oracle_eot_recovered": round(sum(d["label"] == 1 for d in single_eot) / max(1, n_eot), 4),
        }
    return out


# ---------------------------------------------------------------------------
# d. blind listening audit (preparation + automatic proxy)
# ---------------------------------------------------------------------------


def prepare_listening(samples_manifests: list[Path], out_dir: Path, plan: dict[str, int], vad: SileroVAD, seed: int = 0) -> dict:
    rng = random.Random(seed)
    rows = []
    for m in samples_manifests:
        for r in read_jsonl_records(m):
            r["_path"] = resolve_record_path(m, r["path"])
            r["_manifest"] = str(m)
            rows.append(r)
    buckets = {
        "internal_hold": [r for r in rows if r.get("kind") in ("internal", "oracle_hold") and r.get("cut_confidence") != "low"],
        "eot": [r for r in rows if int(r["label"]) == 1 and r.get("cut_confidence") != "low"],
        "whole": [r for r in rows if r.get("kind") == "whole"],
        "low_confidence": [r for r in rows if r.get("cut_confidence") == "low"],
        "medium_confidence": [r for r in rows if r.get("cut_confidence") == "medium"],
    }
    chosen = []
    for name, n in plan.items():
        pool = buckets.get(name, [])
        chosen += [(name, r) for r in rng.sample(pool, min(n, len(pool)))]
    rng.shuffle(chosen)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    key, sheet, proxy = [], [], Counter()
    tail_probs = []
    for i, (bucket, r) in enumerate(chosen):
        name = f"{i:03d}.wav"
        shutil.copyfile(r["_path"], audio_dir / name)
        x = _read(r["_path"])
        probs = vad.probabilities(x)
        tail = probs[-5:] if len(probs) >= 5 else probs  # 160 ms before the cut
        p_tail = float(np.max(tail)) if len(tail) else float("nan")
        tail_probs.append((bucket, p_tail))
        proxy[(bucket, "speech_at_cut" if p_tail > 0.5 else "quiet_at_cut")] += 1
        key.append({"file": name, "bucket": bucket, "id": r["id"], "label": int(r["label"]), "kind": r.get("kind"), "cut_confidence": r.get("cut_confidence"), "manifest": r["_manifest"], "vad_p_speech_last_160ms": round(p_tail, 3)})
        sheet.append({"file": name, "cut_ok": None, "sounds_complete": None, "notes": ""})
    (out_dir / "KEY_do_not_open_before_listening.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
    (out_dir / "listening_audit.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sheet))
    (out_dir / "README.txt").write_text(
        "Blind listening audit. Play audio/NNN.wav in order; for each file set cut_ok (true if the cut is "
        "not inside a word) and sounds_complete (true if the speaker sounds finished) in listening_audit.jsonl. "
        "Only then open the KEY file and run `eot-qa listen-score`.\n"
    )
    by_bucket = defaultdict(list)
    for b, p in tail_probs:
        by_bucket[b].append(p)
    return {
        "n_files": len(chosen), "plan": plan, "dir": str(out_dir),
        "proxy_vad_speech_at_cut_rate": {b: round(float(np.mean(np.array(v) > 0.5)), 3) for b, v in by_bucket.items()},
        "proxy_vad_p_speech_last_160ms": {b: _pct(v) for b, v in by_bucket.items()},
        "note": "proxy only; the human verdicts in listening_audit.jsonl are the check",
    }


def listen_score(audit_dir: Path) -> dict:
    key = {k["file"]: k for k in read_jsonl_records(audit_dir / "KEY_do_not_open_before_listening.jsonl")}
    sheet = read_jsonl_records(audit_dir / "listening_audit.jsonl")
    done = [s for s in sheet if s.get("cut_ok") is not None]
    out: dict = {"n_total": len(sheet), "n_answered": len(done), "by_bucket": {}}
    for bucket in sorted({k["bucket"] for k in key.values()}):
        rows = [s for s in done if key[s["file"]]["bucket"] == bucket]
        if not rows:
            continue
        cut_ok = np.mean([bool(s["cut_ok"]) for s in rows])
        complete = [s for s in rows if s.get("sounds_complete") is not None]
        agree = np.mean([bool(s["sounds_complete"]) == (key[s["file"]]["label"] == 1) for s in complete]) if complete else float("nan")
        out["by_bucket"][bucket] = {"n": len(rows), "cut_ok_rate": round(float(cut_ok), 3), "label_agreement": round(float(agree), 3)}
    return out


# ---------------------------------------------------------------------------
# e. distribution
# ---------------------------------------------------------------------------


def check_distribution(samples_manifest: Path, clips_manifest: Path | None = None) -> dict:
    rows = read_jsonl_records(samples_manifest)
    labels = Counter(int(r["label"]) for r in rows)
    kinds = Counter(str(r.get("kind")) for r in rows)
    sources = Counter(str(r.get("source")) for r in rows)
    n = max(1, len(rows))
    out = {
        "n_samples": len(rows), "labels": dict(labels), "hold_to_eot_ratio": round(labels[0] / max(1, labels[1]), 2),
        "kinds": dict(kinds), "cut_confidence": dict(Counter(str(r.get("cut_confidence")) for r in rows)),
        "detector": dict(Counter(str(r.get("detector")) for r in rows)),
        "source_share": {k: round(v / n, 4) for k, v in sources.most_common()},
        "max_source_share": round(max(sources.values()) / n, 4) if sources else None,
        "time_to_onset_s_hold": _pct([r.get("time_to_onset") for r in rows if int(r["label"]) == 0 and r.get("time_to_onset") is not None]),
        "snr_est_db": _pct([r.get("snr_est_db") for r in rows]),
        "stored_duration_s": _pct([r.get("stored_duration_s") for r in rows if r.get("stored_duration_s") is not None]),
    }
    if clips_manifest is not None:
        clips = read_jsonl_records(clips_manifest)
        with_internal = {r["clip_id"] for r in rows if r.get("kind") == "internal"}
        out["clips"] = len(clips)
        out["clips_without_internal_pause_frac"] = round(1 - len(with_internal) / max(1, len(clips)), 4)
    if any("accent" in r for r in rows):
        out["accents"] = dict(Counter(str(r.get("accent")) for r in rows))
        out["backchannel_holds"] = sum(int(r.get("backchannel") or 0) for r in rows)
        out["disfluent_before_pause"] = sum(int(r.get("disfluent_before_pause") or 0) for r in rows)
        out["eot_gap_s"] = _pct([r.get("eot_gap") for r in rows if r.get("eot_gap") is not None])
        out["digital_silence_fraction"] = _pct([r.get("digital_silence_fraction") for r in rows])
        out["speakers"] = len({r.get("speaker_id") for r in rows})
    out["passed"] = {"max_source_share_lt_0.5": (out["max_source_share"] or 0) < 0.5}
    if "clips_without_internal_pause_frac" in out:
        out["passed"]["clips_without_internal_lt_0.4"] = out["clips_without_internal_pause_frac"] < 0.4
    return out


# ---------------------------------------------------------------------------
# f. Krisp trailing silence vs human annotation
# ---------------------------------------------------------------------------


def check_krisp_tail(krisp_clips: Path, detectors: dict[str, PauseDetector], n_clips: int = 400, seed: int = 0) -> dict:
    rows = read_jsonl_records(krisp_clips)
    rows = random.Random(seed).sample(rows, min(n_clips, len(rows)))
    errs = {name: [] for name in detectors}
    missing = Counter()
    for r in rows:
        x = _read(resolve_record_path(krisp_clips, r["path"]))
        total = len(x) / SAMPLE_RATE
        ref = float(r["last_silence_duration"])
        for name, det in detectors.items():
            pauses = [p for p in det(x) if p.confidence != "low"]
            trailing = [p for p in pauses if abs(p.end - total) < 0.05]
            if not trailing:
                missing[name] += 1
                continue
            errs[name].append((total - trailing[-1].start) - ref)
    return {
        "n_clips": len(rows),
        "signed_error_s (ours - human)": {name: _pct(v) for name, v in errs.items()},
        "abs_error_s": {name: _pct(np.abs(v)) for name, v in errs.items()},
        "no_trailing_pause_found": dict(missing),
        "within_100ms": {name: round(float(np.mean(np.abs(v) <= 0.1)), 3) if v else None for name, v in errs.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--smart-turn-clips", type=Path, default=None)
    run.add_argument("--mined", type=Path, nargs="*", default=[], help="samples.jsonl manifests to profile")
    run.add_argument("--apptek", type=Path, default=None, help="eot-apptek output dir")
    run.add_argument("--apptek-root", type=Path, default=None)
    run.add_argument("--krisp-clips", type=Path, default=None)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--pstn-clips", type=int, default=300)
    run.add_argument("--krisp-n", type=int, default=400)
    run.add_argument("--listen-manifests", type=Path, nargs="*", default=None, help="defaults to --mined")
    run.add_argument("--skip", nargs="*", default=[], choices=["boundaries", "pstn", "single_vs_dual", "listening", "distribution", "krisp_tail"])
    ls = sub.add_parser("listen-score")
    ls.add_argument("--audit-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "listen-score":
        print(json.dumps(listen_score(args.audit_dir), indent=2))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    vad = SileroVAD(threads=4)
    detectors = {"legacy": PauseDetector("legacy"), "energy": PauseDetector("energy"), "ensemble": PauseDetector("ensemble", vad=vad)}
    report: dict = {}
    if (args.out / "report.json").exists() and args.skip:
        report = json.loads((args.out / "report.json").read_text())  # keep skipped sections from the previous run
    report["generated_unix"] = time.time()
    timings = dict(report.get("timings_s", {}))

    def step(name, fn):
        if name in args.skip:
            return
        t = time.time()
        report[name] = fn()
        timings[name] = round(time.time() - t, 1)
        print(f"[{name}] done in {timings[name]}s", flush=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    if args.apptek and args.apptek_root:
        step("boundaries", lambda: check_boundaries(args.apptek, args.apptek_root))
        step("single_vs_dual", lambda: check_single_vs_dual(args.apptek))
    if args.smart_turn_clips:
        step("pstn", lambda: check_pstn(args.smart_turn_clips, detectors, n_clips=args.pstn_clips))
    if args.mined:
        step("distribution", lambda: {str(m): check_distribution(m, args.smart_turn_clips if "smart" in str(m) else None) for m in args.mined})
    if args.apptek:
        for split in ("train", "heldout"):
            m = args.apptek / split / "samples.jsonl"
            if m.exists():
                step(f"distribution_apptek_{split}", lambda m=m: check_distribution(m))
    listen_manifests = args.listen_manifests if args.listen_manifests is not None else list(args.mined)
    if listen_manifests:
        plan = {"internal_hold": 50, "eot": 50, "whole": 25, "low_confidence": 25, "medium_confidence": 25}
        step("listening", lambda: prepare_listening(listen_manifests, args.out / "listening_audit", plan, vad))
    if args.krisp_clips:
        step("krisp_tail", lambda: check_krisp_tail(args.krisp_clips, detectors, n_clips=args.krisp_n))
    report["timings_s"] = timings
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("distribution",)}, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
