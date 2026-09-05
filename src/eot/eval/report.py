"""Experiment plumbing: combine manifests, score held-out AppTek, summarise all runs.

    eot-report combine  --inputs A/samples.jsonl B/samples.jsonl --caps 0 40000 --out data/mined/r3/samples.jsonl
    eot-report heldout  --samples data/mined/apptek-oracle/heldout/samples.jsonl --checkpoint runs/r2/model.pt --out eval/heldout/r2.json
    eot-report summarize --eotbench-dir eval/eotbench/<span_set>/en --krisp-dir eval/krisp --heldout-dir eval/heldout --out eval/summary

``combine`` caps a source by *stratified* random sampling (label ratio preserved, fixed seed) and
rewrites paths relative to the new manifest so training never depends on the working directory.

``heldout`` scores each held-out sample twice: with the training-time room tone (``noise_fill``) and
raw (digital silence between utterances). A model that is much better on raw audio is using the
'exact zero = pause' shortcut the plan warned about (fissure 2).
"""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eot.audio import SAMPLE_RATE, add_room_tone, load_wav
from eot.eval.metrics import roc_auc
from eot.io import atomic_write_json, atomic_write_jsonl, read_jsonl_records, resolve_record_path


def combine(inputs: list[Path], caps: list[int], out: Path, seed: int = 0) -> dict:
    rng = random.Random(seed)
    rows_out, report = [], {"inputs": [], "seed": seed}
    seen = set()
    for manifest, cap in zip(inputs, caps):
        rows = read_jsonl_records(manifest)
        n_before = len(rows)
        if cap and cap < len(rows):
            by_label = defaultdict(list)
            for r in rows:
                by_label[int(r["label"])].append(r)
            frac = cap / len(rows)
            rows = []
            for label, group in sorted(by_label.items()):
                k = int(round(frac * len(group)))
                rows += rng.sample(group, k)
        for r in rows:
            src = resolve_record_path(manifest, r["path"])
            r["path"] = os.path.relpath(src, start=out.parent.resolve())
            if r["id"] in seen:
                r["id"] = f"{manifest.parent.name}:{r['id']}"
            seen.add(r["id"])
            rows_out.append(r)
        report["inputs"].append({"manifest": str(manifest), "rows": n_before, "kept": len(rows), "cap": cap,
                                 "labels_kept": dict(Counter(int(r["label"]) for r in rows))})
    rng.shuffle(rows_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(out, rows_out)
    report["total"] = len(rows_out)
    report["labels"] = dict(Counter(int(r["label"]) for r in rows_out))
    report["sources"] = dict(Counter(str(r.get("source")) for r in rows_out))
    atomic_write_json(out.with_name("combine_report.json"), report)
    return report


def _auc(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(roc_auc(y, p)) if len(set(y.tolist())) == 2 else float("nan")


def score_heldout(samples: Path, adapter, batch_size: int = 32, noise_fill: bool = True, seed: int = 0, score_point_only: bool = True) -> list[dict]:
    rows = read_jsonl_records(samples)
    if score_point_only:
        rows = [r for r in rows if abs(float(r["cut_time"]) - float(r["pause_start"]) - 0.2) < 1e-3]
    out, pending, meta = [], [], []

    def flush():
        for p, m in zip(adapter.predict_batch(pending), meta):
            out.append({**m, "p_eot": float(p)})
        pending.clear()
        meta.clear()

    for i, r in enumerate(rows):
        x = load_wav(resolve_record_path(samples, r["path"]))
        if noise_fill and int(r.get("noise_fill", 0)):
            x = add_room_tone(x, np.random.default_rng(seed * 1_000_003 + i))
        pending.append({"audio": {"array": x, "sampling_rate": SAMPLE_RATE}, "messages": []})
        meta.append({k: r.get(k) for k in ("id", "label", "kind", "accent", "speaker_id", "cut_confidence", "cut_time", "pause_start", "backchannel", "eot_gap", "disfluent_before_pause", "clip_id")})
        if len(pending) >= batch_size:
            flush()
    if pending:
        flush()
    return out


def summarise_heldout(scored: list[dict], score_point_only: bool = True) -> dict:
    rows = scored
    if score_point_only:
        rows = [r for r in scored if abs((r["cut_time"] or 0) - (r["pause_start"] or 0) - 0.2) < 1e-3] or scored
    y = [int(r["label"]) for r in rows]
    p = [r["p_eot"] for r in rows]
    out = {"n": len(rows), "auc": _auc(y, p), "pos_rate": float(np.mean(y)), "by_accent": {}, "by_confidence": {}}
    holds = [r for r in rows if int(r["label"]) == 0]
    eots = [r for r in rows if int(r["label"]) == 1]
    for thr in (0.3, 0.5, 0.7):
        out[f"fc_rate@{thr}"] = float(np.mean([r["p_eot"] >= thr for r in holds])) if holds else float("nan")
        out[f"eot_detect@{thr}"] = float(np.mean([r["p_eot"] >= thr for r in eots])) if eots else float("nan")
    for key, field in (("by_accent", "accent"), ("by_confidence", "cut_confidence")):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get(field))].append(r)
        for g, rs in sorted(groups.items()):
            out[key][g] = {"n": len(rs), "auc": _auc([int(r["label"]) for r in rs], [r["p_eot"] for r in rs])}
    bc = [r for r in holds if int(r.get("backchannel") or 0)]
    out["backchannel_holds_p_eot_mean"] = float(np.mean([r["p_eot"] for r in bc])) if bc else None
    return out


def heldout(samples: Path, adapter, out: Path, seed: int = 0) -> dict:
    filled = score_heldout(samples, adapter, noise_fill=True, seed=seed)
    raw = score_heldout(samples, adapter, noise_fill=False, seed=seed)
    report = {
        "samples": str(samples), "adapter_id": adapter.adapter_id,
        "filled": summarise_heldout(filled), "raw": summarise_heldout(raw),
        "evaluation_scope": "score_point_0.2s",
    }
    report["shortcut_auc_gain_raw_minus_filled"] = report["raw"]["auc"] - report["filled"]["auc"]
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)
    atomic_write_jsonl(out.with_suffix(".scores.jsonl"), filled)
    return report


def _fmt(v, pct=False, ms=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if pct:
        return f"{100 * v:.1f} %"
    if ms:
        return f"{1000 * v:.0f} ms"
    return f"{v:.3f}"


def summarize(eotbench_dir: Path | None, krisp_dir: Path | None, heldout_dir: Path | None, out: Path, names: dict[str, str] | None = None) -> str:
    names = names or {}
    lines = ["# Resumen de resultados (generado por `eot-report summarize`)", ""]
    data: dict = {}
    if eotbench_dir and eotbench_dir.exists():
        lines += ["## EoT Bench EN (harness oficial, score point 0.2 s, hold spans 0.2–5 s)", "",
                  "| Modelo | FC @300 ms | FC @600 ms | Delay @5 % FC | Delay @10 % FC | AUC | n turnos |", "|---|---:|---:|---:|---:|---:|---:|"]
        rows = []
        for run in sorted(p for p in eotbench_dir.iterdir() if (p / "metrics" / "summary.json").exists()):
            s = json.loads((run / "metrics" / "summary.json").read_text())
            man = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
            name = names.get(run.name) or man.get("display_name") or man.get("adapter_id") or run.name
            bs = json.loads((run / "metrics" / "bootstrap.json").read_text()) if (run / "metrics" / "bootstrap.json").exists() else None
            best_at = {}
            try:
                import pyarrow.parquet as pq

                tr = pq.read_table(run / "metrics" / "tradeoff.parquet").to_pandas()
                model = tr[tr["policy_type"] == "model"] if "policy_type" in tr else tr
                for budget in (0.3, 0.6):
                    feas = model[model["mean_latency"] <= budget + 1e-9]
                    best_at[budget] = float(feas["cutoff_rate"].min()) if len(feas) else None
            except Exception:
                best_at = {0.3: None, 0.6: None}
            op5, op10 = s["operating_points"].get("5pct"), s["operating_points"].get("10pct")
            entry = {"name": name, "fc_300": best_at.get(0.3), "fc_600": best_at.get(0.6),
                     "lat_5": op5["mean_latency"] if op5 else None, "lat_10": op10["mean_latency"] if op10 else None,
                     "auc": s.get("auc"), "n_eot": s.get("n_eot_spans"), "bootstrap": bs}
            rows.append(entry)
            data.setdefault("eotbench", {})[run.name] = entry
        rows.sort(key=lambda r: (r["fc_300"] is None, r["fc_300"] if r["fc_300"] is not None else 9))
        for r in rows:
            ci = ""
            if r["bootstrap"] and "latency_at_5pct" in r["bootstrap"]["points"]:
                lo, hi = r["bootstrap"]["points"]["latency_at_5pct"]["mean_latency_ci95"]
                ci = f" [{1000 * lo:.0f}–{1000 * hi:.0f}]"
            lines.append(f"| {r['name']} | {_fmt(r['fc_300'], pct=True)} | {_fmt(r['fc_600'], pct=True)} | {_fmt(r['lat_5'], ms=True)}{ci} | {_fmt(r['lat_10'], ms=True)} | {_fmt(r['auc'])} | {r['n_eot']} |")
        lines.append("")
    if krisp_dir and krisp_dir.exists():
        lines += ["## Krisp Turn-Taking Test v1 (una decisión por clip, bootstrap por speaker)", "",
                  "| Modelo | AUC @0.2 s | FC @300 ms | FC @600 ms | Delay @5 % FC | Delay @10 % FC | n hold / n shift |", "|---|---:|---:|---:|---:|---:|---:|"]
        for f in sorted(krisp_dir.glob("*.metrics.json")):
            m = json.loads(f.read_text())
            ops = m["operating_points"]
            name = names.get(f.stem.replace(".metrics", ""), f.stem.replace(".metrics", ""))

            def g(k, field):
                return ops[k][field] if ops.get(k) else None

            ci = m.get("bootstrap_by_speaker", {}).get("best_latency_at_5pct")
            ci_txt = f" [{1000 * ci['mean_latency_ci95'][0]:.0f}–{1000 * ci['mean_latency_ci95'][1]:.0f}]" if ci else ""
            lines.append(f"| {name} | {_fmt(m['auc_at_0.2s'])} | {_fmt(g('best_cutoff_at_0.3s', 'cutoff_rate'), pct=True)} | {_fmt(g('best_cutoff_at_0.6s', 'cutoff_rate'), pct=True)} | {_fmt(g('best_latency_at_5pct', 'mean_latency'), ms=True)}{ci_txt} | {_fmt(g('best_latency_at_10pct', 'mean_latency'), ms=True)} | {m['n_hold']} / {m['n_eot']} |")
            data.setdefault("krisp", {})[name] = m
        lines.append("")
    if heldout_dir and heldout_dir.exists():
        lines += ["## AppTek held-out (en-IN, en-SG, en-GB_SCT; corte a 0.2 s; relleno de room tone vs crudo)", "",
                  "| Modelo | AUC relleno | AUC crudo | Δ atajo (crudo − relleno) | FC@0.5 relleno | Detect@0.5 relleno | n |", "|---|---:|---:|---:|---:|---:|---:|"]
        for f in sorted(heldout_dir.glob("*.json")):
            if f.name.endswith(".scores.jsonl"):
                continue
            h = json.loads(f.read_text())
            name = names.get(f.stem, f.stem)
            lines.append(f"| {name} | {_fmt(h['filled']['auc'])} | {_fmt(h['raw']['auc'])} | {h['shortcut_auc_gain_raw_minus_filled']:+.3f} | {_fmt(h['filled']['fc_rate@0.5'], pct=True)} | {_fmt(h['filled']['eot_detect@0.5'], pct=True)} | {h['filled']['n']} |")
            data.setdefault("heldout", {})[name] = h
        lines.append("")
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(data, indent=2, default=str))
    return text


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--inputs", type=Path, nargs="+", required=True)
    c.add_argument("--caps", type=int, nargs="+", required=True, help="0 = no cap; one per input")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--seed", type=int, default=0)
    h = sub.add_parser("heldout")
    h.add_argument("--samples", type=Path, required=True)
    h.add_argument("--checkpoint", default=None)
    h.add_argument("--onnx", default=None)
    h.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--eotbench-dir", type=Path, default=None)
    s.add_argument("--krisp-dir", type=Path, default=None)
    s.add_argument("--heldout-dir", type=Path, default=None)
    s.add_argument("--names", type=Path, default=None, help="json mapping run dir / file stem -> display name")
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "combine":
        if len(args.caps) != len(args.inputs):
            raise SystemExit("--caps must have one entry per input")
        print(json.dumps(combine(args.inputs, args.caps, args.out, args.seed), indent=2))
    elif args.cmd == "heldout":
        from eot.eval.eotbench import EOTAdapter

        adapter = EOTAdapter(checkpoint=args.checkpoint, onnx=args.onnx)
        rep = heldout(args.samples, adapter, args.out)
        print(json.dumps({k: v for k, v in rep.items() if k in ("filled", "raw", "shortcut_auc_gain_raw_minus_filled")}, indent=2, default=str))
    else:
        names = json.loads(args.names.read_text()) if args.names else None
        print(summarize(args.eotbench_dir, args.krisp_dir, args.heldout_dir, args.out, names))


if __name__ == "__main__":
    main()
