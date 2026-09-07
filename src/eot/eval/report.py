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
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from eot.audio import SAMPLE_RATE, SCORE_POINT, add_room_tone, load_wav
from eot.eval.eotbench import EOTAdapter, batched_predictions
from eot.io import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl_records,
    read_samples,
    resolve_record_path,
    seed_from_key,
)
from eot.metrics import roc_auc


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
            for _label, group in sorted(by_label.items()):
                k = int(round(frac * len(group)))
                rows += rng.sample(group, k)
        for r in rows:
            src = resolve_record_path(manifest, r["path"])
            r["path"] = os.path.relpath(src, start=out.parent.resolve())
            if r["id"] in seen:
                r["id"] = f"{manifest.parent.name}:{r['id']}"
            seen.add(r["id"])
            rows_out.append(r)
        report["inputs"].append(
            {
                "manifest": str(manifest),
                "rows": n_before,
                "kept": len(rows),
                "cap": cap,
                "labels_kept": dict(Counter(int(r["label"]) for r in rows)),
            }
        )
    rng.shuffle(rows_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(out, rows_out)
    report["total"] = len(rows_out)
    report["labels"] = dict(Counter(int(r["label"]) for r in rows_out))
    report["sources"] = dict(Counter(str(r.get("source")) for r in rows_out))
    atomic_write_json(out.with_name("combine_report.json"), report)
    return report


def room_tone_seed(seed: int, sample_id) -> int:
    """Deterministic room-tone RNG seed for one held-out sample, independent of row position."""
    return seed_from_key(f"heldout-room-tone:{int(seed)}:{sample_id}")


def _at_score_point(row: dict) -> bool:
    return abs(float(row["cut_time"] or 0) - float(row["pause_start"] or 0) - SCORE_POINT) < 1e-3


def _heldout_points(rows: list[dict], noise_fill: bool, seed: int) -> Iterator[tuple[dict, dict]]:
    keep = ("id", "label", "kind", "accent", "speaker_id", "cut_confidence", "cut_time", "pause_start")
    keep += ("backchannel", "eot_gap", "disfluent_before_pause", "clip_id")
    for i, r in enumerate(rows):
        x = load_wav(r["path"])
        if noise_fill and int(r.get("noise_fill", 0)):
            # Keyed by the sample id, not its position, so a per-sample score is reproducible
            # whatever subset of rows is scored (L007 finding: score_point_only changed the noise).
            x = add_room_tone(x, np.random.default_rng(room_tone_seed(seed, r.get("id", i))))
        yield {"audio": {"array": x, "sampling_rate": SAMPLE_RATE}, "messages": []}, {k: r.get(k) for k in keep}


def score_heldout(
    samples: Path, adapter, batch_size: int = 32, noise_fill: bool = True, seed: int = 0, score_point_only: bool = True
) -> list[dict]:
    rows = read_samples(samples)
    if score_point_only:
        rows = [r for r in rows if _at_score_point(r)]
    return list(batched_predictions(adapter, _heldout_points(rows, noise_fill, seed), batch_size))


def summarise_heldout(scored: list[dict], score_point_only: bool = True) -> dict:
    rows = [r for r in scored if _at_score_point(r)] if score_point_only else scored
    if score_point_only and not rows:
        raise ValueError(f"no rows at the score point ({SCORE_POINT} s after the pause start)")
    y = [int(r["label"]) for r in rows]
    p = [r["p_eot"] for r in rows]
    out = {
        "n": len(rows),
        "auc": roc_auc(np.asarray(y), np.asarray(p)),
        "pos_rate": float(np.mean(y)),
        "by_accent": {},
        "by_confidence": {},
    }
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
            out[key][g] = {"n": len(rs), "auc": roc_auc(np.array([int(r["label"]) for r in rs]), np.array([r["p_eot"] for r in rs]))}
    bc = [r for r in holds if int(r.get("backchannel") or 0)]
    out["backchannel_holds_p_eot_mean"] = float(np.mean([r["p_eot"] for r in bc])) if bc else None
    return out


def heldout(samples: Path, adapter, out: Path, seed: int = 0) -> dict:
    filled = score_heldout(samples, adapter, noise_fill=True, seed=seed)
    raw = score_heldout(samples, adapter, noise_fill=False, seed=seed)
    report = {
        "samples": str(samples),
        "adapter_id": adapter.adapter_id,
        "filled": summarise_heldout(filled),
        "raw": summarise_heldout(raw),
        "evaluation_scope": f"score_point_{SCORE_POINT}s",
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


def _op(ops: dict, key: str, field: str):
    """``ops[key][field]`` of a Krisp operating-point dict, or None when the budget was infeasible."""
    return ops[key][field] if ops.get(key) else None


def _best_cutoff_under(run: Path, budgets=(0.3, 0.6)) -> dict[float, float | None]:
    """Lowest model cutoff rate within each mean-latency budget, from the harness tradeoff table."""
    tradeoff = run / "metrics" / "tradeoff.parquet"
    if not tradeoff.exists():
        return dict.fromkeys(budgets)
    import pyarrow.parquet as pq

    tr = pq.read_table(tradeoff).to_pandas()
    model = tr[tr["policy_type"] == "model"] if "policy_type" in tr else tr
    out = {}
    for budget in budgets:
        feasible = model[model["mean_latency"] <= budget + 1e-9]
        out[budget] = float(feasible["cutoff_rate"].min()) if len(feasible) else None
    return out


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _eotbench_rows(eotbench_dir: Path, names: dict[str, str]) -> tuple[list[str], dict]:
    lines = [
        "## EoT Bench EN (official harness, score point 0.2 s, hold spans 0.2–5 s)",
        "",
        "| Model | FC @300 ms | FC @600 ms | Delay @5 % FC | Delay @10 % FC | AUC | n turns |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows, data = [], {}
    for run in sorted(p for p in eotbench_dir.iterdir() if (p / "metrics" / "summary.json").exists()):
        summary = _read_json(run / "metrics" / "summary.json")
        manifest = _read_json(run / "manifest.json", {})
        best_at = _best_cutoff_under(run)
        op5, op10 = summary["operating_points"].get("5pct"), summary["operating_points"].get("10pct")
        entry = {
            "name": names.get(run.name) or manifest.get("display_name") or manifest.get("adapter_id") or run.name,
            "fc_300": best_at[0.3],
            "fc_600": best_at[0.6],
            "lat_5": op5["mean_latency"] if op5 else None,
            "lat_10": op10["mean_latency"] if op10 else None,
            "auc": summary.get("auc"),
            "n_eot": summary.get("n_eot_spans"),
            "bootstrap": _read_json(run / "metrics" / "bootstrap.json"),
        }
        rows.append(entry)
        data[run.name] = entry
    rows.sort(key=lambda r: (r["fc_300"] is None, r["fc_300"] if r["fc_300"] is not None else 9))
    for r in rows:
        ci = ""
        if r["bootstrap"] and "latency_at_5pct" in r["bootstrap"]["points"]:
            lo, hi = r["bootstrap"]["points"]["latency_at_5pct"]["mean_latency_ci95"]
            ci = f" [{1000 * lo:.0f}–{1000 * hi:.0f}]"
        lines.append(
            f"| {r['name']} | {_fmt(r['fc_300'], pct=True)} | {_fmt(r['fc_600'], pct=True)} | {_fmt(r['lat_5'], ms=True)}{ci} | {_fmt(r['lat_10'], ms=True)} | {_fmt(r['auc'])} | {r['n_eot']} |"
        )
    return lines + [""], data


def _krisp_rows(krisp_dir: Path, names: dict[str, str]) -> tuple[list[str], dict]:
    lines = [
        "## Krisp Turn-Taking Test v1 (one decision per clip, bootstrap by speaker)",
        "",
        "| Model | AUC @0.2 s | FC @300 ms | FC @600 ms | Delay @5 % FC | Delay @10 % FC | n hold / n shift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    data = {}
    for f in sorted(krisp_dir.glob("*.metrics.json")):
        m = json.loads(f.read_text())
        ops = m["operating_points"]
        stem = f.name.removesuffix(".metrics.json")
        name = names.get(stem, stem)
        ci = m.get("bootstrap_by_speaker", {}).get("best_latency_at_5pct")
        ci_txt = f" [{1000 * ci['mean_latency_ci95'][0]:.0f}–{1000 * ci['mean_latency_ci95'][1]:.0f}]" if ci else ""
        lines.append(
            f"| {name} | {_fmt(m['auc_at_0.2s'])} | {_fmt(_op(ops, 'best_cutoff_at_0.3s', 'cutoff_rate'), pct=True)} | {_fmt(_op(ops, 'best_cutoff_at_0.6s', 'cutoff_rate'), pct=True)} | {_fmt(_op(ops, 'best_latency_at_5pct', 'mean_latency'), ms=True)}{ci_txt} | {_fmt(_op(ops, 'best_latency_at_10pct', 'mean_latency'), ms=True)} | {m['n_hold']} / {m['n_eot']} |"
        )
        data[name] = m
    return lines + [""], data


def _heldout_rows(heldout_dir: Path, names: dict[str, str]) -> tuple[list[str], dict]:
    lines = [
        f"## AppTek held-out (en-IN, en-SG, en-GB_SCT; cut at {SCORE_POINT} s; room-tone-filled vs raw)",
        "",
        "| Model | AUC filled | AUC raw | Δ shortcut (raw − filled) | FC@0.5 filled | Detect@0.5 filled | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    data = {}
    for f in sorted(heldout_dir.glob("*.json")):
        h = json.loads(f.read_text())
        name = names.get(f.stem, f.stem)
        lines.append(
            f"| {name} | {_fmt(h['filled']['auc'])} | {_fmt(h['raw']['auc'])} | {h['shortcut_auc_gain_raw_minus_filled']:+.3f} | {_fmt(h['filled']['fc_rate@0.5'], pct=True)} | {_fmt(h['filled']['eot_detect@0.5'], pct=True)} | {h['filled']['n']} |"
        )
        data[name] = h
    return lines + [""], data


def summarize(
    eotbench_dir: Path | None, krisp_dir: Path | None, heldout_dir: Path | None, out: Path, names: dict[str, str] | None = None
) -> str:
    """Markdown + JSON summary of every run found under the three result directories (missing ones are skipped)."""
    names = names or {}
    lines = ["# Results summary (generated by `eot-report summarize`)", ""]
    data: dict = {}
    for key, directory, section in (
        ("eotbench", eotbench_dir, _eotbench_rows),
        ("krisp", krisp_dir, _krisp_rows),
        ("heldout", heldout_dir, _heldout_rows),
    ):
        if directory and directory.exists():
            section_lines, data[key] = section(directory, names)
            lines += section_lines
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(data, indent=2, default=str))
    return text


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine", help="merge sample manifests with per-source stratified caps")
    c.add_argument("--inputs", type=Path, nargs="+", required=True, help="samples.jsonl manifests to merge")
    c.add_argument("--caps", type=int, nargs="+", required=True, help="0 = no cap; one per input")
    c.add_argument("--out", type=Path, required=True, help="combined samples.jsonl to write")
    c.add_argument("--seed", type=int, default=0, help="seed of the stratified subsampling and shuffle")
    h = sub.add_parser("heldout", help="score held-out AppTek samples with and without room tone")
    h.add_argument("--samples", type=Path, required=True, help="held-out samples.jsonl")
    h.add_argument("--checkpoint", default=None, help="torch checkpoint (default: $EOT_CHECKPOINT)")
    h.add_argument("--onnx", default=None, help="ONNX model (default: $EOT_ONNX)")
    h.add_argument("--out", type=Path, required=True, help="report json (scores go to <out>.scores.jsonl)")
    s = sub.add_parser("summarize", help="markdown + json tables over every run in the result directories")
    s.add_argument("--eotbench-dir", type=Path, default=None, help="directory of eot-harness run directories")
    s.add_argument("--krisp-dir", type=Path, default=None, help="directory of eot-krisp *.metrics.json files")
    s.add_argument("--heldout-dir", type=Path, default=None, help="directory of eot-report heldout json files")
    s.add_argument("--names", type=Path, default=None, help="json mapping run dir / file stem -> display name")
    s.add_argument("--out", type=Path, required=True, help="directory for summary.md and summary.json")
    args = ap.parse_args(argv)
    if args.cmd == "combine":
        if len(args.caps) != len(args.inputs):
            raise SystemExit("--caps must have one entry per input")
        print(json.dumps(combine(args.inputs, args.caps, args.out, args.seed), indent=2))
    elif args.cmd == "heldout":
        adapter = EOTAdapter(checkpoint=args.checkpoint, onnx=args.onnx)
        rep = heldout(args.samples, adapter, args.out)
        print(
            json.dumps(
                {k: v for k, v in rep.items() if k in ("filled", "raw", "shortcut_auc_gain_raw_minus_filled")}, indent=2, default=str
            )
        )
    else:
        names = json.loads(args.names.read_text()) if args.names else None
        print(summarize(args.eotbench_dir, args.krisp_dir, args.heldout_dir, args.out, names))


if __name__ == "__main__":
    main()
