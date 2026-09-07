"""Qualify the actual engine, including original CPU frontend and serving batches."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eot.io import sha256_file
from eot.modeling.cohere_filterbank import CohereFilterbank
from eot.serving.tensorrt_engine import PlanRunner


def verify(path, source_dir, tolerance):
    source = json.loads((source_dir / "source.json").read_text())
    build = json.loads(path.with_suffix(".build.json").read_text())
    if sha256_file(path) != build["engine_sha256"]:
        raise ValueError("Engine changed")
    panel = np.load(source_dir / "parity-inputs.npz")
    front = CohereFilterbank.from_payload(torch.load(source_dir / "frontend.pt", map_location="cpu", weights_only=True)).eval()
    torch.set_num_threads(2)
    started = time.perf_counter()
    runner = PlanRunner(path)
    cold = time.perf_counter() - started
    values = {}
    differences = {}
    frontend_diffs = {}
    timings = {}
    for batch in (1, 4, 8):
        probs = []
        feats = []
        for i in range(0, len(panel["waves"]), batch):
            with torch.inference_mode():
                f = front(torch.from_numpy(panel["waves"][i : i + batch])).numpy()
            feats.append(f)
            probs.extend(runner.infer(f).tolist())
        values[str(batch)] = probs
        differences[str(batch)] = float(np.max(np.abs(np.array(probs) - panel["reference"])))
        frontend_diffs[str(batch)] = float(np.max(np.abs(np.concatenate(feats) - panel["features"])))
        f = panel["features"][:batch]
        samples = []
        for _ in range(5):
            runner.infer(f)
        for _ in range(50):
            tick = time.perf_counter()
            runner.infer(f)
            samples.append((time.perf_counter() - tick) * 1000)
        timings[str(batch)] = {
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
        }
    batch_diff = max(float(np.max(np.abs(np.array(p) - values["1"]))) for p in values.values())
    accepted = max(differences.values()) <= tolerance and batch_diff <= tolerance
    inspector = runner.engine.create_engine_inspector()
    # One fixed context makes dynamic dimensions readable in the detailed engine report.
    inspector.execution_context = runner.batches[1]["context"]
    path.with_suffix(".layers.json").write_text(inspector.get_engine_information(runner.trt.LayerInformationFormat.JSON))
    result = {
        "accepted": accepted,
        "sha256": sha256_file(path),
        "model_name": source["model_name"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "frontend_sha256": source["frontend_sha256"],
        "checks_sha256": source["checks_sha256"],
        "precision": build["precision"],
        "tensorrt": build["tensorrt"],
        "gpu": build["gpu"],
        "reference": source["reference"],
        "probability_tolerance": tolerance,
        "max_probability_difference_by_batch": differences,
        "max_batch_probability_difference": batch_diff,
        "frontend_feature_difference_by_batch": frontend_diffs,
        "probabilities_by_batch": values,
        "source_probabilities": panel["reference"].tolist(),
        "examples": len(panel["reference"]),
        "engine_load_and_capture_seconds": cold,
        "kernel_with_transfers_ms": timings,
        "cuda_graphs": True,
        "public_quality_status": "Pending; source historical score is not this runtime readout",
        "historical_source_mean_delay_at_5pct_ms": source["historical_source_mean_delay_at_5pct_ms"],
    }
    path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    runner.close()
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["probabilities_by_batch", "source_probabilities"]}),
        flush=True,
    )
    return accepted


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--tolerance", type=float, required=True)
    a = p.parse_args()
    if not np.isfinite(a.tolerance) or a.tolerance < 0:
        p.error("Tolerance must be finite and nonnegative")
    if a.engine.with_suffix(".json").exists():
        p.error("Numerical verdict already exists; preserve it and use a new output")
    if not verify(a.engine, a.source_dir, a.tolerance):
        raise SystemExit(3)
