"""Pinned public quality readouts of real container runtimes, separate from timing.

pack/metrics run on the host. infer runs in the respective inference image.
No labels are used by inference; no calibration or training occurs.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pack(out):
    from eot_harness.io import iter_batches, load_hf_dataset

    from eot.audio import decode_payload, last_window

    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Pack output must be empty; preserve existing benchmark inputs")
    ds = load_hf_dataset(
        "livekit/eot-bench-data",
        "en",
        "validation",
        revision="ca9d98a9686b920a2d8c9eb984224ba9be74e4dd",
    )
    assert len(ds) == 400
    out.mkdir(parents=True, exist_ok=True)
    waves = np.lib.format.open_memmap(out / "waves.npy", mode="w+", dtype=np.float32, shape=(11191, 128000))
    rows = []
    offset = 0
    for meta, batch in iter_batches(ds, batch_size=32, max_audio_sec=8):
        for example in batch:
            waves[offset] = last_window(decode_payload(example["audio"]), 8)
            offset += 1
        rows.extend(meta)
    assert offset == 11191
    waves.flush()
    (out / "metadata.json").write_text(json.dumps(rows))
    (out / "identity.json").write_text(
        json.dumps(
            {
                "rows": offset,
                "turns": 400,
                "dataset_revision": "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd",
                "harness_commit": "6594d8b3b8af385b15f116dde310ce45af92d646",
                "max_audio_sec": 8,
                "inference_interval": 0.1,
                "min_silence_span": 0.1,
                "score_point": 0.2,
                "waves_sha256": sha(out / "waves.npy"),
                "metadata_sha256": sha(out / "metadata.json"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Frozen {offset} causal public prefixes", flush=True)


def infer(kind, model, inputs, out, batches):
    identity = json.loads((inputs / "identity.json").read_text())
    if sha(inputs / "waves.npy") != identity["waves_sha256"]:
        raise ValueError("Input pack changed")
    waves = np.load(inputs / "waves.npy", mmap_mode="r")
    if kind == "gpu":
        from eot.serving.tensorrt_engine import TensorRTEngine

        engine = TensorRTEngine(model)
    elif kind == "diagnostic":
        import torch

        from eot.modeling.cohere_filterbank import CohereFilterbank
        from eot.serving.tensorrt_engine import PlanRunner

        torch.set_num_threads(2)
        front = CohereFilterbank.from_payload(torch.load(model.parent / "frontend.pt", map_location="cpu", weights_only=True)).eval()
        runner = PlanRunner(model)

        class Diagnostic:
            def infer(self, x):
                with torch.inference_mode():
                    f = front(torch.from_numpy(np.stack(x)[:, -64000:])).numpy()
                return runner.infer(f), 0, 0

        engine = Diagnostic()
    else:
        from eot.serving.service import Engine

        engine = Engine(str(model), threads=2)
    out.mkdir(parents=True, exist_ok=True)
    for batch in batches:
        path = out / f"probabilities-b{batch}.npy"
        if path.exists():
            raise FileExistsError("Preserve prior run instead of overwriting")
        values = []
        started = time.perf_counter()
        for i in range(0, len(waves), batch):
            if kind == "cpu":
                values.extend(engine.infer(w, "")[0] for w in waves[i : i + batch])
            else:
                values.extend(engine.infer(list(waves[i : i + batch]))[0].tolist())
            if i % 1024 == 0:
                print(f"batch={batch} {i}/{len(waves)}", flush=True)
        values = np.asarray(values, np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite scores")
        np.save(path, values)
        (out / f"inference-b{batch}.json").write_text(
            json.dumps(
                {
                    "model_sha256": sha(model),
                    "input_identity": identity,
                    "kind": kind,
                    "batch": batch,
                    "examples": len(values),
                    "seconds": time.perf_counter() - started,
                    "scores_sha256": sha(path),
                    "numerical_diagnostic": kind == "diagnostic",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Completed batch {batch}", flush=True)


def metrics(inputs, out):
    import pandas as pd
    from eot_harness.metrics import compute_metrics_from_predictions

    metadata = json.loads((inputs / "metadata.json").read_text())
    for path in sorted(out.glob("probabilities-b*.npy")):
        name = path.stem.removeprefix("probabilities-")
        rows = pd.DataFrame(metadata)
        rows["p_eot"] = np.load(path)
        rows.to_parquet(out / f"predictions-{name}.parquet", index=False)
        tradeoff, summary = compute_metrics_from_predictions(rows, score_point_s=0.2)
        tradeoff.to_csv(out / f"tradeoff-{name}.csv", index=False)
        (out / f"summary-{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(name, summary["operating_points"]["5pct"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["pack", "infer", "metrics"])
    p.add_argument("--kind", choices=["cpu", "gpu", "diagnostic"])
    p.add_argument("--model", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    a = p.parse_args()
    if any(batch < 1 or batch > 8 for batch in a.batches):
        p.error("Batches must be between 1 and 8")
    if a.stage != "pack" and a.inputs is None:
        p.error("infer/metrics require --inputs")
    if a.stage == "infer" and (a.kind is None or a.model is None):
        p.error("infer requires --kind and --model")
    if a.stage == "pack":
        pack(a.out)
    elif a.stage == "infer":
        infer(a.kind, a.model, a.inputs, a.out, a.batches)
    else:
        metrics(a.inputs, a.out)
