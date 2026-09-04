"""LiveKit eot-bench batch adapter + a minimal local scorer with the same schema.

Adapter contract (from ``livekit/eot-bench`` README):

    class MyAdapter:
        adapter_id = "my-model"
        score_point = 0.2
        def predict_batch(self, batch): ...   # batch items: {"audio", "messages"} -> [p_eot]

Run through the official harness (apples-to-apples with the public leaderboard):

    pip install -e /path/to/eot-bench
    EOT_CHECKPOINT=runs/base/model.pt eot-harness predict \
        --path livekit/eot-bench-data --name en --split validation \
        --adapter eot.eotbench_adapter:EOTAdapter

Or, without installing the harness, score any dataset in the public schema locally and feed the
rows to ``eot-sweep`` for quick diagnostics only. Those local aggregate metrics are not an
official EoT Bench result:

    uv run python -m eot.eotbench_adapter --checkpoint runs/base/model.pt --lang en --out preds.jsonl

``messages`` is used only when the checkpoint was trained with ``--use-context``: the last
assistant message is hashed exactly like at training time. LiveKit's own v1 adapter sends audio
only, so a context-aware model is a legitimate, unexploited lever on this benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, log_mel, resample, to_float32, to_mono
from .context import CTX_LEN, hash_context
from .policy import GRID_STEP


def _audio_to_16k(a) -> np.ndarray:
    if isinstance(a, dict):
        if a.get("array") is not None:
            return resample(to_mono(to_float32(np.asarray(a["array"]))), int(a["sampling_rate"]))
        if a.get("bytes") is not None:
            import io

            import soundfile as sf

            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
            return resample(to_mono(np.asarray(x)), int(sr))
        if a.get("path"):
            import soundfile as sf

            x, sr = sf.read(a["path"], dtype="float32", always_2d=False)
            return resample(to_mono(np.asarray(x)), int(sr))
    if isinstance(a, tuple) and len(a) == 2:
        x, sr = a
        return resample(to_mono(to_float32(np.asarray(x))), int(sr))
    return to_mono(to_float32(np.asarray(a)))


def last_assistant_text(messages) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") in ("assistant", "agent", "system_agent") and m.get("content"):
            return str(m["content"])
    return ""


class EOTAdapter:
    """Batch adapter; backend is a torch checkpoint (``EOT_CHECKPOINT``) or an ONNX file (``EOT_ONNX``)."""

    adapter_id = "whisper-tiny-prefix-mined"
    score_point = 0.2
    max_audio_sec = 8.0

    def __init__(self, checkpoint: str | None = None, onnx: str | None = None, device: str | None = None):
        checkpoint = checkpoint or os.environ.get("EOT_CHECKPOINT")
        onnx = onnx or os.environ.get("EOT_ONNX")
        self.use_context = False
        if onnx:
            import onnxruntime as ort

            self.sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
            self.input_names = {item.name for item in self.sess.get_inputs()}
            self.backend = "onnx"
            self.model_sha = hashlib.sha256(Path(onnx).read_bytes()).hexdigest()
            meta = Path(onnx).with_suffix(".json")
            if meta.exists():
                self.use_context = bool(json.loads(meta.read_text()).get("use_context", False))
        elif checkpoint:
            import torch

            from .model import load_checkpoint
            from .train import pick_device

            self.device = pick_device(device)
            self.model = load_checkpoint(checkpoint).to(self.device)
            self.use_context = self.model.cfg.use_context
            self.backend = "torch"
            self._torch = torch
            self.model_sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        else:
            raise RuntimeError("Provide EOT_CHECKPOINT or EOT_ONNX")
        self.adapter_id = (
            f"{self.adapter_id}-{'ctx' if self.use_context else 'audio'}-{self.model_sha[:12]}"
        )

    def predict_batch(self, batch) -> list[float]:
        feats = np.stack([log_mel(_audio_to_16k(item["audio"])) for item in batch])
        ctx = np.stack([hash_context(last_assistant_text(item.get("messages"))) if self.use_context else np.zeros(CTX_LEN, np.int64) for item in batch])
        if self.backend == "onnx":
            feed = {"input_features": feats.astype(np.float32)}
            if "context_ids" in self.input_names:
                feed["context_ids"] = ctx
            p, _ = self.sess.run(None, feed)
            return [float(v) for v in p]
        t = self._torch
        with t.no_grad():
            out = self.model(t.from_numpy(feats).to(self.device), t.from_numpy(ctx).to(self.device))
        return [float(v) for v in out["p_eot"].cpu().numpy()]


# ---------------------------------------------------------------------------
# Local scorer over the public schema (id, audio, silence_spans, messages)
# ---------------------------------------------------------------------------


def score_rows(adapter: EOTAdapter, rows, min_silence: float = 0.1, grid_step: float = GRID_STEP, batch_size: int = 16):
    """Yield eot-bench-style prediction rows for dataset rows in the public schema."""
    pending, meta = [], []

    def flush():
        if not pending:
            return
        for p, m in zip(adapter.predict_batch(pending), meta):
            yield {**m, "p_eot": p}
        pending.clear()
        meta.clear()

    for row in rows:
        x = _audio_to_16k(row["audio"])
        spans = [s for s in row["silence_spans"] if float(s["end"]) - float(s["start"]) >= min_silence]
        for k, s in enumerate(spans):
            label = "eot" if k == len(spans) - 1 else "hold"
            start, end = float(s["start"]), float(s["end"])
            t = start + adapter.score_point
            while t <= end + 1e-9:
                pending.append({"audio": x[: int(t * SAMPLE_RATE)], "messages": row.get("messages") or []})
                meta.append({"id": str(row["id"]), "span_index": k, "timestamp": round(t, 3), "silence_dur": round(t - start, 3), "label": label})
                if len(pending) >= batch_size:
                    yield from flush()
                t += grid_step
    yield from flush()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--dataset", default="livekit/eot-bench-data")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    from datasets import Audio, load_dataset

    ds = load_dataset(args.dataset, name=args.lang, split=args.split).cast_column("audio", Audio(decode=False))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    adapter = EOTAdapter(checkpoint=args.checkpoint, onnx=args.onnx)
    n = 0
    with args.out.open("w") as f:
        for r in score_rows(adapter, ds):
            f.write(json.dumps(r) + "\n")
            n += 1
    print(f"wrote {n} prediction rows -> {args.out}; next: uv run eot-sweep --predictions {args.out}")


if __name__ == "__main__":
    main()
