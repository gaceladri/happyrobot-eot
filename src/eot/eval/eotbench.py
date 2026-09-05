"""LiveKit eot-bench batch adapter + a minimal local scorer with the same schema.

Adapter contract (from ``livekit/eot-bench`` README):

    class MyAdapter:
        adapter_id = "my-model"
        score_point = 0.2
        def predict_batch(self, batch): ...   # batch items: {"audio", "messages"} -> [p_eot]

Run through the official harness (apples-to-apples with the public leaderboard):

    pip install -e /path/to/eot-bench
    EOT_CHECKPOINT=runs/base/model.pt eot-harness predict \\
        --path livekit/eot-bench-data --name en --split validation \\
        --adapter eot.eval.eotbench:EOTAdapter

Or, without installing the harness, score any dataset in the public schema locally and feed the
rows to ``eot-sweep`` for quick diagnostics only. Those local aggregate metrics are not an
official EoT Bench result:

    uv run python -m eot.eval.eotbench --checkpoint runs/base/model.pt --lang en --out preds.jsonl

``messages`` is used only when the checkpoint was trained with ``--use-context``: the last
assistant message is hashed exactly like at training time. LiveKit's own v1 adapter sends audio
only, so a context-aware model is a legitimate, unexploited lever on this benchmark.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np

from eot.audio import GRID_STEP, SAMPLE_RATE, SCORE_POINT, decode_payload, log_mel
from eot.context import CTX_LEN, hash_context
from eot.io import sha256_file
from eot.onnx import cpu_session, input_names, model_feed


def last_assistant_text(messages) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") in ("assistant", "agent", "system_agent") and m.get("content"):
            return str(m["content"])
    return ""


class EOTAdapter:
    """Batch adapter; backend is a torch checkpoint (``EOT_CHECKPOINT``) or an ONNX file (``EOT_ONNX``)."""

    adapter_id = "whisper-tiny-prefix-mined"
    score_point = SCORE_POINT
    max_audio_sec = 8.0

    def __init__(self, checkpoint: str | None = None, onnx: str | None = None, device: str | None = None):
        checkpoint = checkpoint or os.environ.get("EOT_CHECKPOINT")
        onnx = onnx or os.environ.get("EOT_ONNX")
        self.use_context = False
        self.normalize_audio = False
        if onnx:
            self.sess = cpu_session(onnx, threads=int(os.environ.get("EOT_THREADS", "2")))
            self.input_names = input_names(self.sess)
            self.backend = "onnx"
            self.model_sha = sha256_file(Path(onnx))
            meta = Path(onnx).with_suffix(".json")
            info = json.loads(meta.read_text()) if meta.exists() else {}
            self.use_context = bool(info.get("use_context", False))
            self.normalize_audio = bool(info.get("normalize_audio", False))
        elif checkpoint:
            import torch

            from eot.modeling.model import load_checkpoint
            from eot.modeling.train import pick_device

            self.device = pick_device(device)
            # Match the CPU FP32 deployment graph. CUDA's default TF32 convolutions
            # otherwise introduce ~1e-3 probability drift into benchmark traces.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.model = load_checkpoint(checkpoint).to(self.device)
            self.use_context = self.model.cfg.use_context
            self.normalize_audio = self.model.cfg.normalize_audio
            self.backend = "torch"
            self._torch = torch
            self.model_sha = sha256_file(Path(checkpoint))
        else:
            raise RuntimeError("Provide EOT_CHECKPOINT or EOT_ONNX")
        self.adapter_id = (
            f"{self.adapter_id}-{'ctx' if self.use_context else 'audio'}-{self.model_sha[:12]}"
        )
        if self.backend == "torch":
            self.adapter_id += "-ieee-fp32"
        # Shown by `eot-harness compare-models`; lets R2/R2'/R3/R4 be told apart in the report.
        self.display_name = os.environ.get("EOT_DISPLAY_NAME") or self.adapter_id

    def predict_batch(self, batch) -> list[float]:
        feats = np.stack([log_mel(decode_payload(item["audio"]), normalize=self.normalize_audio) for item in batch])
        ctx = np.stack([hash_context(last_assistant_text(item.get("messages"))) if self.use_context else np.zeros(CTX_LEN, np.int64) for item in batch])
        if self.backend == "onnx":
            p, _ = self.sess.run(None, model_feed(self.input_names, feats.astype(np.float32), ctx))
            return [float(v) for v in p]
        t = self._torch
        with t.no_grad():
            out = self.model(t.from_numpy(feats).to(self.device), t.from_numpy(ctx).to(self.device))
        return [float(v) for v in out["p_eot"].cpu().numpy()]


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
        x = decode_payload(row["audio"])
        spans = row["silence_spans"]
        for k, s in enumerate(spans):
            if float(s["end"]) - float(s["start"]) < min_silence - 1e-9:
                continue
            label = "eot" if k == len(spans) - 1 else "hold"
            start, end = float(s["start"]), float(s["end"])
            t = start + adapter.score_point
            while t <= end + 1e-9:
                timestamp = round(t, 6)
                pending.append({"audio": x[: int(np.floor(timestamp * SAMPLE_RATE + 1e-6))], "messages": row.get("messages") or []})
                meta.append({"id": str(row["id"]), "span_index": k, "timestamp": timestamp, "silence_dur": round(t - start, 6), "label": label})
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
