"""EoT Bench adapter and scorers for backbone checkpoints (``eot.modeling.backbone_model``).

Harness contract (``eot-harness predict --adapter eot.eval.backbone:BackboneAdapter`` with ``EOT_CHECKPOINT``):
``adapter_id``, ``score_point`` 0.2, ``max_audio_sec`` 8.0, ``predict_batch(batch)`` over ``{"audio", "messages"}``
items. Each item is decoded with ``eot.audio.decode_payload`` and cut to the last 8 s with ``eot.audio.last_window``,
the same window the Whisper adapter feeds ``log_mel``; the model crops its trailing ``window_s`` inside ``encode``.
Torch backend in IEEE fp32 (TF32 off), the same arithmetic the ONNX deployment graph uses.

CLI (the same functions ``eot-krisp score`` and ``eot-report heldout`` call for Whisper checkpoints):

    uv run eot-backbone-score krisp   --checkpoint delta.pt --clips data/raw/krisp/clips/clips.jsonl --out eval/krisp/<run>.jsonl
    uv run eot-backbone-score heldout --checkpoint delta.pt --samples data/mined/apptek-oracle/heldout/samples.jsonl --out eval/heldout/<run>.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from eot.audio import SCORE_POINT, decode_payload, last_window
from eot.io import sha256_file


class BackboneAdapter:
    adapter_id = "backbone"
    score_point = SCORE_POINT
    max_audio_sec = 8.0

    def __init__(self, checkpoint: str | None = None, device: str | None = None, batch_size: int = 64, merge_lora: bool = False):
        import torch

        from eot.modeling.backbone_model import load_checkpoint
        from eot.modeling.lora import merge_lora as merge

        checkpoint = checkpoint or os.environ.get("EOT_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError("EOT_CHECKPOINT (a backbone checkpoint) is required; backbone models have no ONNX export")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_checkpoint(checkpoint).to(self.device).eval()
        self.merged_lora_layers = merge(self.model) if merge_lora else 0
        self.cfg = self.model.cfg
        self.model_sha = sha256_file(Path(checkpoint))
        self.batch_size = int(os.environ.get("EOT_ADAPTER_BATCH_SIZE", batch_size))
        if self.batch_size < 1:
            raise ValueError("EOT_ADAPTER_BATCH_SIZE must be positive")
        self.adapter_id = f"{self.cfg.backbone}-w{self.cfg.window_s:g}s-audio-{self.model_sha[:12]}-ieee-fp32"
        self.display_name = os.environ.get("EOT_DISPLAY_NAME") or self.adapter_id

    def predict_waves(self, waves: np.ndarray) -> np.ndarray:
        t = self._torch
        out = []
        with t.no_grad():
            for i in range(0, len(waves), self.batch_size):
                x = t.from_numpy(np.ascontiguousarray(waves[i : i + self.batch_size], dtype=np.float32)).to(self.device)
                out.append(self.model(x)["p_eot"].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def predict_batch(self, batch) -> list[float]:
        waves = np.stack([last_window(decode_payload(item["audio"]), self.cfg.input_s) for item in batch])
        return [float(v) for v in self.predict_waves(waves)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    krisp = sub.add_parser("krisp", help="score the Krisp clips manifest")
    krisp.add_argument("--clips", type=Path, required=True)
    heldout = sub.add_parser("heldout", help="score a held-out samples manifest")
    heldout.add_argument("--samples", type=Path, required=True)
    for p in (krisp, heldout):
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", default=None)
        p.add_argument("--merge-lora", action="store_true")
    args = ap.parse_args(argv)
    started = time.time()
    adapter = BackboneAdapter(checkpoint=args.checkpoint, device=args.device, merge_lora=args.merge_lora)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "krisp":
        from eot.eval import krisp as krisp_eval

        report = krisp_eval.score(args.clips, adapter, args.out)
        execution = {
            "device": str(adapter.device),
            "batch_size": adapter.batch_size,
            "dtype": "fp32",
            "elapsed_s": time.time() - started,
            "merged_lora_layers": adapter.merged_lora_layers,
        }
        args.out.with_suffix(".execution.json").write_text(json.dumps(execution, indent=1))
    else:
        from eot.eval import report as report_eval

        report = {
            k: v
            for k, v in report_eval.heldout(args.samples, adapter, args.out).items()
            if k in ("filled", "raw", "shortcut_auc_gain_raw_minus_filled")
        }
    print(json.dumps(report, default=str), f"{time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
