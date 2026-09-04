"""ONNX export with FP32 parity check (and optional INT8 dynamic quantisation).

    uv run eot-export --checkpoint runs/base/model.pt --out artifacts/eot.onnx [--int8]

Writes ``eot.onnx`` plus ``eot.json`` (config, sha256, parity stats). FP32 is the default
serving artifact: 32 MB already meets the <100 ms CPU budget by a wide margin; INT8 is produced
only if requested and is accepted only if max |Δp| stays under ``--parity-tol``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audio import N_FRAMES, N_MELS
from .model import CTX_LEN, ExportWrapper, load_checkpoint


def export(
    checkpoint: Path,
    out: Path,
    opset: int = 18,
    int8: bool = False,
    parity_tol: float = 1e-4,
    int8_parity_tol: float = 0.02,
) -> dict:
    if parity_tol <= 0 or int8_parity_tol <= 0:
        raise ValueError("parity tolerances must be positive")
    model = load_checkpoint(str(checkpoint)).cpu().eval()
    wrapper = ExportWrapper(model).eval()
    feats = torch.randn(2, N_MELS, N_FRAMES)
    ctx = torch.zeros(2, CTX_LEN, dtype=torch.long)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (feats, ctx), str(out), opset_version=opset, dynamo=False, verbose=False,
        input_names=["input_features", "context_ids"], output_names=["p_eot", "p_fvad"],
        dynamic_axes={"input_features": {0: "batch"}, "context_ids": {0: "batch"}, "p_eot": {0: "batch"}, "p_fvad": {0: "batch"}},
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    session_inputs = {item.name for item in sess.get_inputs()}

    def feed(features: np.ndarray, context: np.ndarray) -> dict[str, np.ndarray]:
        values = {"input_features": features}
        # ONNX legitimately prunes context_ids from an audio-only graph. Keep one
        # exporter/service path that also works when a future trained model uses it.
        if "context_ids" in session_inputs:
            values["context_ids"] = context
        return values

    with torch.no_grad():
        ref_p, ref_f = wrapper(feats, ctx)
    got_p, got_f = sess.run(None, feed(feats.numpy(), ctx.numpy()))
    parity = {"max_abs_diff_p_eot": float(np.abs(ref_p.numpy() - got_p).max()), "max_abs_diff_p_fvad": float(np.abs(ref_f.numpy() - got_f).max())}
    t0 = time.perf_counter()
    for _ in range(10):
        sess.run(None, feed(feats[:1].numpy(), ctx[:1].numpy()))
    parity["fp32_ms_per_forward_cpu"] = (time.perf_counter() - t0) / 10 * 1000
    fp32_max_diff = max(parity["max_abs_diff_p_eot"], parity["max_abs_diff_p_fvad"])
    info = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "use_context": model.cfg.use_context, "horizons": list(model.cfg.horizons), "opset": opset, "parity": parity,
        "size_mb": out.stat().st_size / 1e6, "parity_tol": parity_tol,
        "accepted": fp32_max_diff <= parity_tol,
    }
    if not info["accepted"]:
        out.with_suffix(".json").write_text(json.dumps(info, indent=2))
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"FP32 ONNX parity failed: max abs diff {fp32_max_diff:.6g} exceeds {parity_tol:.6g}"
        )
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out.with_name(out.stem + ".int8.onnx")
        quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8)
        qs = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
        q_inputs = {item.name for item in qs.get_inputs()}
        q_feed = {"input_features": feats.numpy()}
        if "context_ids" in q_inputs:
            q_feed["context_ids"] = ctx.numpy()
        qp, _ = qs.run(None, q_feed)
        d = float(np.abs(ref_p.numpy() - qp).max())
        accepted = d <= int8_parity_tol
        info["int8"] = {
            "path": str(q) if accepted else None,
            "max_abs_diff_p_eot": d,
            "accepted": accepted,
            "size_mb": q.stat().st_size / 1e6,
            "parity_tol": int8_parity_tol,
        }
        if not accepted:
            q.unlink(missing_ok=True)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--parity-tol", type=float, default=1e-4, help="maximum FP32 ONNX probability drift")
    ap.add_argument("--int8-parity-tol", type=float, default=0.02)
    args = ap.parse_args(argv)
    print(json.dumps(
        export(
            args.checkpoint,
            args.out,
            args.opset,
            args.int8,
            args.parity_tol,
            args.int8_parity_tol,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
