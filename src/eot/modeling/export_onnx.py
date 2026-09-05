"""ONNX export with enforced FP32 parity (and optional INT8 dynamic quantisation).

    uv run eot-export --checkpoint runs/base/model.pt --out artifacts/eot.onnx [--int8]

Writes ``eot.onnx`` plus ``eot.json`` (config, sha256, parity statistics). FP32 is the serving
artifact: 32 MB already meets the <100 ms CPU budget by a wide margin. INT8 is produced only on
request and kept only if max |Δp| stays under ``--int8-parity-tol`` on real audio; the shipped
model's INT8 variants failed that gate and were discarded rather than served.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from eot.audio import N_FRAMES, N_MELS, SAMPLE_RATE, load_wav, log_mel, telephony_augment
from eot.context import CTX_LEN
from eot.modeling.model import EOTModel, ExportWrapper, load_checkpoint

Check = tuple[np.ndarray, np.ndarray]  # (input_features [B,80,800], context_ids [B,32])


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _feed(session, features: np.ndarray, context: np.ndarray) -> dict[str, np.ndarray]:
    values = {"input_features": features}
    # ONNX legitimately prunes ``context_ids`` from an audio-only graph; keep one path that also
    # works when a trained model uses context.
    if "context_ids" in {item.name for item in session.get_inputs()}:
        values["context_ids"] = context
    return values


def _parity_inputs(model: EOTModel, validation_samples: Path | None, rng: np.random.Generator) -> tuple[list[Check], int]:
    """Random batches of size 1/2/8, digital silence, and (optionally) real clean + telephony audio."""
    checks: list[Check] = []
    for n in (1, 2, 8):
        features = rng.normal(size=(n, N_MELS, N_FRAMES)).astype(np.float32)
        context = (rng.integers(0, 4096, size=(n, CTX_LEN), dtype=np.int64) if model.cfg.use_context
                   else np.zeros((n, CTX_LEN), np.int64))
        checks.append((features, context))
    audio = [np.zeros(SAMPLE_RATE, np.float32)]
    if validation_samples:
        from eot.data.dataset import read_samples

        rows = read_samples(validation_samples)
        for index in np.linspace(0, len(rows) - 1, min(48, len(rows)), dtype=int):
            x = load_wav(rows[index]["path"])
            audio.extend([x, telephony_augment(x, SAMPLE_RATE, rng)])
    features = np.stack([log_mel(x, normalize=model.cfg.normalize_audio) for x in audio])
    for i in range(0, len(features), 8):
        batch = features[i:i + 8]
        checks.append((batch, np.zeros((len(batch), CTX_LEN), np.int64)))
    return checks, len(audio) - 1


def _parity(wrapper: ExportWrapper, session, checks: list[Check]) -> dict:
    diffs, fvad_diffs, flips = [], [], []
    for features, context in checks:
        with torch.no_grad():
            ref_p, ref_f = wrapper(torch.from_numpy(features), torch.from_numpy(context))
        p, f = session.run(None, _feed(session, features, context))
        diffs.extend(np.abs(ref_p.numpy() - p).reshape(-1).tolist())
        fvad_diffs.extend(np.abs(ref_f.numpy() - f).reshape(-1).tolist())
        flips.extend(((ref_p.numpy() >= 0.5) != (p >= 0.5)).reshape(-1).tolist())
    return {
        "max_abs_diff_p_eot": max(diffs),
        "max_abs_diff_p_fvad": max(fvad_diffs),
        "mean_abs_diff_p_eot": float(np.mean(diffs)),
        "decision_flip_rate_at_0_5": float(np.mean(flips)),
        "n_examples": len(diffs),
        "batch_sizes": [1, 2, 8],
    }


def _time_forward(session, features: np.ndarray, warmup: int = 5, repeats: int = 30) -> dict:
    feed = _feed(session, features[:1], np.zeros((1, CTX_LEN), np.int64))
    for _ in range(warmup):
        session.run(None, feed)
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        session.run(None, feed)
        timings.append((time.perf_counter() - t0) * 1000)
    return {"fp32_ms_per_forward_cpu": float(np.mean(timings)), "fp32_p95_ms": float(np.percentile(timings, 95))}


def _fuse(path: Path, model: EOTModel) -> dict:
    from onnxruntime.transformers.optimizer import optimize_model

    optimized = optimize_model(
        str(path), model_type="bart", opt_level=1,
        num_heads=model.encoder.config.encoder_attention_heads, hidden_size=model.encoder.config.d_model,
    )
    optimized.save_model_to_file(str(path))
    return optimized.get_fused_operator_statistics()


def _quantize_int8(fp32_path: Path, wrapper: ExportWrapper, checks: list[Check], tolerance: float, has_real_audio: bool, info: dict) -> dict:
    """Dynamic INT8 on MatMul/Gemm only; the small pooling/classifier heads stay FP32."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q = fp32_path.with_name(fp32_path.stem + ".int8.onnx")
    try:
        excluded = [n.name for n in onnx.load(str(fp32_path)).graph.node if "classifier" in n.name or "/pool/" in n.name]
        quantize_dynamic(
            str(fp32_path), str(q), weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
            op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=excluded,
            extra_options={"DefaultTensorType": 1, "MatMulConstBOnly": True},
        )
        parity = _parity(wrapper, _session(q), checks)
        drift = max(parity["max_abs_diff_p_eot"], parity["max_abs_diff_p_fvad"])
        accepted = drift <= tolerance and has_real_audio  # never accept INT8 on synthetic inputs alone
        result = {"path": str(q) if accepted else None, "parity": parity, "max_abs_diff_p_eot": parity["max_abs_diff_p_eot"],
                  "accepted": accepted, "size_mb": q.stat().st_size / 1e6, "parity_tol": tolerance}
        if accepted:
            q.with_suffix(".json").write_text(json.dumps({
                **info, "onnx": str(q), "sha256": hashlib.sha256(q.read_bytes()).hexdigest(),
                "size_mb": result["size_mb"], "parity": parity, "parity_tol": tolerance,
            }, indent=2))
        else:
            q.unlink(missing_ok=True)
            q.with_suffix(".json").unlink(missing_ok=True)
        return result
    except (RuntimeError, ValueError) as error:
        q.unlink(missing_ok=True)
        q.with_suffix(".json").unlink(missing_ok=True)
        return {"accepted": False, "path": None, "error": str(error)}


def export(
    checkpoint: Path,
    out: Path,
    opset: int = 18,
    int8: bool = False,
    parity_tol: float = 1e-4,
    int8_parity_tol: float = 0.02,
    validation_samples: Path | None = None,
    fuse: bool = True,
) -> dict:
    if parity_tol <= 0 or int8_parity_tol <= 0:
        raise ValueError("parity tolerances must be positive")
    model = load_checkpoint(str(checkpoint)).cpu().eval()
    wrapper = ExportWrapper(model).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (torch.randn(2, N_MELS, N_FRAMES), torch.zeros(2, CTX_LEN, dtype=torch.long)), str(out),
        opset_version=opset, dynamo=False, verbose=False,
        input_names=["input_features", "context_ids"], output_names=["p_eot", "p_fvad"],
        dynamic_axes={name: {0: "batch"} for name in ("input_features", "context_ids", "p_eot", "p_fvad")},
    )
    fusion_stats = _fuse(out, model) if fuse else {}

    session = _session(out)
    checks, n_real_audio = _parity_inputs(model, validation_samples, np.random.default_rng(0))
    parity = {**_parity(wrapper, session, checks), "real_audio_examples": n_real_audio}
    parity.update(_time_forward(session, checks[-1][0]))
    fp32_drift = max(parity["max_abs_diff_p_eot"], parity["max_abs_diff_p_fvad"])
    info = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "size_mb": out.stat().st_size / 1e6,
        "opset": opset,
        "fusion": fusion_stats,
        "normalize_audio": model.cfg.normalize_audio,
        "use_context": model.cfg.use_context,
        "use_fvad": model.cfg.use_fvad,
        "horizons": list(model.cfg.horizons),
        "encoder_layers": len(model.encoder.layers),
        "validation_samples": str(validation_samples) if validation_samples else None,
        "parity": parity,
        "parity_tol": parity_tol,
        "accepted": fp32_drift <= parity_tol,
    }
    if not info["accepted"]:
        out.with_suffix(".json").write_text(json.dumps(info, indent=2))
        out.unlink(missing_ok=True)
        raise RuntimeError(f"FP32 ONNX parity failed: max abs diff {fp32_drift:.6g} exceeds {parity_tol:.6g}")
    if int8:
        info["int8"] = _quantize_int8(out, wrapper, checks, int8_parity_tol, n_real_audio > 0, info)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--validation-samples", type=Path, default=None)
    ap.add_argument("--no-fuse", action="store_true", help="disable attention fusion for a controlled timing comparison")
    ap.add_argument("--parity-tol", type=float, default=1e-4, help="maximum FP32 ONNX probability drift")
    ap.add_argument("--int8-parity-tol", type=float, default=0.02)
    args = ap.parse_args(argv)
    info = export(args.checkpoint, args.out, args.opset, args.int8, args.parity_tol, args.int8_parity_tol,
                  args.validation_samples, not args.no_fuse)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
