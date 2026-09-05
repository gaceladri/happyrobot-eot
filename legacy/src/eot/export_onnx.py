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
    validation_samples: Path | None = None,
    fuse: bool = True,
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

    fusion_stats = {}
    if fuse:
        from onnxruntime.transformers.optimizer import optimize_model
        optimized = optimize_model(str(out), model_type="bart",
                                   num_heads=model.encoder.config.encoder_attention_heads,
                                   hidden_size=model.encoder.config.d_model, opt_level=1)
        fusion_stats = optimized.get_fused_operator_statistics()
        optimized.save_model_to_file(str(out))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(out), options, providers=["CPUExecutionProvider"])
    session_inputs = {item.name for item in sess.get_inputs()}

    def feed(features: np.ndarray, context: np.ndarray) -> dict[str, np.ndarray]:
        values = {"input_features": features}
        # ONNX legitimately prunes context_ids from an audio-only graph. Keep one
        # exporter/service path that also works when a future trained model uses it.
        if "context_ids" in session_inputs:
            values["context_ids"] = context
        return values

    # Cover variable batch shapes, silence, short/long real audio, noise and context.
    rng = np.random.default_rng(0)
    checks = []
    for n in (1, 2, 8):
        f = rng.normal(size=(n, N_MELS, N_FRAMES)).astype(np.float32)
        c = rng.integers(0, 4096, size=(n, CTX_LEN), dtype=np.int64) if model.cfg.use_context else np.zeros((n, CTX_LEN), np.int64)
        checks.append((f, c))
    from .audio import log_mel, SAMPLE_RATE, telephony_augment
    audio = [np.zeros(SAMPLE_RATE, np.float32)]
    if validation_samples:
        import soundfile as sf
        from .data import read_samples
        rows = read_samples(validation_samples)
        for idx in np.linspace(0, len(rows)-1, min(48,len(rows)), dtype=int):
            x, sr = sf.read(rows[idx]['path'], dtype='float32')
            from .audio import resample, to_mono
            x = resample(to_mono(x), sr)
            audio.extend([x, telephony_augment(x, SAMPLE_RATE, rng)])
    af = np.stack([log_mel(x, normalize=model.cfg.normalize_audio) for x in audio])
    for i in range(0, len(af), 8):
        checks.append((af[i:i+8], np.zeros((len(af[i:i+8]),CTX_LEN), np.int64)))
    def check(session):
        diffs, fdiffs, flips = [], [], []
        for f,c in checks:
            with torch.no_grad():
                rp,rf = wrapper(torch.from_numpy(f),torch.from_numpy(c))
            gp,gf = session.run(None,feed(f,c))
            diffs.extend(np.abs(rp.numpy()-gp).reshape(-1).tolist())
            fdiffs.extend(np.abs(rf.numpy()-gf).reshape(-1).tolist())
            flips.extend(((rp.numpy()>=.5)!=(gp>=.5)).reshape(-1).tolist())
        return dict(max_abs_diff_p_eot=max(diffs),max_abs_diff_p_fvad=max(fdiffs),
                    mean_abs_diff_p_eot=float(np.mean(diffs)),decision_flip_rate_at_0_5=float(np.mean(flips)),
                    n_examples=len(diffs),real_audio_examples=len(audio)-1,batch_sizes=[1,2,8])
    parity=check(sess)
    for _ in range(5): sess.run(None, feed(af[:1], np.zeros((1,CTX_LEN),np.int64)))
    timings=[]
    for _ in range(30):
        t0=time.perf_counter()
        sess.run(None, feed(af[:1],np.zeros((1,CTX_LEN),np.int64)))
        timings.append((time.perf_counter()-t0)*1000)
    parity['fp32_ms_per_forward_cpu']=float(np.mean(timings))
    parity['fp32_p95_ms']=float(np.percentile(timings,95))
    fp32_max_diff=max(parity['max_abs_diff_p_eot'],parity['max_abs_diff_p_fvad'])
    info = {
        "checkpoint": str(checkpoint),
        "fusion": fusion_stats,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "normalize_audio": model.cfg.normalize_audio, "use_fvad": model.cfg.use_fvad,
        "encoder_layers": len(model.encoder.layers),
        "validation_samples": str(validation_samples) if validation_samples else None,
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
        try:
            # Keep convolutions in FP32; quantize only supported linear operators.
            import onnx
            # Endpoint probabilities are sensitive to the small pooling/classifier heads.
            # Keep those in FP32; most linear compute is in the encoder feed-forward blocks.
            excluded = [n.name for n in onnx.load(str(out)).graph.node
                        if "classifier" in n.name or "/pool/" in n.name]
            quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8, per_channel=True,
                             reduce_range=False, op_types_to_quantize=['MatMul','Gemm'],
                             nodes_to_exclude=excluded,
                             extra_options={'DefaultTensorType': 1, 'MatMulConstBOnly': True})
            qs = ort.InferenceSession(str(q), options, providers=['CPUExecutionProvider'])
            qparity=check(qs)
            d=max(qparity['max_abs_diff_p_eot'],qparity['max_abs_diff_p_fvad'])
            accepted=d<=int8_parity_tol and validation_samples is not None
            info['int8']=dict(path=str(q) if accepted else None,parity=qparity,
                              max_abs_diff_p_eot=qparity['max_abs_diff_p_eot'],accepted=accepted,
                              size_mb=q.stat().st_size/1e6,parity_tol=int8_parity_tol)
            if accepted:
                qinfo={**info,'onnx':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),
                       'size_mb':q.stat().st_size/1e6,'parity':qparity,'parity_tol':int8_parity_tol}
                q.with_suffix('.json').write_text(json.dumps(qinfo,indent=2))
            else:
                q.unlink(missing_ok=True)
                q.with_suffix('.json').unlink(missing_ok=True)
        except (RuntimeError, ValueError) as error:
            info['int8'] = {'accepted': False, 'path': None, 'error': str(error)}
            q.unlink(missing_ok=True)
            q.with_suffix('.json').unlink(missing_ok=True)
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
    print(json.dumps(
        export(
            args.checkpoint,
            args.out,
            args.opset,
            args.int8,
            args.parity_tol,
            args.int8_parity_tol,
            args.validation_samples,
            not args.no_fuse,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
