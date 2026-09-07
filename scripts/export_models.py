"""Export the user-selected checkpoints, with frozen training-only parity evidence."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from eot.io import sha256_file


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def whisper(root, out):
    from eot.modeling.export_onnx import _fuse, _parity
    from eot.modeling.model import ExportWrapper, load_checkpoint
    from eot.onnx import cpu_session

    checkpoint = root / "data/research/L041/remote/selection/model.pt"
    if sha256_file(checkpoint) != "d82bb9172b83251179b2dd759585532da4db72e38321160dd2b19542df406d4a":
        raise ValueError("Whisper checkpoint is not the selected delivery artifact")
    panel = root / "data/research/L041/export_checks_v2/real_audio_checks.npz"
    if sha256_file(panel) != "af9710fc2cee2b2653dc80a76e75be222b1289ac1345edbf011b3dfe5cc5125c":
        raise ValueError("Selected " + str(panel) + " has changed")
    model = load_checkpoint(str(checkpoint)).float().eval()
    wrapper = ExportWrapper(model).eval()
    z = np.load(panel)
    path = out / "eot.onnx"
    torch.onnx.export(
        wrapper,
        (
            torch.from_numpy(z["input_features"][:2]),
            torch.from_numpy(z["context_ids"][:2]),
        ),
        str(path),
        dynamo=False,
        opset_version=18,
        input_names=["input_features", "context_ids"],
        output_names=["p_eot", "p_fvad"],
        dynamic_axes={k: {0: "batch"} for k in ["input_features", "context_ids", "p_eot", "p_fvad"]},
    )
    fusion = _fuse(path, model)
    session = cpu_session(path, threads=2)
    checks = [(z["input_features"][i : i + b], z["context_ids"][i : i + b]) for b in (1, 4, 8) for i in range(0, 97, b)]
    with torch.inference_mode():
        parity = _parity(wrapper, session, checks)
    parity["batch_sizes"] = [1, 4, 8]
    accepted = max(parity["max_abs_diff_p_eot"], parity["max_abs_diff_p_fvad"]) <= 1e-4
    info = {
        "accepted": accepted,
        "model_name": "Whisper-base distilled",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sha256": sha256_file(path),
        "config": asdict(model.cfg),
        "normalize_audio": model.cfg.normalize_audio,
        "use_context": model.cfg.use_context,
        "use_fvad": model.cfg.use_fvad,
        "horizons": list(model.cfg.horizons),
        "fusion": fusion,
        "parity": parity,
        "parity_tol": 1e-4,
        "checks_sha256": sha256_file(panel),
        "size_mb": path.stat().st_size / 1e6,
        "historical_source_mean_delay_at_5pct_ms": 756.25,
        "runtime": "ONNX Runtime CPU FP32, fused graph",
    }
    write(path.with_suffix(".json"), info)
    print(json.dumps(info), flush=True)
    if not accepted:
        raise RuntimeError("Whisper optimized export failed parity")


def cohere(root, out):
    from eot.modeling.backbone_model import load_checkpoint
    from eot.modeling.export_backbone import BackboneFeatureHead, export_external
    from eot.modeling.lora import merge_lora

    checkpoint = root / "data/research/L041/remote/teachers/seed222/delta.pt"
    if sha256_file(checkpoint) != "98720ce3ee27a4298d47ae532dd06dcfc3893cd7928fff99c1213eff9ebe5ef4":
        raise ValueError("Selected " + str(checkpoint) + " has changed")
    panel = root / "data/research/L041/remote/teacher_qat/checks/model.pt"
    if sha256_file(panel) != "c53e66d53b2db63b9fab4ebb0304c32043ee8d52c802143d24b45b0558534817":
        raise ValueError("Selected " + str(panel) + " has changed")
    waves = torch.load(panel, map_location="cpu", weights_only=False)["waves"]
    from eot.audio import last_window

    x = np.stack([last_window(w, 4) for w in waves])
    model = load_checkpoint(checkpoint).float().eval().requires_grad_(False).cuda()
    reference = []
    with torch.inference_mode():
        for w in x:
            reference.append(float(model(torch.from_numpy(w[None]).cuda())["p_eot"][0]))
    merged = merge_lora(model)
    head = BackboneFeatureHead(model).eval()
    features = []
    merged_reference = []
    with torch.inference_mode():
        for w in x:
            f = model.backbone.frontend(torch.from_numpy(w[None]).cuda())
            features.append(f.cpu().numpy())
            merged_reference.append(float(head(f)[0]))
    diff = float(np.max(np.abs(np.array(reference) - merged_reference)))
    if diff > 1e-4:
        raise RuntimeError(f"Merged source differs: {diff}")
    torch.save(model.backbone.filterbank.to_payload(), out / "frontend.pt")
    np.savez_compressed(
        out / "parity-inputs.npz",
        features=np.concatenate(features),
        reference=np.array(reference),
        waves=x,
    )
    print(f"Exporting Cohere encoder + head; merged {merged}, max diff {diff}", flush=True)
    head.cpu()
    model.cpu()
    torch.cuda.empty_cache()
    export_external(head, torch.from_numpy(features[0]), out / "eot.onnx")
    write(
        out / "source.json",
        {
            "model_name": "Cohere fine-tuned",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "onnx_sha256": sha256_file(out / "eot.onnx"),
            "weights_sha256": sha256_file(out / "eot.onnx.data"),
            "frontend_sha256": sha256_file(out / "frontend.pt"),
            "checks_sha256": sha256_file(panel),
            "merged_lora_layers": merged,
            "merged_probability_diff": diff,
            "historical_source_mean_delay_at_5pct_ms": 615,
            "reference": "Unmerged CUDA FP32, TF32 disabled",
            "frontend": "Unchanged original FP32 Cohere frontend; deterministic dither and BF16-rounded original buffers",
        },
    )
    print("Cohere ONNX exported", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["whisper", "cohere"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project/data root containing the recorded data/ layout",
    )
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Output must be empty; preserve previous exports and receipts")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    {"whisper": whisper, "cohere": cohere}[a.kind](a.root, a.out)
