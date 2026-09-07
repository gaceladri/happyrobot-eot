"""Build a local, reproducible TensorRT engine; never accept it before parity."""

import argparse
import json
import time
from pathlib import Path

import tensorrt as trt
import torch

from eot.io import sha256_file


def build(source, out, precision, parse_only=False, tuned=False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    started = time.perf_counter()
    ok = parser.parse_from_file(str(source))
    meta = {
        "accepted": False,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "precision": precision,
        "tf32": False,
        "batch_profile": {"min": 1, "opt": 4, "max": 8},
        "parser_errors": [str(parser.get_error(i)) for i in range(parser.num_errors)],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if not ok or parse_only:
        meta["status"] = "PARSED" if ok else "PARSER_REJECTED"
        out.with_suffix(".build.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta), flush=True)
        return ok
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.builder_optimization_level = 3
    if tuned:
        config.builder_optimization_level = 4
        config.max_aux_streams = 0
    protected = []
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if (
                layer.type
                in (
                    trt.LayerType.REDUCE,
                    trt.LayerType.SOFTMAX,
                    trt.LayerType.NORMALIZATION,
                )
                or "norm" in layer.name.lower()
                or (tuned and any(k in layer.name for k in ("/pool/", "/classifier/", "/Sigmoid")))
            ) and any(layer.get_output(k).dtype == trt.float32 for k in range(layer.num_outputs)):
                layer.precision = trt.float32
                for k in range(layer.num_outputs):
                    if layer.get_output(k).dtype == trt.float32:
                        layer.set_output_type(k, trt.float32)
                protected.append(layer.name)
    shapes = [(1, 1, 1), (2, 4, 4), (5, 8, 8)] if tuned else [(1, 4, 8)]
    for low, opt, high in shapes:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            name = network.get_input(i).name
            profile.set_shape(name, (low, 128, 401), (opt, 128, 401), (high, 128, 401))
        config.add_optimization_profile(profile)
    meta.update(
        tuned=tuned,
        profiles=shapes,
        builder_optimization_level=config.builder_optimization_level,
        max_aux_streams=config.max_aux_streams,
    )
    cache_file = out.parent / "timing.cache"
    cache = config.create_timing_cache(cache_file.read_bytes() if cache_file.exists() else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    print(
        f"Building {precision}: {network.num_layers} layers, {len(protected)} FP32 constraints",
        flush=True,
    )
    serialized = builder.build_serialized_network(network, config)
    cache_file.write_bytes(config.get_timing_cache().serialize())
    meta.update(
        build_seconds=time.perf_counter() - started,
        protected_layers=protected,
        network_layers=network.num_layers,
    )
    if serialized is None:
        meta["status"] = "BUILD_FAILED"
    else:
        out.write_bytes(serialized)
        meta.update(
            status="BUILT_PENDING_PARITY",
            engine_sha256=sha256_file(out),
            engine_bytes=out.stat().st_size,
        )
    out.with_suffix(".build.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta["status"], flush=True)
    return serialized is not None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    p.add_argument("--parse-only", action="store_true")
    p.add_argument("--tuned", action="store_true")
    a = p.parse_args()
    if a.out.exists() or a.out.with_suffix(".build.json").exists():
        p.error("Output already exists; preserve earlier engines and build receipts")
    if not build(a.onnx, a.out, a.precision, a.parse_only, a.tuned):
        raise SystemExit(2)
