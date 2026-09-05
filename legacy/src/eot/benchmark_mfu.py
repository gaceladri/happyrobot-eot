"""Benchmark training throughput and estimated MFU on one CUDA GPU.

The benchmark uses the real mined-audio Dataset and EOTModel training step. FLOPs are
counted once in eager mode with PyTorch's operator-level FlopCounterMode, then scaled
linearly by batch size. MFU is therefore an estimate: it covers operators known to the
PyTorch counter and uses the configured dense FP16 peak for the GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from .data import MinedDataset, collate, grouped_split, read_samples
from .model import EOTConfig, EOTModel
from .train import seed_everything, seed_worker


@dataclass(frozen=True)
class BenchConfig:
    name: str
    batch_size: int
    workers: int = 0
    fused_adamw: bool = False
    compile_model: bool = False


DEFAULT_CONFIGS = (
    BenchConfig("current-b32", 32),
    BenchConfig("batch32-workers4", 32, workers=4),
    BenchConfig("batch32-fused-compile", 32, workers=4, fused_adamw=True, compile_model=True),
    BenchConfig("batch64", 64),
    BenchConfig("batch128", 128),
    BenchConfig("batch256", 256),
    BenchConfig("batch128-fused", 128, fused_adamw=True),
    BenchConfig("batch128-workers4", 128, workers=4),
    BenchConfig("batch128-fused-compile", 128, workers=4, fused_adamw=True, compile_model=True),
    BenchConfig("batch256-fused-compile", 256, workers=4, fused_adamw=True, compile_model=True),
    BenchConfig("batch512-fused-compile", 512, workers=4, fused_adamw=True, compile_model=True),
    BenchConfig("batch768-fused-compile", 768, workers=4, fused_adamw=True, compile_model=True),
    BenchConfig("batch1024-fused-compile", 1024, workers=4, fused_adamw=True, compile_model=True),
    BenchConfig("batch2048-fused-compile", 2048, workers=4, fused_adamw=True, compile_model=True),
)


def _repeat_rows(rows: list[dict], minimum: int) -> list[dict]:
    repeats = max(1, math.ceil(minimum / len(rows)))
    return (rows * repeats)[: max(minimum, len(rows))]


def _loader(rows: list[dict], cfg: BenchConfig, seed: int) -> DataLoader:
    # Keep enough full batches in one iterator that worker startup is outside the
    # measured region, even when the smoke dataset is smaller than the batch.
    bench_rows = _repeat_rows(rows, cfg.batch_size * 4)
    generator = torch.Generator().manual_seed(seed)
    worker_options = (
        {"multiprocessing_context": "spawn", "persistent_workers": True}
        if cfg.workers > 0 else {"persistent_workers": False}
    )
    return DataLoader(
        MinedDataset(bench_rows, telephony_prob=0.3, seed=seed),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate,
        **worker_options,
    )


def _step(model, optimizer, scaler, batch: dict, *, count_flops: bool = False) -> int:
    device_batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    counter = FlopCounterMode(display=False) if count_flops else None
    context = counter if counter is not None else torch.no_grad()  # replaced below for the normal path
    if count_flops:
        with context:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(
                    device_batch["input_features"], device_batch["context_ids"],
                    device_batch["labels"], device_batch["fvad"], device_batch["fvad_mask"],
                )
            scaler.scale(output["loss"]).backward()
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(
                device_batch["input_features"], device_batch["context_ids"],
                device_batch["labels"], device_batch["fvad"], device_batch["fvad_mask"],
            )
        scaler.scale(output["loss"]).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return int(counter.get_total_flops()) if counter is not None else 0


def _measure_one(
    cfg: BenchConfig,
    rows: list[dict],
    args: argparse.Namespace,
    *,
    flops_per_sample: float | None,
) -> tuple[dict, float]:
    seed_everything(args.seed)
    loader = _loader(rows, cfg, args.seed)
    iterator = iter(loader)
    model = EOTModel(EOTConfig()).to("cuda")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=5e-5,
        weight_decay=0.01,
        fused=cfg.fused_adamw,
    )
    scaler = torch.amp.GradScaler(enabled=True)

    # Count on eager operators once. Compiled graphs can hide operations from the
    # counter, while the mathematical work per sample remains the same.
    if flops_per_sample is None:
        probe = next(iterator)
        flops = _step(model, optimizer, scaler, probe, count_flops=True)
        torch.cuda.synchronize()
        flops_per_sample = flops / cfg.batch_size

    train_model = torch.compile(model, mode="reduce-overhead") if cfg.compile_model else model
    for _ in range(args.warmup):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        _step(train_model, optimizer, scaler, batch)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    total_times: list[float] = []
    compute_times: list[float] = []
    data_times: list[float] = []
    for _ in range(args.steps):
        total_start = time.perf_counter()
        data_start = total_start
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_end = time.perf_counter()
        _step(train_model, optimizer, scaler, batch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        data_times.append(data_end - data_start)
        compute_times.append(end - data_end)
        total_times.append(end - total_start)

    median_total = statistics.median(total_times)
    median_compute = statistics.median(compute_times)
    step_flops = flops_per_sample * cfg.batch_size
    peak_flops = args.peak_fp16_tflops * 1e12
    result = {
        **asdict(cfg),
        "steps": args.steps,
        "warmup_steps": args.warmup,
        "flops_per_sample": flops_per_sample,
        "step_tflops": step_flops / 1e12,
        "median_step_ms": median_total * 1e3,
        "median_compute_ms": median_compute * 1e3,
        "median_data_ms": statistics.median(data_times) * 1e3,
        "samples_per_second": cfg.batch_size / median_total,
        "compute_tflops_per_second": step_flops / median_compute / 1e12,
        "mfu_percent_compute": 100.0 * step_flops / median_compute / peak_flops,
        "mfu_percent_end_to_end": 100.0 * step_flops / median_total / peak_flops,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "peak_fp16_tflops": args.peak_fp16_tflops,
    }

    del iterator, loader, train_model, model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result, flops_per_sample


def benchmark(args: argparse.Namespace) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MFU benchmark")
    rows = read_samples(args.samples)
    train_rows, _, _ = grouped_split(rows, dev_frac=0.15, seed=args.seed, return_diagnostics=True)
    selected = [cfg for cfg in DEFAULT_CONFIGS if not args.config or cfg.name in args.config]
    unknown = set(args.config or ()) - {cfg.name for cfg in DEFAULT_CONFIGS}
    if unknown:
        raise ValueError(f"unknown configs: {sorted(unknown)}")

    results: list[dict] = []
    flops_per_sample: float | None = None
    for cfg in selected:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.group,
            job_type="mfu-benchmark",
            name=cfg.name,
            config={**asdict(cfg), "steps": args.steps, "warmup": args.warmup},
            reinit="finish_previous",
        )
        try:
            result, flops_per_sample = _measure_one(
                cfg, train_rows, args, flops_per_sample=flops_per_sample
            )
            run.log(result)
            run.summary.update(result)
            results.append(result)
            print(json.dumps(result, sort_keys=True))
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            failure = {**asdict(cfg), "status": "oom", "error": str(exc)}
            run.summary.update(failure)
            results.append(failure)
            print(json.dumps(failure, sort_keys=True))
        finally:
            run.finish()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/mfu-benchmark/results.json"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak-fp16-tflops", type=float, default=142.3)
    parser.add_argument("--wandb-project", default="happyrobot-eot")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--group", default="rtx3090-mfu-tuning")
    parser.add_argument("--config", action="append", help="run only one named config; repeatable")
    return parser


def main(argv: list[str] | None = None) -> None:
    benchmark(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
