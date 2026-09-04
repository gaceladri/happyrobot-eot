"""Full fine-tuning of Whisper-Tiny with restart-safe epoch checkpoints.

The audio-only + FVAD path remains the default. Context is opt-in and guarded by measured
coverage so an all-padding context branch can never be shipped accidentally.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import secrets
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    MinedDataset,
    append_jsonl_record,
    atomic_write_json,
    atomic_write_jsonl,
    collate,
    grouped_split,
    read_jsonl_records,
    read_samples,
    sha256_file,
)
from .model import EOTConfig, EOTModel


def pick_device(name: str | None = None, *, require_cuda: bool = False) -> torch.device:
    if name:
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError(
            f"CUDA is required but unavailable (selected={device}, torch={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda!r})"
        )
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint next to its destination and atomically publish it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def should_save_best(current: float, best: float | None, *, has_best: bool) -> bool:
    """Always select the first epoch, then prefer finite AUC improvements."""
    if not has_best:
        return True
    if not math.isfinite(current):
        return False
    return best is None or not math.isfinite(best) or current > best


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC without sklearn."""
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(
    model: EOTModel, loader: DataLoader, device: torch.device, *, non_blocking: bool = False
) -> tuple[dict, list[dict]]:
    model.eval()
    ys, ps, rows = [], [], []
    for batch in loader:
        b = {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}
        out = model(b["input_features"], b["context_ids"], b["labels"], b["fvad"], b["fvad_mask"])
        ys.append(b["labels"].cpu().numpy())
        ps.append(out["p_eot"].cpu().numpy())
        for index in range(len(b["labels"])):
            rows.append({
                "label": int(b["labels"][index]),
                "p_eot": float(out["p_eot"][index]),
                "p_fvad": out["p_fvad"][index].cpu().tolist() if "p_fvad" in out else None,
            })
    if not ys:
        raise ValueError("development loader is empty")
    y, p = np.concatenate(ys), np.concatenate(ps)
    acc = float(((p > 0.5) == (y == 1)).mean())
    return {"auc": roc_auc(y, p), "acc@0.5": acc, "n": int(len(y)), "pos_rate": float(y.mean())}, rows


def _validate_binary_rows(rows: list[dict], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} split is empty")
    labels = {int(row["label"]) for row in rows}
    if labels != {0, 1}:
        raise ValueError(f"{name} split must contain labels 0 and 1; found {sorted(labels)}")


def _context_coverage(rows: list[dict]) -> float:
    return sum(bool(str(row.get("agent_text", "")).strip()) for row in rows) / max(1, len(rows))


def _checkpoint_payload(
    *,
    model: EOTModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    next_epoch: int,
    next_batch: int,
    global_step: int,
    best_auc: float | None,
    has_best: bool,
    args: argparse.Namespace,
    loader_generator: torch.Generator,
    epoch_generator_state: torch.Tensor | None,
    diagnostics: dict,
    planned_total_steps: int,
    samples_sha256: str,
) -> dict:
    return {
        "format_version": 2,
        "cfg": asdict(model.cfg),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "next_epoch": int(next_epoch),
        "next_batch": int(next_batch),
        "global_step": int(global_step),
        "best_auc": best_auc,
        "has_best": bool(has_best),
        "rng_state": capture_rng_state(),
        "loader_generator_state": loader_generator.get_state(),
        "epoch_generator_state": epoch_generator_state,
        "args": vars(args),
        "split_diagnostics": diagnostics,
        "planned_total_steps": int(planned_total_steps),
        "samples_sha256": samples_sha256,
    }


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_resume_args(checkpoint: dict, args: argparse.Namespace) -> None:
    """Reject changes that would alter samples, gradients, or the LR schedule."""
    saved = checkpoint.get("args") or {}
    protected = (
        "base_model", "base_revision", "epochs", "batch_size", "lr", "weight_decay",
        "warmup_ratio", "dev_frac", "telephony_prob", "fvad_weight", "use_context",
        "min_context_coverage", "no_fvad", "freeze_encoder", "workers", "seed", "max_steps",
        "fused_adamw", "compile_model", "compile_mode",
    )
    mismatches = {
        key: (saved[key], getattr(args, key))
        for key in protected
        if key in saved and saved[key] != getattr(args, key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}={old!r} (requested {new!r})"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise ValueError(f"resume arguments changed training semantics: {details}")


def _resolve_resume(value: str | None, out_dir: Path) -> Path | None:
    if value is None:
        return None
    path = out_dir / "checkpoints" / "last.pt" if value == "auto" else Path(value)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def train(args: argparse.Namespace) -> Path:
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs and batch-size must be positive; workers cannot be negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive when set")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    seed_everything(args.seed)
    device = pick_device(args.device, require_cuda=args.require_cuda)
    out_dir = Path(args.out)
    checkpoints_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    resume_path = _resolve_resume(args.resume, out_dir)
    if resume_path is None and any(
        path.exists() for path in (checkpoints_dir / "last.pt", out_dir / "model.pt", out_dir / "history.jsonl")
    ):
        raise FileExistsError(f"{out_dir} already contains a run; pass --resume or choose a new --out")
    restored = _load_checkpoint(resume_path) if resume_path is not None else None
    if restored is not None:
        _validate_resume_args(restored, args)

    samples_sha256 = sha256_file(args.samples)
    if restored is not None and restored.get("samples_sha256") not in (None, samples_sha256):
        raise ValueError("samples manifest changed since the resume checkpoint was written")
    rows = read_samples(args.samples)
    if not rows:
        raise ValueError("samples manifest is empty")
    if {int(row["label"]) for row in rows} != {0, 1}:
        raise ValueError("samples manifest must contain both labels 0 and 1")
    tr_rows, dv_rows, diagnostics = grouped_split(
        rows, dev_frac=args.dev_frac, seed=args.seed, return_diagnostics=True
    )
    _validate_binary_rows(tr_rows, "train")
    _validate_binary_rows(dv_rows, "dev")
    print(json.dumps({"split": diagnostics, "device": str(device)}, sort_keys=True))

    train_context_coverage = _context_coverage(tr_rows)
    dev_context_coverage = _context_coverage(dv_rows)
    if args.use_context and min(train_context_coverage, dev_context_coverage) < args.min_context_coverage:
        raise ValueError(
            "--use-context requested but context coverage is too low: "
            f"train={train_context_coverage:.3f}, dev={dev_context_coverage:.3f}, "
            f"required={args.min_context_coverage:.3f}"
        )

    if restored is not None:
        cfg_values = dict(restored["cfg"])
        cfg_values["horizons"] = tuple(cfg_values["horizons"])
        cfg = EOTConfig(**cfg_values)
        if cfg.use_context != args.use_context or cfg.use_fvad != (not args.no_fvad):
            raise ValueError("resume flags --use-context/--no-fvad must match the checkpoint")
        model = EOTModel(cfg, pretrained=False)
        model.load_state_dict(restored["state_dict"])
    else:
        cfg = EOTConfig(
            base_model=args.base_model,
            base_revision=args.base_revision,
            use_context=args.use_context,
            use_fvad=not args.no_fvad,
            freeze_encoder=args.freeze_encoder,
            fvad_weight=args.fvad_weight,
            pos_weight=sum(int(row["label"]) == 0 for row in tr_rows)
            / max(1, sum(int(row["label"]) == 1 for row in tr_rows)),
        )
        model = EOTModel(cfg)
    model = model.to(device)
    print(f"params total={model.num_params():,} trainable={model.num_params(True):,}")
    train_model = (
        torch.compile(model, mode=args.compile_mode)
        if args.compile_model else model
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    resume_batch = int(restored.get("next_batch", 0)) if restored else 0
    if restored is not None:
        generator_state = (
            restored.get("epoch_generator_state")
            if resume_batch > 0
            else restored.get("loader_generator_state")
        )
        if generator_state is not None:
            loader_generator.set_state(generator_state)
    pin_memory = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    common_loader = {
        "collate_fn": collate,
        "num_workers": args.workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        # Recreate workers from the checkpointed pre-iterator generator state each
        # epoch. This makes a mid-epoch restart replay the skipped augmentations.
        "persistent_workers": False,
    }
    # W&B starts a background async manager. Spawned workers avoid inheriting
    # that live process state, which is invalid after a POSIX fork.
    if args.workers > 0 and args.wandb_project:
        common_loader["multiprocessing_context"] = "spawn"
    tr = DataLoader(
        MinedDataset(tr_rows, telephony_prob=args.telephony_prob, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
        **common_loader,
    )
    dv = DataLoader(
        MinedDataset(dv_rows, seed=args.seed + 1),
        batch_size=args.batch_size * 2,
        shuffle=False,
        **common_loader,
    )

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    use_fused_adamw = device.type == "cuda" if args.fused_adamw is None else args.fused_adamw
    if use_fused_adamw and device.type != "cuda":
        raise ValueError("fused AdamW requires a CUDA device")
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay, fused=use_fused_adamw
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if restored is not None:
        if "optimizer_state_dict" not in restored:
            raise ValueError("resume checkpoint lacks optimizer state")
        optimizer.load_state_dict(restored["optimizer_state_dict"])
        _optimizer_to(optimizer, device)
        scaler.load_state_dict(restored.get("scaler_state_dict", {}))

    steps_per_epoch = math.ceil(len(tr_rows) / args.batch_size)
    planned_total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        planned_total_steps = min(planned_total_steps, args.max_steps)
    if restored is not None and restored.get("planned_total_steps") not in (None, planned_total_steps):
        raise ValueError(
            "resume plan changed total optimizer steps: "
            f"checkpoint={restored['planned_total_steps']}, requested={planned_total_steps}"
        )
    warm_steps = int(args.warmup_ratio * planned_total_steps)

    def lr_at(step: int) -> float:
        if step < warm_steps:
            return args.lr * step / max(1, warm_steps)
        progress = (step - warm_steps) / max(1, planned_total_steps - warm_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    start_epoch = int(restored.get("next_epoch", 0)) if restored else 0
    step = int(restored.get("global_step", 0)) if restored else 0
    best_auc = restored.get("best_auc") if restored else None
    has_best = bool(restored.get("has_best", False)) if restored else False
    if restored is not None:
        restore_rng_state(restored.get("rng_state"))

    history_path = out_dir / "history.jsonl"
    history = read_jsonl_records(history_path, repair_trailing=True)
    source_identity_path = Path(os.environ.get("EOT_SOURCE_IDENTITY", "/opt/eot/SOURCE_SHA256"))
    runtime = {
        "event": "resume" if restored else "run_start",
        "time_unix": time.time(),
        "resume_from": str(resume_path) if resume_path else None,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "image_ref": os.environ.get("EOT_IMAGE_REF"),
        "source_sha256": source_identity_path.read_text().strip() if source_identity_path.is_file() else None,
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("eot", "numpy", "torch", "transformers")
        },
        "args": vars(args),
        "split": diagnostics,
        "context_coverage": {"train": train_context_coverage, "dev": dev_context_coverage},
        "samples_sha256": samples_sha256,
        "performance": {
            "compile_model": args.compile_model,
            "compile_mode": args.compile_mode,
            "fused_adamw": use_fused_adamw,
        },
    }
    provenance_path = out_dir / "provenance.jsonl"
    append_jsonl_record(provenance_path, runtime)
    atomic_write_json(out_dir / "provenance.json", runtime)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_path = out_dir / "wandb.json"
        if wandb_path.exists():
            wandb_metadata = json.loads(wandb_path.read_text())
            wandb_run_id = str(wandb_metadata["run_id"])
        else:
            wandb_run_id = secrets.token_hex(4)
            wandb_metadata = {
                "run_id": wandb_run_id,
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "name": args.wandb_name or out_dir.name,
                "group": args.wandb_group,
            }
            atomic_write_json(wandb_path, wandb_metadata)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or out_dir.name,
            group=args.wandb_group,
            id=wandb_run_id,
            resume="allow",
            mode=args.wandb_mode,
            job_type="train",
            config=json.loads(json.dumps(runtime, default=str)),
        )

    t0 = time.time()
    completed_epoch = False
    for epoch in range(start_epoch, args.epochs):
        if args.max_steps is not None and step >= args.max_steps:
            break
        tr.dataset.set_epoch(epoch)
        model.train()
        if cfg.freeze_encoder:
            model.encoder.eval()
        running_loss = 0.0
        steps_this_log = 0
        # Save the pre-iterator generator state. Replaying it and skipping
        # next_batch recreates the same sample order and worker augmentations.
        epoch_generator_state = loader_generator.get_state()
        skip_batches = resume_batch if epoch == start_epoch else 0
        for batch_index, batch in enumerate(tr):
            if batch_index < skip_batches:
                continue
            if args.max_steps is not None and step >= args.max_steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            b = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = train_model(
                    b["input_features"], b["context_ids"], b["labels"], b["fvad"], b["fvad_mask"]
                )
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(output["loss"].detach())
            steps_this_log += 1
            step += 1
            if step % args.checkpoint_every == 0:
                atomic_torch_save(
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        next_epoch=epoch,
                        next_batch=batch_index + 1,
                        global_step=step,
                        best_auc=best_auc,
                        has_best=has_best,
                        args=args,
                        loader_generator=loader_generator,
                        epoch_generator_state=epoch_generator_state,
                        diagnostics=diagnostics,
                        planned_total_steps=planned_total_steps,
                        samples_sha256=samples_sha256,
                    ),
                    checkpoints_dir / "last.pt",
                )
            if step % args.log_every == 0:
                mean_loss = running_loss / max(1, steps_this_log)
                print(
                    f"ep{epoch} step{step}/{planned_total_steps} "
                    f"loss={mean_loss:.4f} "
                    f"lr={lr_at(step):.2e} {time.time() - t0:.0f}s"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {"train/loss": mean_loss, "train/lr": lr_at(step), "train/epoch": epoch},
                        step=step,
                    )
                running_loss = 0.0
                steps_this_log = 0

        metrics, dev_scores = evaluate(model, dv, device, non_blocking=pin_memory)
        metrics.update(epoch=epoch, step=step, elapsed_s=time.time() - t0)
        is_best = should_save_best(metrics["auc"], best_auc, has_best=has_best)
        if is_best:
            best_auc = metrics["auc"]
            has_best = True
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            next_epoch=epoch + 1,
            next_batch=0,
            global_step=step,
            best_auc=best_auc,
            has_best=has_best,
            args=args,
            loader_generator=loader_generator,
            epoch_generator_state=None,
            diagnostics=diagnostics,
            planned_total_steps=planned_total_steps,
            samples_sha256=samples_sha256,
        )
        last_path = checkpoints_dir / "last.pt"
        if is_best:
            # Publish the new best before last.pt advertises its best_auc. If a
            # restart lands between these atomic writes it safely replays the epoch.
            best_path = checkpoints_dir / "best.pt"
            atomic_torch_save(payload, best_path)
            atomic_copy(best_path, out_dir / "model.pt")
            atomic_write_jsonl(
                out_dir / "dev_scores.jsonl",
                ({**meta, **score, "path": None} for score, meta in zip(dev_scores, dv_rows)),
            )
        atomic_torch_save(payload, last_path)
        history.append(metrics)
        append_jsonl_record(history_path, metrics)
        append_jsonl_record(provenance_path, {"event": "epoch_end", **metrics, "is_best": is_best})
        atomic_write_json(out_dir / "history.json", {"config": vars(args), "history": history})
        print(json.dumps(metrics, sort_keys=True))
        if wandb_run is not None:
            wandb_run.log(
                {
                    "dev/auc": metrics["auc"],
                    "dev/acc_at_0_5": metrics["acc@0.5"],
                    "dev/n": metrics["n"],
                    "dev/pos_rate": metrics["pos_rate"],
                    "epoch": epoch,
                },
                step=step,
            )
            wandb_run.summary["best_auc"] = best_auc
        completed_epoch = True
        resume_batch = 0

    last_path = checkpoints_dir / "last.pt"
    best_path = checkpoints_dir / "best.pt"
    if best_path.exists():
        # Re-publish on every clean exit. This closes the crash window between
        # writing best.pt and its stable public model.pt alias.
        atomic_copy(best_path, out_dir / "model.pt")
    elif last_path.exists():
        atomic_copy(last_path, best_path)
        atomic_copy(last_path, out_dir / "model.pt")
    if not completed_epoch and not last_path.exists():
        raise RuntimeError("training completed no epoch and produced no checkpoint")
    if wandb_run is not None:
        wandb_run.finish()
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-model", default="openai/whisper-tiny")
    parser.add_argument(
        "--base-revision",
        default="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        help="immutable Hugging Face revision for the starting Whisper weights",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.2)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--telephony-prob", type=float, default=0.3)
    parser.add_argument("--fvad-weight", type=float, default=0.5)
    parser.add_argument("--use-context", action="store_true")
    parser.add_argument("--min-context-coverage", type=float, default=0.5)
    parser.add_argument("--no-fvad", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true", help="ablation only")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--fused-adamw", action=argparse.BooleanOptionalAction, default=None,
        help="use fused AdamW; defaults to enabled on CUDA and disabled elsewhere",
    )
    parser.add_argument("--compile-model", action="store_true", help="compile the training forward with torch.compile")
    parser.add_argument(
        "--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--wandb-project", default=None, help="enable W&B logging in this project")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="atomically update checkpoints/last.pt every N optimizer steps and at each epoch",
    )
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="resume from PATH, or from <out>/checkpoints/last.pt when passed without a value",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    print(train(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
