"""Fine-tune a registered audio-encoder backbone (LoRA, top-k layers, full or frozen) + pool + head on a samples manifest.

    uv run eot-train-backbone --samples data/mined/samples.jsonl --out runs/cohere_lora --seed 111 --split-seed 17 \\
        --epochs 1 --max-steps 2924 --batch-size 32 --micro-batch 8 --lr-encoder 2e-5 --lr-head 1e-3 \\
        --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --window-s 4 [--wandb-project ...]

Recipe (L026 R3 mirrored; L031 spec): rows filtered with ``filter_time_to_onset``; ``grouped_split`` by source
(``--dev-frac`` 0.15, ``--split-seed``); ``pos_weight`` = neg / pos of the train split; room tone and telephony
augmentation (0.3) as the Whisper trainer; AdamW (fp32 master weights and states) with weight decay 0.01, one
learning rate for the encoder and one for the head, warm-up 20 % + cosine over the planned updates, gradient
clipping 1.0; bf16 autocast; gradient checkpointing on trainable layers; gradient accumulation to the effective
batch; ``ceil(n_train / batch)`` updates per pass with a partial last batch. The end-of-pass model is published as
``model.pt`` (full) and ``delta.pt`` (trainable tensors only; see ``eot.modeling.backbone_model``) with no dev
selection. ``checkpoints/last.pt`` every ``--checkpoint-every`` updates lets a run resume exactly: the epoch
permutation and the loader's worker seeds come from one generator seeded by (seed, epoch), every per-row augmentation
by (seed, epoch, id), and the checkpoint carries the global RNG state that drives dropout.

The L031 C2 arm (Cohere Transcribe, LoRA r=16 / alpha 32 / dropout 0.05 on all 48 layers, 4 s window, three seeds)
reached 636.2 ms official delay@5 % but 4.1 s HTTP p95 on CPU at concurrency 8: a teacher, not a serving model.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eot.data.dataset import WaveformDataset, collate
from eot.data.splits import grouped_split
from eot.io import append_jsonl_record, atomic_write_json, atomic_write_jsonl, read_jsonl_records, read_samples, seed_from_key, sha256_file
from eot.metrics import roc_auc
from eot.modeling.backbone_model import (
    FROZEN_DTYPES,
    BackboneEOTConfig,
    BackboneEOTModel,
    base_weights_sha256,
    save_checkpoint,
    save_delta_checkpoint,
)
from eot.modeling.training_utils import (
    DATA_FILTER_VERSION,
    HelpFormatter,
    PhaseTimer,
    WandbLogger,
    atomic_torch_save,
    capture_rng_state,
    cosine_scale,
    environment_provenance,
    filter_time_to_onset,
    mean_logged_loss,
    pick_device,
    resolve_resume,
    restore_rng_state,
    seed_everything,
    seed_worker,
    source_digest,
    validate_binary_rows,
    validate_common_args,
    validate_resume_args,
    warmup_steps,
)

# v1 spans two sampler/RNG implementations without identifying which one wrote it.
# Reject it rather than silently changing sample order or the dropout RNG stream.
RESUME_FORMAT = "eot-backbone-resume-v2"
# Arguments whose change on resume would alter samples, gradients or the LR schedule.
PROTECTED_ARGS = (
    "seed",
    "split_seed",
    "lr_encoder",
    "lr_head",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "dev_frac",
    "telephony_prob",
    "frozen_dtype",
    "unfreeze_top_k",
    "freeze_encoder",
    "layer",
    "window_s",
    "input_s",
    "batch_size",
    "micro_batch",
    "epochs",
    "max_steps",
    "warmup_ratio",
    "weight_decay",
    "clip_grad",
    "backbone",
    "repo",
    "revision",
)


def gpu_memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30,
        "device_free_gb": free / 2**30,
        "device_total_gb": total / 2**30,
    }


@torch.no_grad()
def evaluate(model: BackboneEOTModel, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype) -> tuple[dict, list[dict]]:
    """AUC / accuracy over ``loader`` under autocast, plus one ``{"label", "p_eot"}`` row per dev example in loader order."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
            out = model(batch["wave"].to(device, non_blocking=True))
        ys.append(batch["labels"].numpy())
        ps.append(out["p_eot"].float().cpu().numpy())
    if not ys:
        raise ValueError("development loader is empty")
    y, p = np.concatenate(ys), np.concatenate(ps)
    rows = [{"label": int(a), "p_eot": float(q)} for a, q in zip(y, p)]
    return {"auc": roc_auc(y, p), "acc@0.5": float(((p > 0.5) == (y == 1)).mean()), "n": int(len(y)), "pos_rate": float(y.mean())}, rows


def build_config(args: argparse.Namespace, pos_weight: float) -> BackboneEOTConfig:
    return BackboneEOTConfig(
        backbone=args.backbone,
        repo=args.repo,
        revision=args.revision,
        layer=args.layer,
        window_s=args.window_s,
        input_s=args.input_s,
        freeze_encoder=args.freeze_encoder,
        unfreeze_top_k=args.unfreeze_top_k,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pos_weight=pos_weight,
        frozen_dtype=args.frozen_dtype,
    )


def validate_args(args: argparse.Namespace) -> None:
    validate_common_args(args, learning_rates=("lr_encoder", "lr_head"))
    if args.micro_batch <= 0 or args.batch_size % args.micro_batch:
        raise ValueError("--batch-size must be a positive multiple of --micro-batch")
    if args.lora_r < 0:
        raise ValueError("--lora-r must be >= 0")
    if args.eval_batch <= 0:
        raise ValueError("--eval-batch must be positive")


def train(args: argparse.Namespace) -> Path:
    timer = PhaseTimer()
    validate_args(args)
    seed_everything(args.seed)
    device = pick_device(args.device, require_cuda=args.require_cuda)
    out_dir = Path(args.out)
    ck_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(exist_ok=True)
    resume_path = resolve_resume(args.resume, out_dir)
    if resume_path is None and any(path.exists() for path in (ck_dir / "last.pt", out_dir / "model.pt", out_dir / "history.jsonl")):
        raise FileExistsError(f"{out_dir} already contains a run; pass --resume or choose a new --out")
    restored = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path is not None else None
    if restored is not None:
        if restored.get("format") != RESUME_FORMAT:
            raise ValueError(
                f"incompatible resume checkpoint format {restored.get('format')!r}; expected {RESUME_FORMAT}. "
                "Resume with the exact original training code and environment; changing the format tag "
                "does not preserve sample order or dropout RNG."
            )
        validate_resume_args(restored, args, protected=PROTECTED_ARGS)

    samples_sha256 = sha256_file(args.samples)
    rows, tto_filter = filter_time_to_onset(read_samples(args.samples))
    print(json.dumps({"time_to_onset_filter": tto_filter}, sort_keys=True), flush=True)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    tr_rows, dv_rows, diagnostics = grouped_split(rows, dev_frac=args.dev_frac, seed=split_seed, return_diagnostics=True)
    validate_binary_rows(tr_rows, "train")
    validate_binary_rows(dv_rows, "dev")
    print(json.dumps({"split": diagnostics, "device": str(device)}, sort_keys=True), flush=True)
    n_pos = sum(int(r["label"]) == 1 for r in tr_rows)
    pos_weight = (len(tr_rows) - n_pos) / max(1, n_pos)

    cfg = build_config(args, pos_weight)
    t_model = time.perf_counter()
    torch.manual_seed(args.seed)  # head and LoRA initialisation depend only on the seed
    model = BackboneEOTModel(cfg, pretrained=True).to(device)
    model.set_gradient_checkpointing(not args.no_grad_checkpointing and model.backbone.first_trainable is not None)
    print(
        json.dumps(
            {
                "trainable": model.trainable,
                "model_build_s": round(time.perf_counter() - t_model, 1),
                "grad_checkpointing": model.backbone.grad_checkpointing,
            }
        ),
        flush=True,
    )
    enc_params, head_params = model.encoder_parameters(), model.head_parameters()
    groups = [{"params": head_params, "lr": args.lr_head, "base_lr": args.lr_head, "name": "head"}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": args.lr_encoder, "base_lr": args.lr_encoder, "name": "encoder"})
    optimizer = torch.optim.AdamW(groups, lr=args.lr_head, weight_decay=args.weight_decay, fused=device.type == "cuda")
    all_trainable = enc_params + head_params

    steps_per_epoch = math.ceil(len(tr_rows) / args.batch_size)
    planned = args.epochs * steps_per_epoch if args.max_steps is None else min(args.epochs * steps_per_epoch, args.max_steps)

    step, start_epoch, resume_index = 0, 0, 0
    if restored is not None:
        if restored["samples_sha256"] != samples_sha256 or restored["planned"] != planned:
            raise ValueError("resume plan differs: samples manifest or planned updates changed")
        params = dict(model.named_parameters())
        for name, tensor in restored["trainable_state"].items():
            params[name].data.copy_(tensor.to(params[name].device, params[name].dtype))
        optimizer.load_state_dict(restored["optimizer_state_dict"])
        step, start_epoch, resume_index = int(restored["step"]), int(restored["epoch"]), int(restored["samples_done"])
        restore_rng_state(restored.get("rng_state"))
        print(f"resumed at step {step} epoch {start_epoch} sample {resume_index}", flush=True)

    amp_dtype = torch.bfloat16  # frozen bf16 tensors mix with fp32 master weights the same way on CUDA and (tests) CPU
    train_ds = WaveformDataset(tr_rows, telephony_prob=args.telephony_prob, seed=args.seed, input_s=args.input_s)
    if args.max_dev_rows:  # smoke tests only: the real runs evaluate every dev row
        dv_rows = dv_rows[: args.max_dev_rows]
    dev_ds = WaveformDataset(dv_rows, seed=args.seed + 1, input_s=args.input_s)
    loader_kw = {
        "collate_fn": collate,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }
    dv = DataLoader(dev_ds, batch_size=args.eval_batch, shuffle=False, **loader_kw)
    snap = model.backbone.snapshot_dir(cfg.repo, cfg.revision)
    init_sha = base_weights_sha256(snap)
    args_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    runtime = {
        "event": "resume" if restored else "run_start",
        "time_unix": time.time(),
        "resume_from": str(resume_path) if resume_path else None,
        **environment_provenance(device),
        "source_sha256": source_digest(Path(__file__).parent),
        "init_weights_sha256": init_sha,
        "init_weights_repo": cfg.repo,
        "init_weights_revision": cfg.revision,
        "snapshot_dir": str(snap),
        "remote_code": {f.name: sha256_file(f) for f in sorted(snap.glob("*.py"))},
        "args": args_json,
        "split": diagnostics,
        "samples_sha256": samples_sha256,
        "data_filter_version": DATA_FILTER_VERSION,
        **tto_filter,
        "legacy_rows": {"train": train_ds.legacy_counts, "dev": dev_ds.legacy_counts},
        "cfg": asdict(cfg),
        "trainable": model.trainable,
        "recipe": {
            "effective_batch": args.batch_size,
            "micro_batch": args.micro_batch,
            "accumulation": args.batch_size // args.micro_batch,
            "planned_updates": planned,
            "steps_per_epoch": steps_per_epoch,
            "warmup_updates": warmup_steps(planned, args.warmup_ratio),
            "lr_encoder": args.lr_encoder,
            "lr_head": args.lr_head,
            "weight_decay": args.weight_decay,
            "clip_grad_norm": args.clip_grad,
            "autocast": str(amp_dtype),
            "master_weights": "fp32",
            "optimizer": "AdamW, fp32 states",
            "grad_checkpointing": model.backbone.grad_checkpointing,
            "frozen_dtype": cfg.frozen_dtype,
            "pos_weight": pos_weight,
            "telephony_prob": args.telephony_prob,
        },
    }
    append_jsonl_record(out_dir / "provenance.jsonl", runtime)
    atomic_write_json(out_dir / "provenance.json", runtime)
    wandb = WandbLogger(
        args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        reinit=True,
        config={
            "args": args_json,
            "trainable": model.trainable,
            "recipe": runtime["recipe"],
            "cfg": {k: v for k, v in asdict(cfg).items() if k not in ("backbone_config", "frontend_config")},
        },
    )
    history_path = out_dir / "history.jsonl"
    history = read_jsonl_records(history_path, repair_trailing=True) if history_path.exists() else []

    def save_resume(epoch: int, samples_done: int) -> None:
        with timer.phase("checkpoint"):
            atomic_torch_save(
                {
                    "format": RESUME_FORMAT,
                    "step": step,
                    "epoch": epoch,
                    "samples_done": samples_done,
                    "planned": planned,
                    "samples_sha256": samples_sha256,
                    "trainable_state": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": capture_rng_state(),
                    "args": args_json,
                },
                ck_dir / "last.pt",
            )

    t0 = time.time()
    timer.initialized()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log_t, log_step, log_losses, grad_norms = time.time(), step, [], []
    for epoch in range(start_epoch, args.epochs):
        if step >= planned:
            break
        train_ds.set_epoch(epoch)
        # One generator per (seed, epoch) drives both the permutation and the loader's worker seeds, so the
        # global torch RNG (dropout) is consumed only by the model and a restored run replays the same stream.
        epoch_generator = torch.Generator().manual_seed(seed_from_key(f"perm:{args.seed}:{epoch}"))
        perm = torch.randperm(len(train_ds), generator=epoch_generator).tolist()
        first = resume_index if epoch == start_epoch else 0
        tr = DataLoader(
            train_ds,
            batch_size=args.micro_batch,
            sampler=perm[first:],
            shuffle=False,
            drop_last=False,
            generator=epoch_generator,
            **loader_kw,
        )
        model.train()
        if model.backbone.first_trainable is None:
            model.backbone.eval()
        accumulated, seen, n_epoch = 0, first, len(train_ds)
        optimizer.zero_grad(set_to_none=True)
        with timer.phase("train"):
            for batch in tr:
                if step >= planned:
                    break
                m = batch["wave"].shape[0]
                # Samples of this update: the batch boundary in the permutation; the last batch of the epoch is partial.
                batch_start = (seen // args.batch_size) * args.batch_size
                batch_n = min(args.batch_size, n_epoch - batch_start)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
                    out = model(
                        batch["wave"].to(device, non_blocking=True),
                        batch["labels"].to(device, non_blocking=True),
                        weight=batch["weight"].to(device, non_blocking=True),
                    )
                (out["loss"] * (m / batch_n)).backward()
                log_losses.append(out["loss"].detach() * (m / batch_n))
                accumulated += m
                seen += m
                if accumulated < batch_n:
                    continue
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * cosine_scale(step, planned, args.warmup_ratio)
                grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, args.clip_grad) if args.clip_grad else torch.tensor(0.0)
                grad_norms.append(grad_norm.detach())  # stays on device: a float() here would sync the GPU every update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accumulated = 0
                if step % args.checkpoint_every == 0 and step < planned:
                    save_resume(epoch, seen)
                if step % args.log_every == 0 or step == planned:
                    now = time.time()
                    updates_per_s = (step - log_step) / max(1e-9, now - log_t)
                    loss = float(torch.stack(log_losses).sum().item() / max(1, step - log_step))
                    mean_grad_norm = mean_logged_loss(grad_norms)
                    memory = gpu_memory(device)
                    lrs = " ".join(f"{g['name']}={g['lr']:.2e}" for g in optimizer.param_groups)
                    print(
                        f"ep{epoch} step{step}/{planned} loss={loss:.4f} lr={lrs} gnorm={mean_grad_norm:.2f} {updates_per_s:.2f} upd/s "
                        f"peak={memory.get('peak_reserved_gb', 0):.1f}GB eta={(planned - step) / max(1e-9, updates_per_s) / 60:.0f}min {now - t0:.0f}s",
                        flush=True,
                    )
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/grad_norm": mean_grad_norm,
                            "train/updates_per_s": updates_per_s,
                            "train/peak_reserved_gb": memory.get("peak_reserved_gb", 0),
                            **{f"train/lr_{g['name']}": g["lr"] for g in optimizer.param_groups},
                        },
                        step=step,
                    )
                    log_t, log_step, log_losses, grad_norms = now, step, [], []
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        with timer.phase("validation"):
            metrics, dev_scores = evaluate(model, dv, device, amp_dtype)
        metrics.update(
            epoch=epoch,
            step=step,
            elapsed_s=time.time() - t0,
            samples_seen=seen,
            **gpu_memory(device),
            updates_per_s=step / max(1e-9, timer.seconds("train")),
        )
        with timer.phase("checkpoint"):
            save_checkpoint(model, out_dir / "model.pt")
            save_delta_checkpoint(model, out_dir / "delta.pt", base_weights_sha256=init_sha)
            atomic_write_jsonl(out_dir / "dev_scores.jsonl", ({**meta, **score, "path": None} for score, meta in zip(dev_scores, dv_rows)))
        history.append(metrics)
        append_jsonl_record(history_path, metrics)
        append_jsonl_record(out_dir / "provenance.jsonl", {"event": "epoch_end", **metrics})
        atomic_write_json(out_dir / "history.json", {"config": args_json, "history": history})
        print(json.dumps(metrics, sort_keys=True), flush=True)
        wandb.log({"dev/auc": metrics["auc"], "dev/acc_at_0_5": metrics["acc@0.5"], "epoch": epoch}, step=step)
        resume_index = 0
    timing = {
        **timer.finish(),
        "global_step": step,
        "planned_updates": planned,
        "scope": "this invocation",
        "resume_from": str(resume_path) if resume_path else None,
        **{f"final_{k}": v for k, v in gpu_memory(device).items()},
    }
    timing["updates_per_s"] = step / max(1e-9, timing["train_s"])
    atomic_write_json(out_dir / "timing.json", timing)
    wandb.finish(
        {
            "cost/peak_vram_reserved_gb": timing.get("final_peak_reserved_gb"),
            "cost/train_wall_s": timing["train_s"],
            "cost/trainer_elapsed_s": timing["trainer_elapsed_s"],
            "cost/updates": step,
            "cost/updates_per_s": timing["updates_per_s"],
            "cost/trainable_params": model.trainable["trainable_total"],
            "cost/encoder_trainable_params": model.trainable["encoder_trainable"],
            "dev/auc_final": history[-1]["auc"] if history else None,
        }
    )
    if not args.keep_resume:
        shutil.rmtree(ck_dir, ignore_errors=True)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    ap.add_argument("--samples", type=Path, required=True, help="samples.jsonl manifest (eot-mine output)")
    ap.add_argument("--out", type=Path, required=True, help="run directory (model.pt, delta.pt, history, provenance, checkpoints/)")
    ap.add_argument("--backbone", default=BackboneEOTConfig.backbone, help="registered backbone name (eot.modeling.backbones)")
    ap.add_argument("--repo", default=BackboneEOTConfig.repo, help="Hugging Face repo of the pretrained encoder")
    ap.add_argument("--revision", default=BackboneEOTConfig.revision, help="immutable Hugging Face revision of the encoder")
    ap.add_argument("--layer", type=int, default=-1, help="-1 = last layer; k = after k layers")
    ap.add_argument("--window-s", type=float, default=4.0, help="trailing seconds encoded (crop inside the model)")
    ap.add_argument("--input-s", type=float, default=8.0, help="shared audio window fed to the model")
    ap.add_argument("--freeze-encoder", action="store_true", help="control arm: train the head only")
    ap.add_argument("--unfreeze-top-k", type=int, default=None, help="train only the top K used encoder layers (default: all)")
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank on the backbone's target linears (0 = no adapters)")
    ap.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA scaling numerator (scale = alpha / r)")
    ap.add_argument("--lora-dropout", type=float, default=0.0, help="dropout on the LoRA input path")
    ap.add_argument("--frozen-dtype", default="bfloat16", choices=FROZEN_DTYPES, help="storage dtype of frozen encoder tensors")
    ap.add_argument("--no-grad-checkpointing", action="store_true", help="keep every trainable layer's activations in memory")
    ap.add_argument(
        "--split-seed", type=int, default=None, help="keep the data split fixed while replicating training seeds (default: --seed)"
    )
    ap.add_argument("--seed", type=int, default=0, help="training seed (initialisation, sample order, augmentation)")
    ap.add_argument("--dev-frac", type=float, default=0.15, help="fraction of rows held out as the grouped dev split")
    ap.add_argument("--epochs", type=int, default=1, help="passes over the training split")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer updates (also shortens the LR schedule)")
    ap.add_argument("--batch-size", type=int, default=32, help="effective batch (one optimizer update)")
    ap.add_argument("--micro-batch", type=int, default=8, help="rows per forward pass; --batch-size must be a multiple")
    ap.add_argument("--eval-batch", type=int, default=64, help="rows per development batch")
    ap.add_argument("--max-dev-rows", type=int, default=None, help="smoke tests only: evaluate a prefix of the dev split")
    ap.add_argument("--lr-encoder", type=float, default=2e-5, help="peak learning rate of the encoder (LoRA / unfrozen) parameters")
    ap.add_argument("--lr-head", type=float, default=1e-3, help="peak learning rate of the pool + head")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    ap.add_argument("--warmup-ratio", type=float, default=0.2, help="fraction of planned updates spent in linear warm-up")
    ap.add_argument("--clip-grad", type=float, default=1.0, help="gradient-norm clipping threshold (0 disables)")
    ap.add_argument("--telephony-prob", type=float, default=0.3, help="probability of the telephony augmentation per training row")
    ap.add_argument("--workers", type=int, default=6, help="DataLoader worker processes")
    ap.add_argument("--log-every", type=int, default=20, help="print and log the running loss every N updates")
    ap.add_argument("--checkpoint-every", type=int, default=500, help="write checkpoints/last.pt every N updates")
    ap.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from PATH, or from <out>/checkpoints/last.pt when passed without a value",
    )
    ap.add_argument("--keep-resume", action="store_true", help="keep checkpoints/last.pt after a clean finish")
    ap.add_argument("--device", default=None, help="torch device (default: cuda, then mps, then cpu)")
    ap.add_argument("--require-cuda", action="store_true", help="fail instead of silently training on CPU")
    ap.add_argument("--wandb-project", default=None, help="enable W&B logging in this project")
    ap.add_argument("--wandb-group", default=None, help="W&B run group")
    ap.add_argument("--wandb-name", default=None, help="W&B run name")
    return ap


def main(argv: list[str] | None = None) -> None:
    print(train(build_parser().parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()
