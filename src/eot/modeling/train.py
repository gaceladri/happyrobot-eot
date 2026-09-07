"""Full fine-tuning of a Whisper encoder detector with restart-safe epoch checkpoints.

The audio-only + FVAD path remains the default. Context is opt-in and guarded by measured
coverage so an all-padding context branch can never be shipped accidentally.

Two options reproduce the L040 recipe (Cohere-teacher distillation + tail texture):

* ``--tail-texture-prob P`` rewrites the pause tail of training rows with a label-independent
  texture (``eot.audio.tail_texture``); dev audio is never augmented.
* ``--teacher-cache DIR`` trains on cached teacher logits (``eot-distill-cache``) with the Bernoulli
  KL of ``eot.modeling.distill`` at ``--kd-temperature`` instead of the hard labels. The cache is
  bound to one (seed, epoch, augmentation) view, so distillation runs are single-pass; every item
  re-checks the waveform hash the teacher saw. Dev metrics report the hard-label BCE as well.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eot.data.dataset import MinedDataset, collate
from eot.data.splits import grouped_split
from eot.data.teacher_cache import DistillationDataset
from eot.io import (
    append_jsonl_record,
    atomic_replace,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl_records,
    read_samples,
    sha256_file,
)
from eot.metrics import roc_auc
from eot.modeling.distill import DEFAULT_KD_TEMPERATURE
from eot.modeling.model import EOTConfig, EOTModel, load_checkpoint
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
)


def publish_checkpoint(source: Path, destination: Path) -> None:
    """Publish an immutable checkpoint without copying its tensor bytes again.

    Checkpoints are always replaced atomically, never overwritten in place, so a
    hard link keeps the previous best intact when last.pt advances. Fall back to
    a copy on filesystems that cannot link the two paths.
    """

    def write(tmp: Path) -> None:
        # A killed publisher may leave a linked temp file; never copy over its inode.
        tmp.unlink(missing_ok=True)
        try:
            os.link(source, tmp)
        except OSError:
            shutil.copyfile(source, tmp)

    atomic_replace(destination, write)


def should_save_best(current: float, best: float | None, *, has_best: bool) -> bool:
    """Always select the first epoch, then prefer finite AUC improvements."""
    if not has_best:
        return True
    if not math.isfinite(current):
        return False
    return best is None or not math.isfinite(best) or current > best


@torch.no_grad()
def evaluate(model: EOTModel, loader: DataLoader, device: torch.device, *, non_blocking: bool = False) -> tuple[dict, list[dict]]:
    model.eval()
    ys, ps, rows = [], [], []
    hard_bce_sum, hard_weight_sum = 0.0, 0.0
    pos_weight = torch.as_tensor(model.cfg.pos_weight, dtype=torch.float32)
    for batch in loader:
        # Targets stay on CPU; validation only consumes predictions, not losses.
        out = model(
            batch["input_features"].to(device, non_blocking=non_blocking),
            batch["context_ids"].to(device, non_blocking=non_blocking),
        )
        logits = out["logit"].float().cpu()
        # The hard-label loss is reported for every run (a distilled student never sees it in training).
        hard = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["labels"].float(), pos_weight=pos_weight, reduction="none"
        )
        weights = batch["weight"].float()
        hard_bce_sum += float((hard * weights).sum())
        hard_weight_sum += float(weights.sum())
        labels = batch["labels"].cpu().numpy()
        probabilities = out["p_eot"].cpu().numpy()
        fvad = out["p_fvad"].cpu().tolist() if "p_fvad" in out else None
        ys.append(labels)
        ps.append(probabilities)
        for index in range(len(labels)):
            rows.append(
                {
                    "label": int(labels[index]),
                    "p_eot": float(probabilities[index]),
                    "p_fvad": fvad[index] if fvad is not None else None,
                }
            )
    if not ys:
        raise ValueError("development loader is empty")
    y, p = np.concatenate(ys), np.concatenate(ps)
    acc = float(((p > 0.5) == (y == 1)).mean())
    metrics = {"auc": roc_auc(y, p), "acc@0.5": acc, "n": int(len(y)), "pos_rate": float(y.mean())}
    if hard_weight_sum > 0:
        metrics["hard_bce"] = hard_bce_sum / hard_weight_sum
    return metrics, rows


def _context_coverage(rows: list[dict]) -> float:
    return sum(bool(str(row.get("agent_text", "")).strip()) for row in rows) / max(1, len(rows))


def exclude_training_ids(train: list[dict], dev: list[dict], ids: list[str]) -> list[dict]:
    """Drop listed sample ids from the training rows only, after the frozen split.

    Data ablations (e.g. removing ambiguous labels) must never touch the development set, and a
    matched random-exclusion control needs the same interface.
    """
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate training exclusion IDs")
    excluded = set(ids)
    if excluded & {r["id"] for r in dev}:
        raise ValueError("cannot exclude development examples")
    if not excluded <= {r["id"] for r in train}:
        raise ValueError("unknown training exclusion IDs")
    kept = [r for r in train if r["id"] not in excluded]
    validate_binary_rows(kept, "filtered train")
    return kept


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
        "format_version": 3,
        "data_filter_version": DATA_FILTER_VERSION,
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


def _read_resume_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class TrainingRows:
    """The filtered, split manifest a run trains on, plus what provenance records about it."""

    samples_sha256: str
    train: list[dict]
    dev: list[dict]
    diagnostics: dict
    tto_filter: dict
    context_coverage: dict  # {"train": fraction of rows with agent text, "dev": ...}


def _prepare_rows(args: argparse.Namespace, restored: dict | None) -> TrainingRows:
    """Hash, filter and split ``--samples``; apply ``--exclude-train-ids``; check the context coverage guard."""
    samples_sha256 = sha256_file(args.samples)
    if restored is not None and restored.get("samples_sha256") not in (None, samples_sha256):
        raise ValueError("samples manifest changed since the resume checkpoint was written")
    # Preserve the original manifest; exclude cuts after the onset, clamp float noise, and record
    # both counts in run provenance (see filter_time_to_onset).
    rows, tto_filter = filter_time_to_onset(read_samples(args.samples))
    # Filter version 1 dropped every negative row; version 2 keeps the float-noise ones, so the
    # sample set differs from an older checkpoint's exactly when something was clamped.
    if (
        restored is not None
        and tto_filter["clamped_float_noise_time_to_onset"]
        and restored.get("data_filter_version") != DATA_FILTER_VERSION
    ):
        raise ValueError("resume would change the training sample filter; use --init-checkpoint in a new run instead")
    print(json.dumps({"time_to_onset_filter": tto_filter}, sort_keys=True))
    if not rows:
        raise ValueError("samples manifest is empty")
    if {int(row["label"]) for row in rows} != {0, 1}:
        raise ValueError("samples manifest must contain both labels 0 and 1")
    split_seed = args.seed if args.split_seed is None else args.split_seed
    tr_rows, dv_rows, diagnostics = grouped_split(rows, dev_frac=args.dev_frac, seed=split_seed, return_diagnostics=True)
    if args.exclude_train_ids:
        before = len(tr_rows)
        tr_rows = exclude_training_ids(tr_rows, dv_rows, json.loads(args.exclude_train_ids.read_text()))
        diagnostics["training_exclusion"] = {
            "before": before,
            "after": len(tr_rows),
            "sha256": args.exclude_train_sha256,
            "dev_unchanged": True,
        }
    validate_binary_rows(tr_rows, "train")
    validate_binary_rows(dv_rows, "dev")
    coverage = {"train": _context_coverage(tr_rows), "dev": _context_coverage(dv_rows)}
    if args.use_context and min(coverage.values()) < args.min_context_coverage:
        raise ValueError(
            "--use-context requested but context coverage is too low: "
            f"train={coverage['train']:.3f}, dev={coverage['dev']:.3f}, required={args.min_context_coverage:.3f}"
        )
    return TrainingRows(samples_sha256, tr_rows, dv_rows, diagnostics, tto_filter, coverage)


def _build_model(args: argparse.Namespace, restored: dict | None, tr_rows: list[dict]) -> tuple[EOTConfig, EOTModel]:
    """The model to train: restored from a resume checkpoint, fine-tuned from ``--init-checkpoint``, or fresh Whisper weights."""
    if restored is not None:
        cfg_values = dict(restored["cfg"])
        cfg_values["horizons"] = tuple(cfg_values["horizons"])
        cfg = EOTConfig(**cfg_values)
        if cfg.use_context != args.use_context or cfg.use_fvad != (not args.no_fvad):
            raise ValueError("resume flags --use-context/--no-fvad must match the checkpoint")
        model = EOTModel(cfg, pretrained=False)
        model.load_state_dict(restored["state_dict"])
    elif args.init_checkpoint:
        model = load_checkpoint(args.init_checkpoint)
        cfg = model.cfg
        if cfg.use_fvad != (not args.no_fvad) or cfg.use_context != args.use_context:
            raise ValueError("initial checkpoint heads must match requested context/FVAD flags")
        cfg.freeze_encoder = args.freeze_encoder
        for p in model.encoder.parameters():
            p.requires_grad_(not cfg.freeze_encoder)
        model.encoder.embed_positions.weight.requires_grad_(False)
    else:
        cfg = EOTConfig(
            base_model=args.base_model,
            base_revision=args.base_revision,
            use_context=args.use_context,
            use_fvad=not args.no_fvad,
            freeze_encoder=args.freeze_encoder,
            fvad_weight=args.fvad_weight,
            pos_weight=sum(int(row["label"]) == 0 for row in tr_rows) / max(1, sum(int(row["label"]) == 1 for row in tr_rows)),
            normalize_audio=args.normalize_audio,
            **({"max_source_positions": args.max_source_positions} if args.max_source_positions is not None else {}),
        )
        model = EOTModel(cfg)
    return cfg, model


def _wandb_run_id(out_dir: Path, args: argparse.Namespace) -> str:
    """The W&B run id of this output directory, minted once and kept in ``wandb.json`` so a resume continues the run."""
    wandb_path = out_dir / "wandb.json"
    if wandb_path.exists():
        return str(json.loads(wandb_path.read_text())["run_id"])
    run_id = secrets.token_hex(4)
    atomic_write_json(
        wandb_path,
        {
            "run_id": run_id,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name or out_dir.name,
            "group": args.wandb_group,
        },
    )
    return run_id


def train(args: argparse.Namespace) -> Path:
    timer = PhaseTimer()
    validate_common_args(args, learning_rates=("lr",))
    if args.eval_batch is not None and args.eval_batch <= 0:
        raise ValueError("--eval-batch must be positive when set")
    if args.teacher_cache is not None:
        if args.epochs != 1:
            raise ValueError("--teacher-cache holds one augmentation epoch: distillation runs are single-pass (--epochs 1)")
        if not args.kd_temperature > 0:
            raise ValueError("--kd-temperature must be positive")
    if args.max_source_positions is not None and not 1 <= args.max_source_positions <= 400:
        raise ValueError("--max-source-positions must lie in [1, 400] (the 8 s interface holds 400 encoder positions)")

    args.exclude_train_sha256 = sha256_file(args.exclude_train_ids) if args.exclude_train_ids else None
    seed_everything(args.seed)
    device = pick_device(args.device, require_cuda=args.require_cuda)
    out_dir = Path(args.out)
    checkpoints_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    resume_path = resolve_resume(args.resume, out_dir)
    if resume_path is None and any(
        path.exists() for path in (checkpoints_dir / "last.pt", out_dir / "model.pt", out_dir / "history.jsonl")
    ):
        raise FileExistsError(f"{out_dir} already contains a run; pass --resume or choose a new --out")
    restored = _read_resume_checkpoint(resume_path) if resume_path is not None else None
    if restored is not None:
        validate_resume_args(restored, args)

    data = _prepare_rows(args, restored)
    tr_rows, dv_rows, diagnostics = data.train, data.dev, data.diagnostics
    print(json.dumps({"split": diagnostics, "device": str(device)}, sort_keys=True))
    cfg, model = _build_model(args, restored, tr_rows)
    model = model.to(device)
    print(f"params total={model.num_params():,} trainable={model.num_params(True):,}")
    train_model = torch.compile(model, mode=args.compile_mode) if args.compile_model else model

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    resume_batch = int(restored.get("next_batch", 0)) if restored else 0
    if restored is not None:
        generator_state = restored.get("epoch_generator_state") if resume_batch > 0 else restored.get("loader_generator_state")
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
    view = {
        "telephony_prob": args.telephony_prob,
        "seed": args.seed,
        "normalize_audio": cfg.normalize_audio,
        "tail_texture_prob": args.tail_texture_prob,
    }
    if args.teacher_cache is not None:
        train_dataset = DistillationDataset(tr_rows, args.teacher_cache, **view)
    else:
        train_dataset = MinedDataset(tr_rows, **view)
    dev_dataset = MinedDataset(dv_rows, seed=args.seed + 1, normalize_audio=cfg.normalize_audio)
    legacy_rows = {"train": train_dataset.legacy_counts, "dev": dev_dataset.legacy_counts}
    print(json.dumps({"legacy_rows": legacy_rows}, sort_keys=True))
    tr = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
        **common_loader,
    )
    dv = DataLoader(dev_dataset, batch_size=args.eval_batch or 2 * args.batch_size, shuffle=False, **common_loader)

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    use_fused_adamw = device.type == "cuda" if args.fused_adamw is None else args.fused_adamw
    if use_fused_adamw and device.type != "cuda":
        raise ValueError("fused AdamW requires a CUDA device")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=use_fused_adamw)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if restored is not None:
        if "optimizer_state_dict" not in restored:
            raise ValueError("resume checkpoint lacks optimizer state")
        optimizer.load_state_dict(restored["optimizer_state_dict"])  # moves the state to the parameters' device
        scaler.load_state_dict(restored.get("scaler_state_dict", {}))

    steps_per_epoch = math.ceil(len(tr_rows) / args.batch_size)
    planned_total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        planned_total_steps = min(planned_total_steps, args.max_steps)
    if restored is not None and restored.get("planned_total_steps") not in (None, planned_total_steps):
        raise ValueError(
            f"resume plan changed total optimizer steps: checkpoint={restored['planned_total_steps']}, requested={planned_total_steps}"
        )

    def lr_at(step: int) -> float:
        return args.lr * cosine_scale(step, planned_total_steps, args.warmup_ratio)

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
        **environment_provenance(device),
        "image_ref": os.environ.get("EOT_IMAGE_REF"),
        "source_sha256": source_identity_path.read_text().strip()
        if source_identity_path.is_file()
        else source_digest(Path(__file__).parent),
        "init_checkpoint_sha256": sha256_file(args.init_checkpoint) if args.init_checkpoint else None,
        "exclude_train_sha256": args.exclude_train_sha256,
        "args": vars(args),
        "split": diagnostics,
        "context_coverage": data.context_coverage,
        "samples_sha256": data.samples_sha256,
        **data.tto_filter,
        "legacy_rows": legacy_rows,
        "augmentation": {"telephony_prob": args.telephony_prob, "tail_texture_prob": args.tail_texture_prob},
        "distillation": None
        if args.teacher_cache is None
        else {
            "teacher_cache": str(args.teacher_cache),
            "kd_temperature": args.kd_temperature,
            "cache_fingerprint": train_dataset.metadata["fingerprint"],
            "teacher": train_dataset.metadata["identity"]["teacher"],
            "objective": "pure Bernoulli KL (T^2 scaled, row-weighted, no pos_weight, no hard-label term)",
        },
        "performance": {
            "trainer_performance_version": 1,
            "compile_model": args.compile_model,
            "compile_mode": args.compile_mode,
            "fused_adamw": use_fused_adamw,
        },
    }
    provenance_path = out_dir / "provenance.jsonl"
    append_jsonl_record(provenance_path, runtime)
    atomic_write_json(out_dir / "provenance.json", runtime)
    wandb = WandbLogger(
        args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or out_dir.name,
        group=args.wandb_group,
        id=_wandb_run_id(out_dir, args) if args.wandb_project else None,
        resume="allow",
        mode=args.wandb_mode,
        job_type="train",
        config=json.loads(json.dumps(runtime, default=str)),
    )

    t0 = time.time()
    timer.initialized()
    completed_epoch = False
    for epoch in range(start_epoch, args.epochs):
        if args.max_steps is not None and step >= args.max_steps:
            break
        tr.dataset.set_epoch(epoch)
        model.train()
        if cfg.freeze_encoder:
            model.encoder.eval()
        logged_losses = []
        # Save the pre-iterator generator state. Replaying it and skipping
        # next_batch recreates the same sample order and worker augmentations.
        epoch_generator_state = loader_generator.get_state()
        skip_batches = resume_batch if epoch == start_epoch else 0
        with timer.phase("train"):
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
                        b["input_features"],
                        b["context_ids"],
                        b["labels"],
                        b["fvad"],
                        b["fvad_mask"],
                        weight=b.get("weight"),
                        teacher_logits=b.get("teacher_logits"),
                        kd_temperature=args.kd_temperature,
                    )
                scaler.scale(output["loss"]).backward()
                scaler.unscale_(optimizer)
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                # Clone: compiled CUDA graphs can reuse the output's storage next step.
                logged_losses.append(output["loss"].detach().clone())
                step += 1
                if step % args.checkpoint_every == 0:
                    with timer.phase("checkpoint"):
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
                                samples_sha256=data.samples_sha256,
                            ),
                            checkpoints_dir / "last.pt",
                        )
                if step % args.log_every == 0:
                    mean_loss = mean_logged_loss(logged_losses)
                    print(f"ep{epoch} step{step}/{planned_total_steps} loss={mean_loss:.4f} lr={lr_at(step):.2e} {time.time() - t0:.0f}s")
                    wandb.log({"train/loss": mean_loss, "train/lr": lr_at(step), "train/epoch": epoch}, step=step)
                    logged_losses.clear()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        with timer.phase("validation"):
            metrics, dev_scores = evaluate(model, dv, device, non_blocking=pin_memory)
        metrics.update(epoch=epoch, step=step, elapsed_s=time.time() - t0)
        is_best = should_save_best(metrics["auc"], best_auc, has_best=has_best)
        if is_best:
            best_auc = metrics["auc"]
            has_best = True
        with timer.phase("checkpoint"):
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
                samples_sha256=data.samples_sha256,
            )
            last_path = checkpoints_dir / "last.pt"
            if is_best:
                # Publish the new best before last.pt advertises its best_auc. If a
                # restart lands between these atomic writes it safely replays the epoch.
                best_path = checkpoints_dir / "best.pt"
                atomic_torch_save(payload, best_path)
                publish_checkpoint(best_path, out_dir / "model.pt")
                atomic_write_jsonl(
                    out_dir / "dev_scores.jsonl",
                    ({**meta, **score, "path": None} for score, meta in zip(dev_scores, dv_rows)),
                )
                publish_checkpoint(best_path, last_path)
            else:
                atomic_torch_save(payload, last_path)
        history.append(metrics)
        append_jsonl_record(history_path, metrics)
        append_jsonl_record(provenance_path, {"event": "epoch_end", **metrics, "is_best": is_best})
        atomic_write_json(out_dir / "history.json", {"config": vars(args), "history": history})
        print(json.dumps(metrics, sort_keys=True))
        wandb.log(
            {
                "dev/auc": metrics["auc"],
                "dev/acc_at_0_5": metrics["acc@0.5"],
                "dev/n": metrics["n"],
                "dev/pos_rate": metrics["pos_rate"],
                **({"dev/hard_bce": metrics["hard_bce"]} if "hard_bce" in metrics else {}),
                "epoch": epoch,
            },
            step=step,
        )
        wandb.summarize({"best_auc": best_auc})
        completed_epoch = True
        resume_batch = 0

    last_path = checkpoints_dir / "last.pt"
    best_path = checkpoints_dir / "best.pt"
    with timer.phase("checkpoint"):
        if best_path.exists():
            # Re-publish on every clean exit. This closes the crash window between
            # writing best.pt and its stable public model.pt alias.
            publish_checkpoint(best_path, out_dir / "model.pt")
        elif last_path.exists():
            publish_checkpoint(last_path, best_path)
            publish_checkpoint(last_path, out_dir / "model.pt")
    if not completed_epoch and not last_path.exists():
        raise RuntimeError("training completed no epoch and produced no checkpoint")
    wandb.finish()
    atomic_write_json(
        out_dir / "timing.json",
        {
            **timer.finish(),
            "scope": "this invocation, trainer entry through final checkpoint publication and tracking finish",
            "resume_from": str(resume_path) if resume_path else None,
            "global_step": step,
        },
    )
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("--samples", type=Path, required=True, help="samples.jsonl manifest (eot-mine output)")
    parser.add_argument("--out", type=Path, required=True, help="run directory (model.pt, history, provenance, checkpoints/)")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="fine-tune existing EOT weights, with a fresh optimizer")
    parser.add_argument("--exclude-train-ids", type=Path, default=None, help="JSON id list removed from train only, after the frozen split")
    parser.add_argument(
        "--split-seed", type=int, default=None, help="keep the data split fixed while replicating training seeds (default: --seed)"
    )
    parser.add_argument("--base-model", default="openai/whisper-tiny", help="Hugging Face Whisper repo of the starting encoder")
    parser.add_argument(
        "--base-revision",
        default="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        help="immutable Hugging Face revision for the starting Whisper weights",
    )
    parser.add_argument("--epochs", type=int, default=3, help="passes over the training split")
    parser.add_argument("--batch-size", type=int, default=32, help="rows per optimizer update")
    parser.add_argument("--eval-batch", type=int, default=None, help="rows per development batch (default: 2x --batch-size)")
    parser.add_argument("--lr", type=float, default=5e-5, help="peak AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.2, help="fraction of planned updates spent in linear warm-up")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="gradient-norm clipping threshold (0 disables)")
    parser.add_argument("--dev-frac", type=float, default=0.15, help="fraction of rows held out as the grouped dev split")
    parser.add_argument("--telephony-prob", type=float, default=0.3, help="probability of the telephony augmentation per training row")
    parser.add_argument("--fvad-weight", type=float, default=0.5, help="weight of the future-speech auxiliary loss")
    parser.add_argument("--use-context", action="store_true", help="condition on the previous agent utterance")
    parser.add_argument(
        "--min-context-coverage", type=float, default=0.5, help="minimum fraction of rows with agent text for --use-context"
    )
    parser.add_argument("--no-fvad", action="store_true", help="drop the future-speech heads")
    parser.add_argument("--freeze-encoder", action="store_true", help="ablation only")
    parser.add_argument(
        "--max-source-positions",
        type=int,
        default=None,
        help="encoder positions kept behind the 8 s interface (400 = full window, 200 = 4 s internal context); default: the model default",
    )
    parser.add_argument(
        "--normalize-audio", action="store_true", help="peak-normalise each window before the log-mel (stored in the checkpoint)"
    )
    parser.add_argument(
        "--tail-texture-prob",
        type=float,
        default=0.0,
        help="probability of rewriting a training row's pause tail with a label-independent gated/room-noise texture (L037 A1, L040)",
    )
    parser.add_argument(
        "--teacher-cache", type=Path, default=None, help="distil from cached teacher logits (eot-distill-cache) instead of hard labels"
    )
    parser.add_argument(
        "--kd-temperature", type=float, default=DEFAULT_KD_TEMPERATURE, help="distillation temperature (only with --teacher-cache)"
    )
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None, help="pin loader memory (default: on CUDA)")
    parser.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use fused AdamW; defaults to enabled on CUDA and disabled elsewhere",
    )
    parser.add_argument("--compile-model", action="store_true", help="compile the training forward with torch.compile")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help="torch.compile mode for --compile-model",
    )
    parser.add_argument("--device", default=None, help="torch device (default: cuda, then mps, then cpu)")
    parser.add_argument("--require-cuda", action="store_true", help="fail instead of silently training on CPU")
    parser.add_argument("--seed", type=int, default=0, help="training seed (initialisation, sample order, augmentation)")
    parser.add_argument("--log-every", type=int, default=20, help="print and log the running loss every N updates")
    parser.add_argument("--wandb-project", default=None, help="enable W&B logging in this project")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity")
    parser.add_argument("--wandb-name", default=None, help="W&B run name (default: the --out directory name)")
    parser.add_argument("--wandb-group", default=None, help="W&B run group")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online", help="W&B mode")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer updates (also shortens the LR schedule)")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="atomically update checkpoints/last.pt every N optimizer steps and at each epoch",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from PATH, or from <out>/checkpoints/last.pt when passed without a value",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    print(train(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
