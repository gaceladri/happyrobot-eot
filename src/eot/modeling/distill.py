"""Knowledge distillation of the Whisper detector from a large teacher (Cohere Transcribe + LoRA).

Objective: pure Bernoulli KL between the teacher's and the student's end-of-turn distributions at a
fixed temperature (T² · KL(q_teacher ‖ p_student)), weighted by the manifest row weights. There is
no hard-label term and no class ``pos_weight``: the teacher's soft targets replace the labels.
Teacher and student see the *same realised waveform*, including tail texture, which is why the
teacher is run once offline over the training split (``eot-distill-cache``) and its logits are
cached next to a hash of every waveform (``eot.data.teacher_cache``). Training then goes through the
central trainer with ``eot-train --teacher-cache``.

This is the recipe of L040 arm B11 (teacher: L031 C2 LoRA seed 111, T = 2, one pass of 2,924 updates
on fresh Whisper-base with a 4 s internal context, ``normalize_audio``, tail texture p = 0.5):
official delay@5 % 801.8 ms vs 850.0 ms for distillation alone and 896.8 ms for the hard-label
control (three-seed means). Nothing was promoted; see the L040 report on the archived branch ``archive/research``.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_KD_TEMPERATURE = 2.0  # T of L040 B11; shared by the model's forward and the trainer's --kd-temperature


def bernoulli_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = DEFAULT_KD_TEMPERATURE,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """``T² · KL(Bernoulli(t / T) ‖ Bernoulli(s / T))`` averaged with optional per-row weights, in fp32.

    Teacher logits are detached. The T² factor keeps gradient magnitudes comparable across
    temperatures (Hinton et al., 2015). Zero weights exclude rows; all-zero weights give 0.
    """
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    s = student_logits.float() / temperature
    t = teacher_logits.detach().float() / temperature
    q = torch.sigmoid(t)
    per_row = (q * (F.logsigmoid(t) - F.logsigmoid(s)) + (1 - q) * (F.logsigmoid(-t) - F.logsigmoid(-s))) * temperature**2
    if weight is None:
        return per_row.mean()
    w = weight.float()
    return (per_row * w).sum() / w.sum().clamp(min=torch.finfo(torch.float32).tiny)


# --------------------------------------------------------------------------------------------- teacher cache producer
class BackboneTeacher:
    """Scores 8 s waveforms with a backbone checkpoint in IEEE fp32 (TF32 off), returning raw logits."""

    def __init__(self, checkpoint: Path, device: str | None = None, merge_lora: bool = False):
        from eot.io import sha256_file
        from eot.modeling.backbone_model import load_checkpoint
        from eot.modeling.lora import merge_lora as merge

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_checkpoint(checkpoint).float().to(self.device)
        self.merged_lora_layers = merge(self.model) if merge_lora else 0
        self.model.requires_grad_(False).eval()
        self.identity = {
            "checkpoint_sha256": sha256_file(Path(checkpoint)),
            "backbone": self.model.cfg.backbone,
            "repo": self.model.cfg.repo,
            "revision": self.model.cfg.revision,
            "precision": "ieee-fp32",
            "merged_lora": bool(merge_lora),
        }

    @torch.inference_mode()
    def __call__(self, waves: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(waves, dtype=np.float32)).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            logits = self.model(x)["logit"].float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("teacher produced non-finite logits")
        return logits


def batching_parity(teacher, waves: np.ndarray, batch_size: int, tolerance: float = 1e-5) -> float:
    """Max |logit| difference between single-row and batched scoring; batching must not change targets."""
    reference = np.concatenate([teacher(w[None]) for w in waves])
    batched = np.concatenate([teacher(waves[i : i + batch_size]) for i in range(0, len(waves), batch_size)])
    diff = float(np.max(np.abs(reference - batched)))
    if diff > tolerance:
        raise ValueError(f"teacher batching parity failed: max |Δlogit| = {diff:.3g} > {tolerance}")
    return diff


def build_cache(
    dataset, teacher, out: Path, teacher_identity: dict, *, batch_size: int = 32, workers: int = 4, log_every: int = 50
) -> dict:
    """Score every row of ``dataset`` (a ``MinedDataset`` at its current epoch) and publish a complete cache at ``out``.

    Resumes from the last complete shard. Waveforms are produced in threads (I/O and numpy release the GIL).
    """
    from eot.audio import WINDOW_SECONDS, last_window
    from eot.data.teacher_cache import CacheWriter, cache_identity, load_cache, waveform_sha256

    rows = dataset.rows
    identity = cache_identity(rows, dataset, teacher_identity)
    out = Path(out)
    if (out / "metadata.json").exists():
        _, _, metadata = load_cache(out, rows, identity)
        print(json.dumps({"status": "COMPLETE_ALREADY", "n_rows": len(rows), "cache": str(out)}), flush=True)
        return metadata

    def window(i: int) -> np.ndarray:
        return np.ascontiguousarray(last_window(dataset.waveform(i), WINDOW_SECONDS), dtype=np.float32)

    started = time.time()
    with CacheWriter(out, identity, rows) as writer, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        start = writer.next_start()
        if start == 0:
            probe = np.stack(list(pool.map(window, range(min(8, len(rows))))))
            parity = batching_parity(teacher, probe, min(batch_size, len(probe)))
            print(json.dumps({"batching_parity_max_abs_logit": parity}), flush=True)
        n_batches = 0
        while start < len(rows):
            end = min(start + batch_size, len(rows))
            waves = np.stack(list(pool.map(window, range(start, end))))
            writer.append(start, teacher(waves), [waveform_sha256(w) for w in waves])
            start, n_batches = end, n_batches + 1
            if n_batches % log_every == 0:
                rate = start / max(1e-9, time.time() - started)
                print(f"{start}/{len(rows)} rows  {rate:.1f} rows/s  eta {(len(rows) - start) / max(rate, 1e-9) / 60:.0f} min", flush=True)
        metadata = writer.finish()
    load_cache(out, rows, identity)
    print(
        json.dumps({"status": "COMPLETE", "n_rows": len(rows), "elapsed_s": round(time.time() - started, 1), "cache": str(out)}), flush=True
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    from eot.modeling.training_utils import HelpFormatter

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    ap.add_argument("--samples", type=Path, required=True, help="the training manifest (the same one passed to eot-train)")
    ap.add_argument("--teacher", type=Path, required=True, help="backbone checkpoint (full or delta) of the teacher")
    ap.add_argument("--out", type=Path, required=True, help="cache directory")
    ap.add_argument("--seed", type=int, required=True, help="student training seed (drives the augmentation stream)")
    ap.add_argument("--split-seed", type=int, default=None, help="as eot-train (default: --seed)")
    ap.add_argument("--dev-frac", type=float, default=0.15, help="as eot-train")
    ap.add_argument("--telephony-prob", type=float, default=0.3, help="as eot-train (part of the cached view)")
    ap.add_argument("--tail-texture-prob", type=float, default=0.0, help="as eot-train (part of the cached view)")
    ap.add_argument("--no-noise-fill", action="store_true", help="skip room tone on noise_fill rows (diagnostic views only)")
    ap.add_argument("--batch-size", type=int, default=32, help="teacher scoring batch")
    ap.add_argument("--workers", type=int, default=4, help="threads producing waveforms")
    ap.add_argument("--device", default=None, help="torch device for the teacher (default: cuda when available)")
    ap.add_argument(
        "--merge-lora", action="store_true", help="fold LoRA adapters before scoring (faster; targets differ by float rounding only)"
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    """Score the training split of ``--samples`` at epoch 0, the single pass ``eot-train --teacher-cache`` runs."""
    args = build_parser().parse_args(argv)
    from eot.data.dataset import MinedDataset
    from eot.data.splits import grouped_split
    from eot.io import read_samples
    from eot.modeling.training_utils import filter_time_to_onset

    rows, _ = filter_time_to_onset(read_samples(args.samples))
    train_rows, _dev = grouped_split(rows, dev_frac=args.dev_frac, seed=args.seed if args.split_seed is None else args.split_seed)
    dataset = MinedDataset(
        train_rows,
        telephony_prob=args.telephony_prob,
        seed=args.seed,
        noise_fill=not args.no_noise_fill,
        tail_texture_prob=args.tail_texture_prob,
    )
    teacher = BackboneTeacher(args.teacher, device=args.device, merge_lora=args.merge_lora)
    build_cache(dataset, teacher, args.out, teacher.identity, batch_size=args.batch_size, workers=args.workers)


if __name__ == "__main__":
    main()
