"""Fixed-weight model soups: average two fine-tunes that share an initialisation.

Averaging nearby solutions (Wortsman et al., 2022) can reduce seed variance without any extra
serving computation. The mix is fixed up front (``--alpha``) and scored once on the development split the parents
were trained against (read from their ``provenance.json``); alpha is never searched against a
benchmark.

    uv run eot-soup --left runs/a/model.pt --right runs/b/model.pt --alpha 0.5 \\
        --samples data/optimization/mixed.jsonl --out runs/soup_ab
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eot.data.dataset import MinedDataset, collate
from eot.data.splits import grouped_split
from eot.io import atomic_write_json, atomic_write_jsonl, read_samples, sha256_file
from eot.modeling.model import EOTModel, load_checkpoint
from eot.modeling.train import atomic_torch_save, evaluate


def average_models(left: EOTModel, right: EOTModel, alpha: float) -> EOTModel:
    """Blend ``right`` into ``left`` in place: every floating tensor becomes ``lerp(left, right, alpha)``.

    Configurations, parameter names and non-floating buffers must match exactly.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    if asdict(left.cfg) != asdict(right.cfg):
        raise ValueError("model and preprocessing configs must match exactly")
    ls, rs = left.state_dict(), right.state_dict()
    if ls.keys() != rs.keys():
        raise ValueError("state dict keys differ")
    with torch.no_grad():
        for key, value in ls.items():  # state-dict tensors alias the parameters, so lerp_ updates the model
            if value.is_floating_point():
                value.lerp_(rs[key], alpha)
            elif not torch.equal(value, rs[key]):
                raise ValueError(f"non-floating buffer {key!r} differs")
    return left


def parent_split(checkpoint: Path) -> dict | None:
    """Effective split seed, dev fraction and samples hash recorded by ``eot-train`` next to a checkpoint."""
    provenance = checkpoint.parent / "provenance.json"
    if not provenance.exists():
        return None
    info = json.loads(provenance.read_text())
    args = info.get("args") or {}
    seed = args.get("split_seed")
    return {
        "split_seed": args.get("seed") if seed is None else seed,
        "dev_frac": args.get("dev_frac"),
        "samples_sha256": info.get("samples_sha256"),
    }


def resolve_split(parents: list[dict | None], split_seed: int | None, dev_frac: float | None, samples_sha256: str) -> tuple[int, float]:
    """Choose the development split the parents were trained against, refusing silent mismatches.

    A soup scored on a different split than its parents would report an inflated "frozen dev"
    number, because rows the parents trained on would land in its dev set.
    """
    known = [p for p in parents if p is not None]
    for key, requested in (("split_seed", split_seed), ("dev_frac", dev_frac)):
        recorded = {p[key] for p in known if p.get(key) is not None}
        if len(recorded) > 1:
            raise ValueError(f"parents were trained with different {key} values: {sorted(recorded)}")
        if requested is not None and recorded and requested not in recorded:
            raise ValueError(f"requested {key}={requested} but parents used {recorded.pop()}")
    for p in known:
        if p.get("samples_sha256") and p["samples_sha256"] != samples_sha256:
            raise ValueError("samples manifest differs from the one the parents were trained on")
    seed = split_seed if split_seed is not None else next((p["split_seed"] for p in known if p.get("split_seed") is not None), None)
    frac = dev_frac if dev_frac is not None else next((p["dev_frac"] for p in known if p.get("dev_frac") is not None), 0.15)
    if seed is None:
        raise ValueError("no provenance.json next to the parents; pass --split-seed explicitly")
    return int(seed), float(frac)


def soup(left: Path, right: Path, samples: Path, out: Path, alpha: float, *, split_seed: int | None = None, dev_frac: float | None = None, workers: int = 0) -> dict:
    if (out / "model.pt").exists():
        raise FileExistsError(f"{out} already holds a completed candidate")
    samples_sha256 = sha256_file(samples)
    split_seed, dev_frac = resolve_split([parent_split(left), parent_split(right)], split_seed, dev_frac, samples_sha256)
    torch.manual_seed(split_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = average_models(load_checkpoint(str(left)), load_checkpoint(str(right)), alpha)
    out.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"cfg": asdict(model.cfg), "state_dict": model.state_dict()}, out / "candidate.pt")

    rows = [r for r in read_samples(samples) if r.get("time_to_onset") is None or r["time_to_onset"] >= 0]
    _, dev, diagnostics = grouped_split(rows, dev_frac=dev_frac, seed=split_seed, return_diagnostics=True)
    loader = DataLoader(MinedDataset(dev, seed=split_seed + 1, normalize_audio=model.cfg.normalize_audio),
                        batch_size=64, num_workers=workers, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    metrics, scores = evaluate(model.to(device), loader, device)
    metrics.update(step=0, elapsed_s=time.time() - started)

    atomic_write_jsonl(out / "dev_scores.jsonl", ({**r, **s, "path": None} for r, s in zip(dev, scores)))
    atomic_write_jsonl(out / "history.jsonl", [metrics])
    atomic_write_json(out / "provenance.json", {
        "method": "fixed_weight_soup", "alpha": alpha, "left_sha256": sha256_file(left),
        "right_sha256": sha256_file(right), "samples_sha256": samples_sha256, "split": diagnostics,
        "split_seed": split_seed, "dev_frac": dev_frac,
    })
    os.replace(out / "candidate.pt", out / "model.pt")  # publish only after scoring completed
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True, help="weight of --right; 0.5 is an equal soup")
    ap.add_argument("--split-seed", type=int, default=None, help="default: the split seed recorded in the parents' provenance.json")
    ap.add_argument("--dev-frac", type=float, default=None, help="default: the parents' dev fraction")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers for the development pass")
    args = ap.parse_args(argv)
    metrics = soup(args.left, args.right, args.samples, args.out, args.alpha,
                   split_seed=args.split_seed, dev_frac=args.dev_frac, workers=args.workers)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
