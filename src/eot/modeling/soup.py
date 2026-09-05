"""Fixed-weight model soups: average two fine-tunes that share an initialisation.

Averaging nearby solutions (Wortsman et al., 2022) can reduce seed variance without any extra
serving computation. The mix is fixed up front (``--alpha``) and scored once on the frozen
development split; alpha is never searched against a benchmark.

    uv run eot-soup --left runs/a/model.pt --right runs/b/model.pt --alpha 0.5 \\
        --samples data/optimization/mixed.jsonl --out runs/soup_ab
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eot.data.dataset import MinedDataset, collate, read_samples
from eot.data.splits import grouped_split
from eot.io import atomic_write_json, atomic_write_jsonl, sha256_file
from eot.modeling.model import EOTModel, load_checkpoint
from eot.modeling.train import atomic_copy, atomic_torch_save, evaluate


def average_models(left: EOTModel, right: EOTModel, alpha: float) -> EOTModel:
    """Return ``left`` with every floating tensor replaced by ``lerp(left, right, alpha)``.

    Configurations, parameter names and non-floating buffers must match exactly.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    if asdict(left.cfg) != asdict(right.cfg):
        raise ValueError("model and preprocessing configs must match exactly")
    ls, rs = left.state_dict(), right.state_dict()
    if ls.keys() != rs.keys():
        raise ValueError("state dict keys differ")
    for key, value in ls.items():
        if value.is_floating_point():
            ls[key] = torch.lerp(value, rs[key], alpha)
        elif not torch.equal(value, rs[key]):
            raise ValueError(f"non-floating buffer {key!r} differs")
    left.load_state_dict(ls)
    return left


def soup(left: Path, right: Path, samples: Path, out: Path, alpha: float, *, split_seed: int = 17, dev_frac: float = 0.15) -> dict:
    if (out / "model.pt").exists():
        raise FileExistsError(f"{out} already holds a completed candidate")
    torch.manual_seed(split_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = average_models(load_checkpoint(str(left)), load_checkpoint(str(right)), alpha)
    out.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"cfg": asdict(model.cfg), "state_dict": model.state_dict()}, out / "candidate.pt")

    rows = [r for r in read_samples(samples) if r.get("time_to_onset") is None or r["time_to_onset"] >= 0]
    _, dev, diagnostics = grouped_split(rows, dev_frac=dev_frac, seed=split_seed, return_diagnostics=True)
    loader = DataLoader(MinedDataset(dev, seed=split_seed + 1, normalize_audio=model.cfg.normalize_audio),
                        batch_size=64, num_workers=0, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    metrics, scores = evaluate(model.to(device), loader, device)
    metrics.update(step=0, elapsed_s=time.time() - started)

    atomic_write_jsonl(out / "dev_scores.jsonl", ({**r, **s, "path": None} for r, s in zip(dev, scores)))
    atomic_write_jsonl(out / "history.jsonl", [metrics])
    atomic_write_json(out / "provenance.json", {
        "method": "fixed_weight_soup", "alpha": alpha, "left_sha256": sha256_file(left),
        "right_sha256": sha256_file(right), "samples_sha256": sha256_file(samples), "split": diagnostics,
    })
    atomic_copy(out / "candidate.pt", out / "model.pt")  # publish only after scoring completed
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True, help="weight of --right; 0.5 is an equal soup")
    ap.add_argument("--split-seed", type=int, default=17)
    args = ap.parse_args(argv)
    print(json.dumps(soup(args.left, args.right, args.samples, args.out, args.alpha, split_seed=args.split_seed)))


if __name__ == "__main__":
    main()
