"""Dependency-free evaluation helpers shared by Krisp, held-out and local sweeps.
"""
from __future__ import annotations
import numpy as np


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


def pareto_front(results: list[dict], *, latency_key: str = "mean_latency", cutoff_key: str = "cutoff_rate") -> list[dict]:
    """Non-dominated (latency, cutoff) policies, sorted by latency."""
    front: list[dict] = []
    best = float("inf")
    for r in sorted(results, key=lambda r: (r[latency_key], r[cutoff_key])):
        if r[cutoff_key] < best:
            front.append(r)
            best = r[cutoff_key]
    return front
