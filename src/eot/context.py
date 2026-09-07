"""Torch-free text-context features shared by training and ONNX serving."""

from __future__ import annotations

import hashlib
import re

import numpy as np

CTX_VOCAB = 2**16
CTX_LEN = 32
_TOKEN_RE = re.compile(r"[a-z0-9]+|\?|[^\sa-z0-9]")


def hash_context(text: str, max_len: int = CTX_LEN, vocab: int = CTX_VOCAB) -> np.ndarray:
    """Hash unigrams and bigrams into a fixed, padding-zero int64 vector."""
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if vocab <= 1:
        raise ValueError("vocab must be greater than one")
    toks = _TOKEN_RE.findall((text or "").lower())
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    ids = [int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "little") % (vocab - 1) + 1 for g in grams][-max_len:]
    out = np.zeros(max_len, dtype=np.int64)
    if ids:
        out[-len(ids) :] = ids
    return out


__all__ = ["CTX_LEN", "CTX_VOCAB", "hash_context"]
