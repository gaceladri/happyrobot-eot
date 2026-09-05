"""onnxruntime session helpers shared by serving, benchmark adapters, export and the VAD (torch-free)."""

from __future__ import annotations

import os

import numpy as np


def cpu_session(path: str | os.PathLike, threads: int = 1, *, log_severity: int | None = None):
    """CPU inference session with a fixed intra-op thread count and sequential inter-op execution."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    if log_severity is not None:
        options.log_severity_level = log_severity
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def input_names(session) -> frozenset[str]:
    return frozenset(item.name for item in session.get_inputs())


def model_feed(names: frozenset[str] | set[str], features: np.ndarray, context: np.ndarray) -> dict[str, np.ndarray]:
    """Feed for an EoT graph. ONNX legitimately prunes ``context_ids`` from an audio-only model, so the
    same code path serves graphs with and without the context input."""
    feed = {"input_features": features}
    if "context_ids" in names:
        feed["context_ids"] = context
    return feed
