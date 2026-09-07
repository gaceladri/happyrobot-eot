"""TensorRT-only encoder/head with pinned buffers and a CUDA graph per batch size.

All calls must run on one executor. Each captured graph owns a fixed-shape execution context; no
context or addresses change after capture. ``tensorrt``, ``cuda`` and ``torch`` are imported lazily
so the module (and its constants) stay importable on a CPU-only host.
"""

from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from eot.audio import SAMPLE_RATE, last_window
from eot.io import sha256_file

MAX_BATCH = 8  # the engine is built with profiles covering batches 1..8; one context and graph per size
COHERE_WINDOW_SECONDS = 4.0  # the Cohere student scores the trailing 4 s
COHERE_N_MELS = 128
COHERE_HOP = 160  # 10 ms at 16 kHz, centre STFT: 1 + N // hop frames
COHERE_N_FRAMES = 1 + int(COHERE_WINDOW_SECONDS * SAMPLE_RATE) // COHERE_HOP
FEATURE_SHAPE = (COHERE_N_MELS, COHERE_N_FRAMES)
SIDECAR_KEYS = ("accepted", "sha256", "frontend_sha256", "model_name", "checkpoint_sha256", "precision")


def _checked(result: tuple) -> Any:
    """Unwrap a ``cuda.bindings.runtime`` call: ``(error, *values)`` -> values, raising on a non-zero error."""
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


class PlanRunner:
    """Deserialised plan with a warmed, fixed-shape execution context per batch size 1..``MAX_BATCH``."""

    def __init__(self, path: str | Path, *, graphs: bool = True):
        import tensorrt as trt
        from cuda.bindings import runtime as cuda

        self.trt, self.cuda = trt, cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize TensorRT engine on this hardware/runtime")
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if names != {"features", "p_eot"}:
            raise ValueError(f"Unexpected engine interface: {names}")
        self.stream = _checked(cuda.cudaStreamCreate())
        self.batches: dict[int, dict[str, Any]] = {}
        self.graphs = graphs
        try:
            self._prepare_batches()
        except Exception:
            self.close()
            raise

    def _prepare_batches(self) -> None:
        cuda, trt = self.cuda, self.trt
        for batch in range(1, MAX_BATCH + 1):
            context = self.engine.create_execution_context()
            if context is None:
                raise RuntimeError("Cannot create TensorRT execution context")
            for profile in range(self.engine.num_optimization_profiles):
                low, _, high = self.engine.get_tensor_profile_shape("features", profile)
                if low[0] <= batch <= high[0]:
                    break
            else:
                raise ValueError(f"No profile for batch {batch}")
            if not context.set_optimization_profile_async(profile, self.stream):
                raise ValueError("Cannot select TensorRT optimization profile")
            if not context.set_input_shape("features", (batch, *FEATURE_SHAPE)):
                raise ValueError(f"Engine must support batches 1 through {MAX_BATCH}")
            state: dict[str, Any] = {"context": context, "buffers": {}, "graph": None, "instance": None}
            self.batches[batch] = state
            for name in ("features", "p_eot"):
                shape = tuple(context.get_tensor_shape(name))
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                if dtype != np.float32 or any(d <= 0 for d in shape):
                    raise ValueError("Engine I/O must be fixed-shape FP32 after setting batch")
                size = int(np.prod(shape)) * dtype.itemsize
                device = _checked(cuda.cudaMalloc(size))
                try:
                    host = _checked(cuda.cudaHostAlloc(size, 0))
                except Exception:
                    _checked(cuda.cudaFree(device))
                    raise
                # Register allocations immediately, including partial initialization.
                state["buffers"][name] = (device, host, None, size)
                array = np.ctypeslib.as_array((ctypes.c_float * (size // dtype.itemsize)).from_address(host)).reshape(shape)
                array.fill(0)
                state["buffers"][name] = (device, host, array, size)
                if not context.set_tensor_address(name, device):
                    raise RuntimeError(f"Cannot bind TensorRT tensor {name}")
            device, host, _, size = state["buffers"]["features"]
            _checked(cuda.cudaMemcpyAsync(device, host, size, cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
            if not context.execute_async_v3(self.stream):
                raise RuntimeError("TensorRT warmup enqueue failed")
            _checked(cuda.cudaStreamSynchronize(self.stream))
            if self.graphs:
                _checked(cuda.cudaStreamBeginCapture(self.stream, cuda.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
                try:
                    enqueued = context.execute_async_v3(self.stream)
                finally:
                    # Always leave capture mode; its status is only checked once the enqueue itself succeeded,
                    # so an enqueue error is never masked by the EndCapture one it causes.
                    ended = cuda.cudaStreamEndCapture(self.stream)
                if not enqueued:
                    raise RuntimeError("TensorRT capture enqueue failed")
                state["graph"] = _checked(ended)
                state["instance"] = _checked(cuda.cudaGraphInstantiate(state["graph"], 0))

    def infer(self, features: np.ndarray) -> np.ndarray:
        """``[batch, 128, 401]`` float32 features -> ``[batch]`` probabilities."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 3 or features.shape[1:] != FEATURE_SHAPE or len(features) not in self.batches:
            raise ValueError(f"Expected features [batch 1..{MAX_BATCH}, {COHERE_N_MELS}, {COHERE_N_FRAMES}]")
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite features")
        state = self.batches[len(features)]
        device, host, array, size = state["buffers"]["features"]
        np.copyto(array, features)
        cuda = self.cuda
        _checked(cuda.cudaMemcpyAsync(device, host, size, cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if state["instance"] is not None:
            _checked(cuda.cudaGraphLaunch(state["instance"], self.stream))
        elif not state["context"].execute_async_v3(self.stream):
            raise RuntimeError("TensorRT enqueue failed")
        device, host, array, size = state["buffers"]["p_eot"]
        _checked(cuda.cudaMemcpyAsync(host, device, size, cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        _checked(cuda.cudaStreamSynchronize(self.stream))
        result = array.copy()
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError("Invalid TensorRT probabilities")
        return result

    def close(self) -> None:
        if self.stream is None:
            return
        cuda = self.cuda
        _checked(cuda.cudaStreamSynchronize(self.stream))
        for state in self.batches.values():
            if state["instance"] is not None:
                _checked(cuda.cudaGraphExecDestroy(state["instance"]))
            if state["graph"] is not None:
                _checked(cuda.cudaGraphDestroy(state["graph"]))
            state["context"] = None
            for device, host, _, _ in state["buffers"].values():
                _checked(cuda.cudaFree(device))
                _checked(cuda.cudaFreeHost(host))
        self.batches.clear()
        _checked(cuda.cudaStreamDestroy(self.stream))
        self.stream = None


class TensorRTEngine:
    """Verified plan (``eot.json`` sidecar) plus the original Cohere front-end (``frontend.pt``) on CPU torch."""

    def __init__(self, path: str | Path, *, threads: int = 2, graphs: bool = True):
        import torch

        from eot.modeling.cohere_filterbank import CohereFilterbank

        path = Path(path)
        self.meta: dict = json.loads(path.with_suffix(".json").read_text())
        missing = [key for key in SIDECAR_KEYS if key not in self.meta]
        if missing:
            raise ValueError(f"TensorRT sidecar {path.with_suffix('.json')} lacks {missing}")
        if self.meta["accepted"] is not True or sha256_file(path) != self.meta["sha256"]:
            raise ValueError("TensorRT engine is unverified or its hash differs")
        frontend = path.parent / "frontend.pt"
        if sha256_file(frontend) != self.meta["frontend_sha256"]:
            raise ValueError("Frontend hash differs")
        torch.set_num_threads(threads)
        self.frontend = CohereFilterbank.from_payload(torch.load(frontend, map_location="cpu", weights_only=True)).eval()
        self.runner = PlanRunner(path, graphs=graphs)
        self.torch, self.threads = torch, threads
        self.sha: str = self.meta["sha256"][:16]

    def infer(self, waves: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
        """``(p_eot per wave, features_ms, inference_ms)`` for 1..``MAX_BATCH`` 16 kHz waveforms."""
        start = time.perf_counter()
        x = np.stack([last_window(w, COHERE_WINDOW_SECONDS) for w in waves])
        with self.torch.inference_mode():
            features = self.frontend(self.torch.from_numpy(x)).numpy()
        front = time.perf_counter()
        values = self.runner.infer(features)
        return values, (front - start) * 1000, (time.perf_counter() - front) * 1000

    def metadata(self) -> dict:
        return {
            "model_name": self.meta["model_name"],
            "engine_sha256": self.meta["sha256"],
            "checkpoint_sha256": self.meta["checkpoint_sha256"],
            "runtime": "TensorRT",
            "tensorrt": self.runner.trt.__version__,
            "precision": self.meta["precision"],
            "cuda_graphs": self.runner.graphs,
            "threads": self.threads,
            "original_frontend": True,
        }

    def close(self) -> None:
        self.runner.close()
