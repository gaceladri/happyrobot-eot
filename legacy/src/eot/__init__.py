"""End-of-turn (EOT) detection for voice agents.

Package layout (each module is runnable, see ``[project.scripts]`` in pyproject):

- ``audio``          : 16 kHz framing, last-8 s window, Whisper log-mel features, telephony augmentation.
- ``prefix_mining``  : turns clip-level EOT/HOLD data into pause-level causal samples (train/eval alignment).
- ``model``          : Whisper-Tiny encoder + attention pooling + EOT head, optional agent-context
                       conditioning and multi-horizon future-speech auxiliary heads.
- ``data``           : dataset loading, source-grouped splits, collate.
- ``train``          : full fine-tune recipe (Smart Turn v3 defaults), device-agnostic.
- ``policy``         : threshold / action-delay / timeout endpointing policy, causal replay and sweep.
- ``eotbench_adapter``: batch adapter implementing the LiveKit eot-bench contract.
- ``export_onnx``    : ONNX export with parity check.
- ``serve``          : FastAPI + onnxruntime inference service.
- ``load_test``      : concurrency / latency stress test against the service.
- ``challenge``      : structured-data (numbers, MC numbers, rates) challenge-set generator.
"""

__version__ = "0.1.0"
