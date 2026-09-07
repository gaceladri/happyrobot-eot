"""End-of-turn (EoT) detection for voice agents.

The package follows the pipeline order; each stage only imports from the stages before it:

- ``eot.io``        durable JSONL/WAV helpers (torch-free)
- ``eot.audio``     waveform conventions, log-mel front-end, pause detectors, augmentation
- ``eot.context``   hashed agent-text features shared by training and ONNX serving
- ``eot.metrics``   numpy-only AUC / Pareto helpers
- ``eot.onnx``      onnxruntime session helpers
- ``eot.labeling``  clip -> pause-level samples: prefix mining, dual-channel oracle, event
                    targets, labeler QA
- ``eot.data``      Smart Turn acquisition, grouped splits, torch datasets, cached teacher logits
- ``eot.modeling``  Whisper model and trainer, LoRA, large-encoder backbones (Cohere Transcribe) and
                    their trainer, teacher distillation, weight soups, ONNX export
- ``eot.eval``      endpointing policy, EoT Bench adapter, Krisp test set, reports, challenge set
- ``eot.serving``   one FastAPI contract over the CPU (onnxruntime) and GPU (TensorRT) scorers, bundle verifier, load test

Serving and evaluation adapters stay importable without torch; ``tests/test_layering.py`` enforces the order.
"""

__version__ = "0.3.0"
