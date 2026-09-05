"""End-of-turn (EoT) detection for voice agents.

The package follows the pipeline order; each stage only imports from the stages before it:

- ``eot.io``        durable JSONL/WAV helpers (torch-free)
- ``eot.audio``     waveform conventions, log-mel front-end, pause detectors, augmentation
- ``eot.context``   hashed agent-text features shared by training and ONNX serving
- ``eot.metrics``   numpy-only AUC / Pareto helpers
- ``eot.onnx``      onnxruntime session helpers
- ``eot.labeling``  clip -> pause-level samples: prefix mining, dual-channel oracle, event
                    targets, labeler QA
- ``eot.data``      Smart Turn acquisition, grouped splits, torch dataset
- ``eot.modeling``  Whisper-Tiny model, training, weight soups, ONNX export
- ``eot.eval``      endpointing policy, EoT Bench adapter, Krisp test set, reports, challenge set
- ``eot.serving``   FastAPI/onnxruntime service and load test

Serving and evaluation adapters stay importable without torch; ``tests/test_layering.py`` enforces the order.
"""

__version__ = "0.2.0"
