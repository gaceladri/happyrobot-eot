"""End-of-turn (EoT) detection for voice agents.

The package follows the pipeline order; each stage only imports from the stages before it:

- ``eot.io``        durable JSONL/WAV helpers (torch-free)
- ``eot.audio``     waveform conventions, log-mel front-end, pause detectors, augmentation
- ``eot.context``   hashed agent-text features shared by training and ONNX serving
- ``eot.labeling``  clip -> pause-level samples: prefix mining, dual-channel oracle, event
                    targets, labeler QA
- ``eot.data``      corpora (Smart Turn, Krisp), grouped splits, torch dataset
- ``eot.modeling``  Whisper-Tiny model, training, weight soups, ONNX export
- ``eot.eval``      endpointing policy, EoT Bench adapter, reports, challenge set
- ``eot.serving``   FastAPI/onnxruntime service and load test

Serving and evaluation adapters stay importable without torch.
"""

__version__ = "0.2.0"
