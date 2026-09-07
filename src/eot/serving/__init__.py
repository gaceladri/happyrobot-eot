"""HTTP inference services: one contract (``http``) over a CPU onnxruntime scorer (``service``, torch-free)
and a GPU TensorRT scorer (``tensorrt_service``); ``artifacts`` verifies/installs the model bundle and
``load_test`` measures either service."""
