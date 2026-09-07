"""ONNX export of a backbone model's encoder + head, with the waveform front-end kept outside the graph.

The Cohere front-end (dither, pre-emphasis, STFT, per-feature normalisation) runs in PyTorch on the
CPU at serving time (``eot.modeling.cohere_filterbank``); the graph exported here takes those
features and returns ``p_eot``. Weights are written as external data next to the ``.onnx`` file, so a
1.9 B-parameter encoder is not embedded in a protobuf. The TensorRT plan of the GPU service is built
from this export (``scripts/build_trt.py``) and qualified against the same PyTorch model
(``scripts/verify_trt.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eot.audio import SAMPLE_RATE
from eot.modeling.backbone_model import BackboneEOTModel

OPSET = 18


class BackboneFeatureHead(nn.Module):
    """``[B, n_mels, T]`` front-end features -> ``p_eot``: the fixed-window encoder, pool and classifier of a backbone model.

    The window is the model's ``window_s`` (4 s for the delivered Cohere model): the number of valid
    frames and encoder positions is derived from the backbone, not hard-coded, and the batch axis is
    dynamic so the same graph serves batches 1-8.
    """

    def __init__(self, model: BackboneEOTModel):
        super().__init__()
        backbone = model.backbone
        self.encoder = backbone.encoder
        self.pool = model.pool
        self.classifier = model.classifier
        n_samples = int(round(model.cfg.window_s * SAMPLE_RATE))
        self.n_frames, self.n_valid = backbone.n_frames(n_samples)
        self.n_positions = self.n_valid // int(backbone.config_dict().get("subsampling_factor", 8))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        lengths = torch.full((features.shape[0],), self.n_valid, dtype=torch.long, device=features.device)
        x, lengths = self.encoder.pre_encode(features, lengths)
        lengths = lengths.to(torch.int64)
        x, pos = self.encoder.pos_enc(x)
        # The upstream attention expands a singleton positional batch behind a Python shape
        # condition that a batch-one trace drops; express the broadcast explicitly so the ONNX
        # graph keeps a dynamic batch.
        pos = pos.expand(x.shape[0], -1, -1)
        pad_mask, att_mask = self.encoder._create_masks(padding_length=lengths, max_audio_length=x.size(1), device=x.device)
        for layer in self.encoder.layers:
            x = layer(x, pos, mask=att_mask, pad_mask=pad_mask)
        return torch.sigmoid(self.classifier(self.pool(x[:, : self.n_positions])).squeeze(-1).float())


def _external_tensor(name: str, array: np.ndarray, blob, filename: str):
    import onnx

    array = np.asarray(array)
    raw = array.tobytes(order="C")
    offset = blob.tell()
    blob.write(raw)
    tensor = onnx.TensorProto(
        name=name, data_type=onnx.helper.np_dtype_to_tensor_dtype(array.dtype), dims=array.shape, data_location=onnx.TensorProto.EXTERNAL
    )
    for key, value in (("location", filename), ("offset", str(offset)), ("length", str(len(raw)))):
        field = tensor.external_data.add()
        field.key, field.value = key, value
    return tensor


def export_external(module: nn.Module, example: torch.Tensor, path: Path) -> Path:
    """Trace ``module`` to ``path`` with every parameter stored in ``<path>.data`` (ONNX external data).

    ``torch.onnx.export`` is run with ``export_params=False`` so the parameters appear as graph inputs;
    each is then rewritten as an external-data initializer. Returns the weights file.
    """
    import onnx

    path = Path(path)
    torch.onnx.export(
        module,
        (example,),
        str(path),
        export_params=False,
        dynamo=False,
        opset_version=OPSET,
        input_names=["features"],
        output_names=["p_eot"],
        do_constant_folding=False,
        dynamic_axes={"features": {0: "batch"}, "p_eot": {0: "batch"}},
    )
    graph = onnx.load(path, load_external_data=False)
    state = module.state_dict()
    weights = path.with_name(path.name + ".data")
    kept = []
    with weights.open("wb") as blob:
        for item in graph.graph.input:
            if item.name == "features":
                kept.append(item)
                continue
            if item.name not in state:
                raise ValueError(f"exported input does not name a model tensor: {item.name}")
            graph.graph.initializer.append(_external_tensor(item.name, state[item.name].detach().cpu().numpy(), blob, weights.name))
    del graph.graph.input[:]
    graph.graph.input.extend(kept)
    onnx.save(graph, path)
    onnx.checker.check_model(str(path))
    return weights
