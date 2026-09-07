"""Large pretrained audio encoders as end-of-turn backbones (teacher / upper-bound models).

A backbone owns (1) its own front-end (16 kHz waveform -> model features, computed on the model's
device inside ``forward`` so training and every evaluation path share one implementation), (2) an
encoder made of transformer layers that can be frozen from the bottom, gradient-checkpointed or
read at an intermediate layer, and (3) metadata (configuration, parameter count).

Registered backbones:

    cohere_transcribe   CohereLabs/cohere-transcribe-03-2026 Fast Conformer encoder (1.9 B, bidirectional,
                        12.5 Hz). Used in L031 (LoRA r=16 on all 48 layers = the L040 teacher).

Adding one means subclassing ``Backbone`` and adding it to ``REGISTRY``; nothing else in the package
is model-specific. Pretrained snapshots are read from an optional local mirror (``EOT_HF_MIRROR``)
before the Hugging Face cache, because the shared cache on the training host is volatile.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from eot.audio import SAMPLE_RATE
from eot.io import atomic_copy
from eot.modeling.cohere_filterbank import CohereFilterbank


class Backbone(nn.Module):
    """Interface every backbone implements; see the module docstring."""

    name = "abstract"
    lora_targets: tuple[str, ...] = ()  # attribute names of the nn.Linear layers LoRA adapts
    d_model: int
    n_layers: int

    def __init__(self):
        super().__init__()
        self.first_trainable: int | None = None  # lowest transformer layer that needs gradients (None = no layer)
        self.pre_trainable = False  # whether the modules before the layers (subsampling, ...) need gradients
        self.grad_checkpointing = False

    # ---- to implement --------------------------------------------------------------------------------------------
    def frontend(self, wave: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def layer_modules(self) -> list[nn.Module]:
        raise NotImplementedError

    def post_layer_modules(self) -> list[nn.Module]:
        raise NotImplementedError

    def config_dict(self) -> dict:
        raise NotImplementedError

    def frontend_dict(self) -> dict:
        raise NotImplementedError

    def forward(self, wave: torch.Tensor, layer: int = -1) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def snapshot_dir(cls, repo: str, revision: str | None) -> Path:
        raise NotImplementedError

    # ---- shared helpers -------------------------------------------------------------------------------------------
    def fp32_parameters(self) -> set[int]:
        """ids of parameters that must stay fp32 even when frozen (default: none)."""
        return set()

    def encoder_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def layers_used(self, layer: int) -> int:
        n = len(self.layer_modules())
        upto = n if layer == -1 else int(layer)
        if not 1 <= upto <= n:
            raise ValueError(f"layer must be -1 or in [1, {n}], got {layer}")
        return upto

    def run_layer(self, i: int, layer: nn.Module, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Frozen prefix under ``no_grad``; trainable layers optionally gradient-checkpointed (non-reentrant)."""
        needs_grad = self.first_trainable is not None and i >= self.first_trainable and torch.is_grad_enabled()
        if not needs_grad:
            with torch.no_grad():
                return layer(x, *args, **kwargs)
        if self.grad_checkpointing and self.training:
            return cp.checkpoint(layer, x, *args, use_reentrant=False, **kwargs)
        return layer(x, *args, **kwargs)


def mirrored_snapshot(repo: str, revision: str | None, files: tuple[str, ...]) -> Path:
    """Locate ``files`` of a pinned Hub snapshot: local mirror (``EOT_HF_MIRROR``) -> HF cache -> download.

    When a mirror directory is configured the snapshot is copied into it, so later runs survive a
    wiped Hub cache. Gated repositories need ``HF_TOKEN`` for the download step.
    """
    from huggingface_hub import constants, snapshot_download

    name = f"models--{repo.replace('/', '--')}"
    mirror_root = os.environ.get("EOT_HF_MIRROR")
    mirror = Path(mirror_root) / name / (revision or "main") if mirror_root else None
    if mirror is not None and all((mirror / f).exists() for f in files):
        return mirror
    local = Path(constants.HF_HUB_CACHE) / name / "snapshots" / (revision or "")
    if not (revision and all((local / f).exists() for f in files)):
        local = Path(snapshot_download(repo, revision=revision, allow_patterns=list(files)))
    if mirror is None:
        return local
    mirror.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not (mirror / f).exists():
            atomic_copy(local / f, mirror / f)
    return mirror


class Fp32BatchNorm1d(nn.BatchNorm1d):
    """BatchNorm evaluated in fp32 with the input's dtype restored, so CPU fp32 and GPU bf16 autocast agree."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(x.float()).to(x.dtype)


# ------------------------------------------------------------------------------------------------ Cohere Transcribe
class CohereTranscribeBackbone(Backbone):
    """``CohereLabs/cohere-transcribe-03-2026`` encoder (remote code ``modeling_cohere_asr.ConformerEncoder`` at a pinned sha).

    128 log-mel at 100 Hz (25 ms / 10 ms, n_fft 512, pre-emphasis 0.97, slaney mels 0-8 kHz, ``log(x + 2^-24)``,
    per-feature normalisation over the whole window, deterministic dither 1e-5) -> depthwise-striding conv
    subsampling x8 -> linear to d 1280 -> 48 Conformer layers (macaron FF 5120, rel-pos MHSA 8 heads, full
    bidirectional context, depthwise conv k=9 + BatchNorm) -> 12.5 Hz (80 ms per position). The ASR decoder is
    never loaded; only ``encoder.*`` tensors of ``model.safetensors`` (bf16 on disk) are read.

    The front-end is a ``CohereFilterbank`` (``check_frontend`` compares it with the remote extractor). BatchNorm layers run in fp32 with frozen running statistics in every training mode;
    their affine weights train with their layer.
    """

    name = "cohere_transcribe"
    lora_targets = ("linear_q", "linear_k", "linear_v", "linear_out", "linear1", "linear2")  # attention + both FF
    DEFAULT_REPO = "CohereLabs/cohere-transcribe-03-2026"
    DEFAULT_REVISION = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"
    FILES = (
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "generation_config.json",
        "configuration_cohere_asr.py",
        "modeling_cohere_asr.py",
        "processing_cohere_asr.py",
        "tokenization_cohere_asr.py",
        "model.safetensors",
    )

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        revision: str | None = DEFAULT_REVISION,
        pretrained: bool = True,
        backbone_config: dict | None = None,
        frontend_config: dict | None = None,
    ):
        super().__init__()
        self.repo, self.revision = repo, revision
        snap = self.snapshot_dir(repo, revision)  # the remote code is always needed to build the module
        if backbone_config is None:
            backbone_config = json.loads((snap / "config.json").read_text())["encoder"]
        if frontend_config is None:
            raw = json.loads((snap / "preprocessor_config.json").read_text())
            frontend_config = {k: v for k, v in raw.items() if k not in ("auto_map", "feature_extractor_type")}
        self._backbone_config = dict(backbone_config)
        self._frontend_config = dict(frontend_config)
        module = self.remote_module(snap)
        cfg = self.remote_config(snap)
        cfg.encoder = dict(backbone_config)
        self.encoder = module.ConformerEncoder(cfg)
        self.d_model = int(backbone_config["d_model"])
        self.n_layers = int(backbone_config["n_layers"])
        fe = self._frontend_config
        if fe.get("normalize", "per_feature") != "per_feature" or int(fe.get("pad_to", 0)) != 0 or int(fe.get("frame_splicing", 1)) != 1:
            raise ValueError(f"unsupported Cohere feature extractor settings: {fe}")
        n_fft, win = int(fe.get("n_fft", 512)), int(fe.get("n_window_size", 400))
        self.filterbank = CohereFilterbank(
            n_fft=n_fft,
            hop=int(fe.get("n_window_stride", 160)),
            win=win,
            preemph=fe.get("preemph", 0.97),
            dither=float(fe.get("dither", CohereFilterbank.DITHER)),
            window=self._default_window(win),
            mel_filters=self._default_mel_filters(fe, n_fft),
        )
        if pretrained:
            self.load_pretrained(snap / "model.safetensors")
        else:
            self._load_frontend_buffers(snap / "model.safetensors")
        for m in self.encoder.modules():
            if type(m) is nn.BatchNorm1d:
                m.__class__ = Fp32BatchNorm1d  # same parameters and state-dict keys, fp32 arithmetic

    # ---- snapshot / remote code ------------------------------------------------------------------------------------
    @classmethod
    def snapshot_dir(cls, repo: str, revision: str | None) -> Path:
        return mirrored_snapshot(repo, revision, cls.FILES)

    @staticmethod
    def remote_module(snap: Path):
        """``modeling_cohere_asr`` from the pinned snapshot, through transformers' dynamic-module loader."""
        import sys

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module("modeling_cohere_asr.ConformerEncoder", str(snap))
        return sys.modules[cls.__module__]

    @staticmethod
    def remote_config(snap: Path):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(str(snap), trust_remote_code=True)

    @staticmethod
    def _default_window(win: int) -> torch.Tensor:
        # The extractor stores its buffers in bf16; round through bf16 to match it bit for bit.
        return torch.hann_window(win, periodic=False).to(torch.bfloat16).float()

    @staticmethod
    def _default_mel_filters(fe: dict, n_fft: int) -> torch.Tensor:
        """Slaney-scale, slaney-normalised mel filterbank (librosa's default, as the extractor builds it), bf16-rounded.

        The checkpoint carries the extractor's own filterbank; ``_load_frontend_buffers`` replaces this default with it.
        """
        from transformers.audio_utils import mel_filter_bank

        sr = int(fe.get("sampling_rate", SAMPLE_RATE))
        fb = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=int(fe.get("feature_size", 128)),
            min_frequency=float(fe.get("lowfreq", 0.0)),
            max_frequency=float(fe.get("highfreq") or sr / 2),
            sampling_rate=sr,
            norm=fe.get("mel_norm", "slaney"),
            mel_scale="slaney",
        )
        return torch.from_numpy(fb).float().T.to(torch.bfloat16).float().contiguous()

    def _load_frontend_buffers(self, path: Path) -> None:
        from safetensors import safe_open

        if not path.exists():
            return
        with safe_open(str(path), "pt") as f:
            keys = set(f.keys())
            if "preprocessor.featurizer.fb" in keys:
                fb = f.get_tensor("preprocessor.featurizer.fb").reshape(-1, 1 + self.filterbank.n_fft // 2)
                self.filterbank.mel_filters = fb.to(torch.bfloat16).float().contiguous()
            if "preprocessor.featurizer.window" in keys:
                self.filterbank.window = f.get_tensor("preprocessor.featurizer.window").to(torch.bfloat16).float()

    def load_pretrained(self, path: Path) -> None:
        """Load only ``encoder.*`` tensors (bf16 on disk); the decoder is never read."""
        from safetensors import safe_open

        state = {}
        with safe_open(str(path), "pt") as f:
            for k in f.keys():
                if k.startswith("encoder."):
                    state[k[len("encoder.") :]] = f.get_tensor(k)
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.endswith("pos_enc.pe")]  # positional table is recomputed
        if missing or unexpected:
            raise RuntimeError(f"Cohere encoder weights do not match the module: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self._load_frontend_buffers(path)

    # ---- interface ------------------------------------------------------------------------------------------------
    def config_dict(self) -> dict:
        return dict(self._backbone_config)

    def frontend_dict(self) -> dict:
        return dict(self._frontend_config)

    def fp32_parameters(self) -> set[int]:
        return {id(p) for m in self.encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) for p in m.parameters()}

    def layer_modules(self) -> list[nn.Module]:
        return list(self.encoder.layers)

    def post_layer_modules(self) -> list[nn.Module]:
        return []  # every Conformer layer ends with its own LayerNorm; there is no separate final norm

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():  # running statistics frozen in every arm; affine weights train with their layer
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def n_frames(self, n_samples: int) -> tuple[int, int]:
        return self.filterbank.n_frames(n_samples)

    def frontend(self, wave: torch.Tensor) -> torch.Tensor:
        """``[B, N]`` float32 16 kHz -> ``[B, 128, 1 + N // 160]`` exactly as ``FilterbankFeatures.forward``; fp32 regardless of autocast."""
        return self.filterbank(wave)

    def check_frontend(self, wave: torch.Tensor) -> float:
        """Max abs difference between ``frontend`` (CPU) and the repo's own feature extractor on the same waveform."""
        from transformers import AutoFeatureExtractor

        fe = AutoFeatureExtractor.from_pretrained(str(self.snapshot_dir(self.repo, self.revision)), trust_remote_code=True)
        ref, _ = fe.filterbank(wave.cpu().float().clone(), torch.tensor([wave.shape[-1]] * wave.shape[0], dtype=torch.float))
        return float((self.frontend(wave.cpu()) - ref).abs().max())

    def forward(self, wave: torch.Tensor, layer: int = -1) -> torch.Tensor:
        feats = self.frontend(wave)  # [B, 128, T]
        enc = self.encoder
        n_valid = self.n_frames(wave.shape[-1])[1]
        lengths = torch.full((feats.shape[0],), n_valid, device=feats.device, dtype=torch.long)
        conv_dtype = enc.pre_encode.conv[0].weight.dtype
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.pre_trainable):
            x, lengths = enc.pre_encode(feats.to(conv_dtype), lengths)
        lengths = lengths.to(torch.int64)
        x, pos_emb = enc.pos_enc(x)
        pad_mask, att_mask = enc._create_masks(padding_length=lengths, max_audio_length=x.size(1), device=x.device)
        layers = self.layer_modules()
        for i in range(self.layers_used(layer)):
            x = self.run_layer(i, layers[i], x, pos_emb, mask=att_mask, pad_mask=pad_mask)
        return x[:, : int(lengths[0].item())]  # positions past the valid length pad the extractor's extra STFT frame


REGISTRY: dict[str, type[Backbone]] = {CohereTranscribeBackbone.name: CohereTranscribeBackbone}


def build_backbone(name: str, **kwargs) -> Backbone:
    if name not in REGISTRY:
        raise KeyError(f"unknown backbone {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
