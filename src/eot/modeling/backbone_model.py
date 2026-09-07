"""Backbone encoder + the package's attention pool + MLP head, with the window crop inside the model.

The model receives the shared 8 s waveform window (16 kHz float32, left zero-padded like
``eot.audio.last_window``) and crops its trailing ``window_s`` seconds itself, exactly as the Whisper
model crops trailing mel positions inside ``EOTModel.encode``: the harness, the Krisp scorer, the
held-out report and the trainer all feed the same 8 s of audio.

Checkpoint formats (``save_checkpoint`` / ``save_delta_checkpoint`` / ``load_checkpoint``):

    full   ``{"format": "eot-backbone-v1", "cfg", "state_dict" (fp32), "trainable"}``
    delta  ``{"format": "eot-backbone-delta-v1", "cfg", "state_dict": trainable tensors only (LoRA A/B, unfrozen layers,
           pool, head), "trainable", "base_weights_sha256"}`` - the frozen encoder tensors are re-read from the pinned
           pretrained snapshot, so the delta rebuilds the same fp32 model (``check_delta`` asserts it). A LoRA delta of
           the 1.9 B Cohere encoder is ~120 MB instead of 7.6 GB and is what travels back from a training pod. Loading
           a delta refuses a snapshot whose ``model.safetensors`` sha256 differs from the recorded one.

The L031 research formats (``l031-backbone-v1``, ``l028-backbone-v1``, ``l031-delta-v1``) are read as well, so the
trained teacher checkpoints load unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eot.io import cached_sha256
from eot.modeling.backbones import CohereTranscribeBackbone, build_backbone
from eot.modeling.lora import apply_lora, lora_parameters
from eot.modeling.model import AttentionPool
from eot.modeling.training_utils import atomic_torch_save

FORMAT_FULL = "eot-backbone-v1"
FORMAT_DELTA = "eot-backbone-delta-v1"
FORMATS_FULL = (FORMAT_FULL, "l031-backbone-v1", "l028-backbone-v1")
FORMATS_DELTA = (FORMAT_DELTA, "l031-delta-v1")
SAMPLE_RATE = 16_000
FROZEN_DTYPES = ("bfloat16", "float32")


@dataclass
class BackboneEOTConfig:
    backbone: str = CohereTranscribeBackbone.name
    repo: str = CohereTranscribeBackbone.DEFAULT_REPO
    revision: str | None = CohereTranscribeBackbone.DEFAULT_REVISION
    layer: int = -1  # -1 = last layer; k = hidden state after k layers (1-based)
    window_s: float = 4.0  # trailing seconds encoded (crop inside the model)
    input_s: float = 8.0  # the shared audio interface every scorer feeds
    freeze_encoder: bool = False
    unfreeze_top_k: int | None = None  # None = every used layer trainable (unless freeze_encoder / LoRA)
    lora_r: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    pool_hidden: int = 256
    pos_weight: float = 1.0
    frozen_dtype: str = "bfloat16"  # storage dtype of frozen encoder parameters while training
    backbone_config: dict | None = None  # stored so a checkpoint rebuilds without the Hub config
    frontend_config: dict | None = None

    def __post_init__(self) -> None:
        if self.freeze_encoder and (self.lora_r > 0 or self.unfreeze_top_k):
            raise ValueError("freeze_encoder cannot be combined with lora_r > 0 or unfreeze_top_k")
        if self.unfreeze_top_k is not None and self.unfreeze_top_k < 1:
            raise ValueError(f"unfreeze_top_k must be None or >= 1, got {self.unfreeze_top_k}")
        if self.lora_r < 0:
            raise ValueError(f"lora_r must be >= 0, got {self.lora_r}")
        if not 0.0 <= self.lora_dropout <= 1.0:
            raise ValueError(f"lora_dropout must lie in [0, 1], got {self.lora_dropout}")
        if self.frozen_dtype not in FROZEN_DTYPES:
            raise ValueError(f"frozen_dtype must be one of {FROZEN_DTYPES}, got {self.frozen_dtype!r}")


class BackboneEOTModel(nn.Module):
    def __init__(self, cfg: BackboneEOTConfig, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(
            cfg.backbone,
            repo=cfg.repo,
            revision=cfg.revision,
            pretrained=pretrained,
            backbone_config=cfg.backbone_config,
            frontend_config=cfg.frontend_config,
        )
        cfg.backbone_config = self.backbone.config_dict()
        cfg.frontend_config = self.backbone.frontend_dict()
        d = self.backbone.d_model
        self.pool = AttentionPool(d, cfg.pool_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(d, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        for m in list(self.pool.net) + list(self.classifier):  # EOTModel's head initialisation
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                nn.init.zeros_(m.bias)
        self.n_lora = 0
        if cfg.lora_r > 0:
            targets = self.backbone.lora_targets
            if not targets:
                raise ValueError(f"backbone {cfg.backbone!r} declares no LoRA targets")
            layers = self.backbone.layer_modules()
            first, n_used = self._trainable_layer_range()
            self.n_lora = sum(apply_lora(layers[i], cfg.lora_r, cfg.lora_alpha, targets, cfg.lora_dropout) for i in range(first, n_used))
        self.trainable = self.configure_trainable()

    # ---- freezing ---------------------------------------------------------------------------------------------------
    def _trainable_layer_range(self) -> tuple[int, int]:
        n_used = self.backbone.layers_used(self.cfg.layer)
        first = max(0, n_used - int(self.cfg.unfreeze_top_k)) if self.cfg.unfreeze_top_k else 0
        return first, n_used

    def configure_trainable(self) -> dict:
        """Apply the freezing mode of ``cfg`` and cast frozen encoder tensors to ``frozen_dtype``; returns a summary."""
        cfg, bb = self.cfg, self.backbone
        layers = bb.layer_modules()
        first, n_used = self._trainable_layer_range()
        for p in bb.parameters():
            p.requires_grad_(False)
        bb.pre_trainable, bb.first_trainable = False, None
        if cfg.lora_r > 0:
            for p in lora_parameters(bb):
                p.requires_grad_(True)
            bb.first_trainable = first
            mode = f"lora r={cfg.lora_r} on layers {first}..{n_used - 1}"
        elif cfg.freeze_encoder:
            mode = "frozen"
        elif cfg.unfreeze_top_k:
            for i in range(first, n_used):
                for p in layers[i].parameters():
                    p.requires_grad_(True)
            bb.first_trainable = first
            mode = f"top {n_used - first} of {n_used} layers ({first}..{n_used - 1})"
        else:
            for p in bb.parameters():
                p.requires_grad_(True)
            for i in range(n_used, len(layers)):  # layers above the readout are never used
                for p in layers[i].parameters():
                    p.requires_grad_(False)
            bb.pre_trainable, bb.first_trainable = True, 0
            mode = "full"
        if cfg.lora_r > 0 or cfg.unfreeze_top_k:
            for m in bb.post_layer_modules():  # a final norm trains only when the readout is the last layer
                for p in m.parameters():
                    p.requires_grad_(cfg.layer == -1)
        frozen_dtype = getattr(torch, cfg.frozen_dtype)
        keep_fp32 = bb.fp32_parameters()
        for p in bb.parameters():
            p.data = p.data.to(torch.float32 if (p.requires_grad or id(p) in keep_fp32) else frozen_dtype)
        for p in self.head_parameters():
            p.requires_grad_(True)
            p.data = p.data.float()
        enc_train = sum(p.numel() for p in bb.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head_parameters())
        return {
            "mode": mode,
            "encoder_params": bb.encoder_param_count(),
            "encoder_trainable": enc_train,
            "head_params": head,
            "trainable_total": enc_train + head,
            "first_trainable_layer": bb.first_trainable,
            "pre_layers_trainable": bb.pre_trainable,
            "n_layers_used": n_used,
            "readout_layer": cfg.layer,
            "lora_layers": self.n_lora,
        }

    def encoder_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.pool.parameters()) + list(self.classifier.parameters())

    def set_gradient_checkpointing(self, on: bool) -> None:
        self.backbone.grad_checkpointing = bool(on)

    # ---- forward ----------------------------------------------------------------------------------------------------
    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        n = int(round(self.cfg.window_s * SAMPLE_RATE))
        if wave.shape[-1] > n:
            wave = wave[:, -n:]
        return self.pool(self.backbone(wave, layer=self.cfg.layer))  # [B, T, D] -> [B, D]

    def forward(self, wave: torch.Tensor, labels: torch.Tensor | None = None, weight: torch.Tensor | None = None) -> dict:
        logit = self.classifier(self.encode(wave)).squeeze(-1).float()
        out = {"logit": logit, "p_eot": torch.sigmoid(logit)}
        if labels is not None:
            labels = labels.float()
            pos_w = torch.as_tensor(self.cfg.pos_weight, dtype=logit.dtype, device=logit.device)
            if weight is None:
                loss = F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_w)
            else:
                w = weight.to(logit.dtype)
                per_row = F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_w, reduction="none")
                loss = (per_row * w).sum() / w.sum().clamp(min=torch.finfo(logit.dtype).tiny)
            out["loss"] = loss
        return out


# ---- checkpoints -----------------------------------------------------------------------------------------------------
def trainable_state(model: BackboneEOTModel) -> dict[str, torch.Tensor]:
    """Every parameter that receives gradients (unfrozen encoder tensors, LoRA A/B, pool, head), fp32 on CPU."""
    return {n: p.detach().to("cpu", torch.float32) for n, p in model.named_parameters() if p.requires_grad}


def base_weights_sha256(snapshot: Path) -> str:
    """sha256 of the snapshot's ``model.safetensors``, memoised next to it (a multi-GB file is hashed once per host)."""
    weights = Path(snapshot) / "model.safetensors"
    return cached_sha256(weights, weights.with_name("model.safetensors.sha256.json"))


def save_checkpoint(model: BackboneEOTModel, path: str | Path, dtype: torch.dtype = torch.float32) -> None:
    """Atomically write the full model (``FORMAT_FULL``); floating tensors are stored as ``dtype`` on CPU."""
    state = {k: (v.detach().to("cpu", dtype) if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}
    atomic_torch_save({"format": FORMAT_FULL, "cfg": asdict(model.cfg), "state_dict": state, "trainable": model.trainable}, Path(path))


def save_delta_checkpoint(model: BackboneEOTModel, path: str | Path, base_weights_sha256: str | None = None) -> None:
    """Atomically write the trainable tensors only (``FORMAT_DELTA``) and the sha256 of the base weights they apply to."""
    atomic_torch_save(
        {
            "format": FORMAT_DELTA,
            "cfg": asdict(model.cfg),
            "state_dict": trainable_state(model),
            "trainable": model.trainable,
            "base_weights_sha256": base_weights_sha256,
        },
        Path(path),
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> BackboneEOTModel:
    """Rebuild an fp32 evaluation model from a full or delta checkpoint (deltas read the pinned pretrained snapshot)."""
    ck = torch.load(str(path), map_location=map_location, weights_only=False)
    fmt = ck.get("format")
    cfg = BackboneEOTConfig(**ck["cfg"])
    if fmt in FORMATS_FULL:
        model = BackboneEOTModel(cfg, pretrained=False)
        model.load_state_dict(ck["state_dict"], strict=True)
    elif fmt in FORMATS_DELTA:
        model = BackboneEOTModel(cfg, pretrained=True)
        recorded = ck.get("base_weights_sha256")
        if recorded is not None:
            actual = base_weights_sha256(model.backbone.snapshot_dir(cfg.repo, cfg.revision))
            if actual != recorded:
                raise ValueError(f"delta checkpoint was trained on base weights {recorded[:12]}, the snapshot holds {actual[:12]}")
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        if names != set(ck["state_dict"]):
            raise ValueError(
                f"delta checkpoint tensors do not match the trainable set: missing {sorted(names - set(ck['state_dict']))[:5]}, "
                f"unexpected {sorted(set(ck['state_dict']) - names)[:5]}"
            )
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        if unexpected or set(missing) & names:
            raise ValueError(f"delta checkpoint could not be applied: {unexpected[:5]} {sorted(set(missing) & names)[:5]}")
    else:
        raise ValueError(f"unknown backbone checkpoint format {fmt!r}")
    model.float()
    model.requires_grad_(False)
    return model.eval()


def check_delta(full_path: str | Path, delta_path: str | Path) -> float:
    """Max abs difference between the model rebuilt from the delta and the full checkpoint (expected 0.0)."""
    a, b = load_checkpoint(full_path).state_dict(), load_checkpoint(delta_path).state_dict()
    if set(a) != set(b):
        raise ValueError(f"state dicts differ: {sorted(set(a) ^ set(b))[:5]}")
    return max(float((a[k].float() - b[k].float()).abs().max()) if a[k].is_floating_point() else float((a[k] != b[k]).any()) for k in a)


def is_backbone_checkpoint(path: str | Path) -> bool:
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 - any unreadable file is simply not a backbone checkpoint
        return False
    return isinstance(ck, dict) and ck.get("format") in FORMATS_FULL + FORMATS_DELTA
