"""EOT model: Whisper-Tiny encoder + attention pooling + heads.

Baseline path is the Smart Turn v3 architecture (so a like-for-like comparison is possible):
``WhisperEncoder(max_source_positions=400)`` -> attention pooling -> MLP 384->256->64->1.

Two additions, both optional and both ablatable with a flag:

1. **Agent-context conditioning** (``use_context=True``). The previous agent utterance is known
   *before* the user speaks and costs zero latency ("What is your MC number?" predicts a digit
   string with internal pauses; "Anything else?" predicts a short answer). LiveKit's v1 adapter
   on EoT Bench sends audio only, so this is an unexploited lever the harness permits via
   ``messages``. Text is hashed into n-gram ids -> EmbeddingBag -> gated fusion with the pooled
   audio vector. No tokenizer or extra model download; ONNX-friendly.

2. **Multi-horizon future-speech heads** (``use_fvad=True``). Predict "user speaks again within
   h seconds" for h in ``HORIZONS`` (0.24/0.64/1.2/2.0 s). Targets come free from prefix mining;
   the auxiliary loss regularises the binary EOT decision with a graded temporal signal
   (Next-Turn / DualTurn-FVAD flavour). The heads are also useful at inference for
   *anticipation*: a low P(speech within 2 s) is a cheap speculative-execution trigger.

Training recipe defaults mirror the public Smart Turn ``train.py``: full fine-tune (nothing
frozen), lr 5e-5, cosine, warmup 20 %, weight decay 0.01. Freezing the encoder is exposed as a
flag purely as an ablation; it is *not* the recommended path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context import CTX_LEN, CTX_VOCAB, hash_context
from .prefix_mining import HORIZONS


@dataclass
class EOTConfig:
    base_model: str = "openai/whisper-tiny"  # multilingual: one model for EN + ES bonus
    base_revision: str | None = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
    max_source_positions: int = 400  # 8 s -> 800 mel frames -> 400 encoder positions
    use_context: bool = False
    ctx_dim: int = 128
    context_dropout: float = 0.3  # train-time: drop context so the model stays usable without it
    use_fvad: bool = True
    fvad_weight: float = 0.5
    pos_weight: float = 1.0
    freeze_encoder: bool = False  # ablation only
    horizons: tuple[float, ...] = field(default_factory=lambda: HORIZONS)
    # Stored in checkpoints so restoring a trained model never needs the Hub.
    whisper_config: dict | None = None
    normalize_audio: bool = False
    encoder_layers: int | None = None


class AttentionPool(nn.Module):
    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # [B, T, D] -> [B, D]
        w = torch.softmax(self.net(h), dim=1)
        return (h * w).sum(dim=1)


class EOTModel(nn.Module):
    def __init__(self, cfg: EOTConfig, pretrained: bool = True):
        super().__init__()
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        self.cfg = cfg
        if cfg.whisper_config is None:
            wcfg = WhisperConfig.from_pretrained(cfg.base_model, revision=cfg.base_revision)
            cfg.whisper_config = wcfg.to_dict()
        else:
            wcfg = WhisperConfig.from_dict(cfg.whisper_config)
        self.encoder = WhisperEncoder(wcfg)
        if pretrained:
            from transformers import WhisperModel

            full = WhisperModel.from_pretrained(cfg.base_model, revision=cfg.base_revision)
            self.encoder.load_state_dict(full.encoder.state_dict())
            del full
        # Shrink positional table to the 8 s window instead of re-initialising it.
        pos = self.encoder.embed_positions.weight.data[: cfg.max_source_positions].clone()
        self.encoder.embed_positions = nn.Embedding(cfg.max_source_positions, wcfg.d_model)
        self.encoder.embed_positions.weight.data.copy_(pos)
        self.encoder.embed_positions.weight.requires_grad_(False)
        self.encoder.config.max_source_positions = cfg.max_source_positions
        if cfg.encoder_layers is not None:
            if not 1 <= cfg.encoder_layers <= len(self.encoder.layers):
                raise ValueError("encoder_layers must be between 1 and the backbone depth")
            self.encoder.layers = nn.ModuleList(list(self.encoder.layers)[:cfg.encoder_layers])
        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        d = wcfg.d_model
        self.pool = AttentionPool(d)
        fused = d
        if cfg.use_context:
            # Embedding + masked mean instead of EmbeddingBag(padding_idx): identical maths, ONNX-exportable.
            self.ctx_emb = nn.Embedding(CTX_VOCAB, cfg.ctx_dim, padding_idx=0)
            self.ctx_gate = nn.Sequential(nn.Linear(d + cfg.ctx_dim, d), nn.Sigmoid())
            self.ctx_proj = nn.Linear(cfg.ctx_dim, d)
        self.classifier = nn.Sequential(
            nn.Linear(fused, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )
        if cfg.use_fvad:
            self.fvad_head = nn.Sequential(nn.Linear(fused, 128), nn.GELU(), nn.Linear(128, len(cfg.horizons)))
        for m in list(self.pool.net) + list(self.classifier):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                nn.init.zeros_(m.bias)

    # -- forward -----------------------------------------------------------------------------
    def encode(self, input_features: torch.Tensor, context_ids: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(input_features=input_features).last_hidden_state
        z = self.pool(h)
        if self.cfg.use_context:
            if context_ids is None:
                context_ids = torch.zeros(z.shape[0], CTX_LEN, dtype=torch.long, device=z.device)
            if self.training and self.cfg.context_dropout > 0:
                keep = (torch.rand(z.shape[0], 1, device=z.device) > self.cfg.context_dropout).long()
                context_ids = context_ids * keep
            mask = (context_ids > 0).float().unsqueeze(-1)  # [B, L, 1]
            c = (self.ctx_emb(context_ids) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            has_ctx = (context_ids.sum(dim=1, keepdim=True) > 0).float()
            g = self.ctx_gate(torch.cat([z, c], dim=-1)) * has_ctx
            z = z + g * self.ctx_proj(c)
        return z

    def forward(
        self,
        input_features: torch.Tensor,
        context_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        fvad: torch.Tensor | None = None,
        fvad_mask: torch.Tensor | None = None,
    ) -> dict:
        z = self.encode(input_features, context_ids)
        logit = self.classifier(z).squeeze(-1)
        out = {"logit": logit, "p_eot": torch.sigmoid(logit)}
        if self.cfg.use_fvad:
            fl = self.fvad_head(z)
            out["fvad_logit"] = fl
            out["p_fvad"] = torch.sigmoid(fl)
        if labels is not None:
            labels = labels.float()
            pos_w = torch.as_tensor(self.cfg.pos_weight, dtype=logit.dtype, device=logit.device)
            loss = F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_w)
            out["loss_eot"] = loss
            if self.cfg.use_fvad and fvad is not None:
                m = (fvad_mask if fvad_mask is not None else torch.ones_like(labels)).float()
                if m.ndim == 1:
                    m = m.unsqueeze(-1).expand_as(out["fvad_logit"])
                l_f = F.binary_cross_entropy_with_logits(out["fvad_logit"], fvad.float(), reduction="none")
                l_f = (l_f * m).sum() / m.sum().clamp(min=1)
                out["loss_fvad"] = l_f
                loss = loss + self.cfg.fvad_weight * l_f
            out["loss"] = loss
        return out

    # -- helpers -----------------------------------------------------------------------------
    def num_params(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))


class ExportWrapper(nn.Module):
    """Fixed-signature module for ONNX: (input_features[B,80,800], context_ids[B,32]) -> (p_eot[B], p_fvad[B,H])."""

    def __init__(self, model: EOTModel):
        super().__init__()
        self.m = model

    def forward(self, input_features: torch.Tensor, context_ids: torch.Tensor):
        out = self.m(input_features, context_ids if self.m.cfg.use_context else None)
        p_fvad = out.get("p_fvad")
        if p_fvad is None:
            p_fvad = torch.zeros(input_features.shape[0], len(self.m.cfg.horizons), device=input_features.device)
        return out["p_eot"], p_fvad


def save_checkpoint(model: EOTModel, path: str) -> None:
    from dataclasses import asdict

    torch.save({"cfg": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> EOTModel:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = EOTConfig(**{**ck["cfg"], "horizons": tuple(ck["cfg"]["horizons"])})
    model = EOTModel(cfg, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


# Compatibility for callers that historically imported context helpers from this module.
__all__ = [
    "CTX_LEN", "CTX_VOCAB", "hash_context", "EOTConfig", "EOTModel", "ExportWrapper",
    "save_checkpoint", "load_checkpoint",
]
