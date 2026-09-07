"""EOT model: Whisper-Tiny encoder + attention pooling + heads.

Baseline path is the Smart Turn v3 architecture (so a like-for-like comparison is possible):
``WhisperEncoder(max_source_positions=400)`` -> attention pooling -> MLP 384->256->64->1.

Two additions, both optional and both ablatable with a flag:

1. **Agent-context conditioning** (``use_context=True``). The previous agent utterance is known
   *before* the user speaks and costs zero latency ("What is your MC number?" predicts a digit
   string with internal pauses; "Anything else?" predicts a short answer). LiveKit's v1 adapter
   on EoT Bench sends audio only, so this is an unexploited lever the harness permits via
   ``messages``. Text is hashed into n-gram ids -> embedding + masked mean -> gated fusion with the
   pooled audio vector. No tokenizer or extra model download; ONNX-friendly.

2. **Multi-horizon future-speech heads** (``use_fvad=True``). Predict "user speaks again within
   h seconds" for h in ``HORIZONS`` (0.24/0.64/1.2/2.0 s). Targets come free from prefix mining;
   the auxiliary loss regularises the binary EOT decision with a graded temporal signal
   (Next-Turn / DualTurn-FVAD flavour). The heads are also useful at inference for
   *anticipation*: a low P(speech within 2 s) is a cheap speculative-execution trigger.

Training recipe defaults mirror the public Smart Turn ``train.py``: full fine-tune (nothing
frozen), lr 5e-5, cosine, warmup 20 %, weight decay 0.01. Freezing the encoder is exposed as a
flag purely as an ablation; it is *not* the recommended path.

Distillation (``teacher_logits`` in ``forward``) replaces the hard-label EoT loss with the Bernoulli
KL of ``eot.modeling.distill``; the model itself has no extra parameters, so a distilled checkpoint
loads, exports and serves exactly like a hard-label one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eot.context import CTX_LEN, CTX_VOCAB
from eot.labeling.samples import HORIZONS
from eot.modeling.distill import DEFAULT_KD_TEMPERATURE, bernoulli_kl


@dataclass
class EOTConfig:
    base_model: str = "openai/whisper-tiny"  # multilingual: one model for EN + ES bonus
    base_revision: str | None = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
    max_source_positions: int = 400  # 8 s -> 800 mel frames -> 400 encoder positions; smaller = shorter internal context
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
    encoder_layers: int | None = None  # depth of a pruned checkpoint; the training CLI no longer prunes


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
            self.encoder.layers = nn.ModuleList(list(self.encoder.layers)[: cfg.encoder_layers])
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
            nn.Linear(fused, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        if cfg.use_fvad:
            self.fvad_head = nn.Sequential(nn.Linear(fused, 128), nn.GELU(), nn.Linear(128, len(cfg.horizons)))
        for m in list(self.pool.net) + list(self.classifier):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                nn.init.zeros_(m.bias)

    # -- forward -----------------------------------------------------------------------------
    def encode(self, input_features: torch.Tensor, context_ids: torch.Tensor | None = None) -> torch.Tensor:
        # The model's internal context may be shorter than the shared 8 s frontend contract
        # (``max_source_positions`` < 400); only the trailing frames are encoded, so evaluation
        # and serving keep feeding exactly the same audio window. Static shapes let tracing drop
        # the branch, so the default full-window model exports without a Slice node.
        frames = 2 * self.cfg.max_source_positions
        if input_features.shape[-1] > frames:
            input_features = input_features[:, :, -frames:]
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
        weight: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        kd_temperature: float = DEFAULT_KD_TEMPERATURE,
    ) -> dict:
        """Score a batch; with ``labels`` (or ``teacher_logits``) also return the training losses.

        ``teacher_logits`` switches the EoT term to distillation: ``loss_kd`` (pure KL, row-weighted, no
        ``pos_weight``) replaces ``loss_eot`` and the labels are ignored for that term. The future-speech
        auxiliary loss is unchanged in both modes.
        """
        z = self.encode(input_features, context_ids)
        logit = self.classifier(z).squeeze(-1)
        out = {"logit": logit, "p_eot": torch.sigmoid(logit)}
        if self.cfg.use_fvad:
            fl = self.fvad_head(z)
            out["fvad_logit"] = fl
            out["p_fvad"] = torch.sigmoid(fl)
        if teacher_logits is not None:
            loss = bernoulli_kl(logit, teacher_logits, kd_temperature, weight)
            out["loss_kd"] = loss
        elif labels is not None:
            labels = labels.float()
            pos_w = torch.as_tensor(self.cfg.pos_weight, dtype=logit.dtype, device=logit.device)
            if weight is None:
                loss = F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_w)
            else:
                # Per-row weights are relative (weighted mean): all-ones reproduces the plain mean,
                # a zero excludes the row, and the loss scale does not depend on the weight scale.
                w = weight.to(logit.dtype)
                per_row = F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_w, reduction="none")
                loss = (per_row * w).sum() / w.sum().clamp(min=torch.finfo(logit.dtype).tiny)
            out["loss_eot"] = loss
        if teacher_logits is not None or labels is not None:
            if self.cfg.use_fvad and fvad is not None:
                m = (fvad_mask if fvad_mask is not None else torch.ones_like(logit)).float()
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


def save_checkpoint(model: EOTModel, path: str | Path) -> None:
    torch.save({"cfg": asdict(model.cfg), "state_dict": model.state_dict()}, str(path))


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> EOTModel:
    ck = torch.load(str(path), map_location=map_location, weights_only=False)
    cfg = EOTConfig(**{**ck["cfg"], "horizons": tuple(ck["cfg"]["horizons"])})
    model = EOTModel(cfg, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.eval()
