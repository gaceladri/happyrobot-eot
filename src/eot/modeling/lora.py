"""Minimal LoRA (Hu et al., 2021) on ``nn.Linear`` without a ``peft`` dependency.

``y = W x + (alpha / r) * B (A x)`` with ``A`` Kaiming-uniform and ``B`` zero, so a freshly wrapped
layer computes exactly what the frozen layer did. Adapters are kept in fp32 (master weights); the
frozen base may live in bf16. ``merge_lora`` folds the adapters back into plain ``nn.Linear`` layers
for inference, so a served model pays no extra matmul.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = int(r), float(alpha)
        self.scaling = self.alpha / self.r
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        # The adapter path runs in the adapters' dtype (fp32 masters; bf16 under autocast, where
        # F.linear casts anyway) and is added back in y's dtype.
        h = F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A)
        return y + (F.linear(h, self.lora_B) * self.scaling).to(y.dtype)

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scaling


def apply_lora(module: nn.Module, r: int, alpha: float = 16.0, targets: Iterable[str] = (), dropout: float = 0.0) -> int:
    """Wrap every ``nn.Linear`` child whose attribute name is in ``targets`` (recursively).

    Returns the number of layers wrapped. Already wrapped layers are left alone.
    """
    targets = set(targets)
    if not targets:
        raise ValueError("apply_lora needs at least one target attribute name")
    if r <= 0:
        raise ValueError(f"LoRA rank must be positive, got {r}")
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r, alpha, dropout))
            n += 1
        elif not isinstance(child, LoRALinear):
            n += apply_lora(child, r, alpha, targets, dropout)
    return n


def merge_lora(module: nn.Module) -> int:
    """Fold every ``LoRALinear`` into its base layer (``W += scaling * B A``) and put the plain layer back.

    Returns the number of layers merged. The result is a regular module with no LoRA parameters.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            with torch.no_grad():
                child.base.weight.copy_(child.merged_weight())
            setattr(module, name, child.base)
            n += 1
        else:
            n += merge_lora(child)
    return n


def is_lora_parameter_name(name: str) -> bool:
    return name.endswith(("lora_A", "lora_B"))


def lora_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    """The adapter tensors (``lora_A`` / ``lora_B``) of every ``LoRALinear`` under ``module``."""
    for name, p in module.named_parameters():
        if is_lora_parameter_name(name):
            yield p
