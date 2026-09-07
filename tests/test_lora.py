"""LoRA wraps frozen linears without changing their output at init and folds back exactly."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from eot.modeling.lora import LoRALinear, apply_lora, lora_parameters, merge_lora


def _block() -> nn.Module:
    torch.manual_seed(3)
    layer = nn.Module()
    layer.linear_q = nn.Linear(8, 6)
    layer.linear_out = nn.Linear(6, 8)
    layer.other = nn.Linear(8, 8)
    return nn.Sequential(layer, nn.Sequential(nn.Linear(8, 4)))


def test_wrapping_is_identity_at_init_and_freezes_base():
    module = _block()
    x = torch.randn(5, 8)
    before = module[0].linear_q(x)
    n = apply_lora(module, r=2, alpha=4.0, targets=("linear_q", "linear_out"))
    assert n == 2
    assert isinstance(module[0].linear_q, LoRALinear) and isinstance(module[0].linear_out, LoRALinear)
    assert isinstance(module[0].other, nn.Linear) and isinstance(module[1][0], nn.Linear)
    torch.testing.assert_close(module[0].linear_q(x), before)
    assert not module[0].linear_q.base.weight.requires_grad
    assert {p.shape for p in lora_parameters(module)} == {(2, 8), (6, 2), (2, 6), (8, 2)}
    with pytest.raises(ValueError):
        apply_lora(module, r=0, targets=("linear_q",))
    with pytest.raises(ValueError):
        apply_lora(module, r=2)


def test_merge_reproduces_adapted_output_and_removes_adapters():
    module = _block()
    apply_lora(module, r=3, alpha=6.0, targets=("linear_q", "linear_out"))
    with torch.no_grad():
        for p in lora_parameters(module):
            p.normal_()
    x = torch.randn(4, 8)
    adapted = module[0].linear_out(module[0].linear_q(x))
    assert merge_lora(module) == 2
    assert not list(lora_parameters(module))
    assert isinstance(module[0].linear_q, nn.Linear)
    torch.testing.assert_close(module[0].linear_out(module[0].linear_q(x)), adapted, rtol=1e-5, atol=1e-6)


def test_adapter_path_is_fp32_and_casts_back_to_the_base_dtype():
    base = nn.Linear(8, 6).to(torch.bfloat16)
    layer = LoRALinear(base, r=2, alpha=2.0)
    assert layer.lora_A.dtype == torch.float32
    out = layer(torch.randn(3, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    assert layer.lora_A.grad is not None and base.weight.grad is None


def test_dropout_only_affects_the_adapter_path_in_training_mode():
    torch.manual_seed(5)
    base = nn.Linear(8, 6)
    layer = LoRALinear(base, r=2, alpha=2.0, dropout=0.5)
    assert isinstance(layer.dropout, nn.Dropout) and isinstance(LoRALinear(nn.Linear(8, 6), r=2).dropout, nn.Identity)
    with torch.no_grad():
        layer.lora_B.normal_()
    x = torch.randn(64, 8)
    layer.eval()
    torch.testing.assert_close(layer(x), layer(x))  # deterministic in eval
    layer.train()
    a, b = layer(x), layer(x)
    assert not torch.equal(a, b)
    # Dropout on the adapter input never touches the frozen base path.
    torch.testing.assert_close(base(x), layer.base(x))
    with pytest.raises(ValueError):
        LoRALinear(base, r=0)


def test_apply_lora_is_idempotent_and_leaves_wrapped_layers_alone():
    module = _block()
    assert apply_lora(module, r=2, targets=("linear_q", "linear_out")) == 2
    with torch.no_grad():
        for p in lora_parameters(module):
            p.normal_()
    x = torch.randn(3, 8)
    before = module[0].linear_out(module[0].linear_q(x))
    assert apply_lora(module, r=4, alpha=8.0, targets=("linear_q", "linear_out")) == 0
    assert module[0].linear_q.r == 2 and not isinstance(module[0].linear_q.base, LoRALinear)
    torch.testing.assert_close(module[0].linear_out(module[0].linear_q(x)), before)
