import torch

from eot.modeling.model import EOTConfig, EOTModel

TINY = dict(d_model=16, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32, num_mel_bins=80, max_source_positions=1500)


def tiny_model(seed: int = 17, **overrides) -> EOTModel:
    torch.manual_seed(seed)
    return EOTModel(EOTConfig(use_fvad=False, whisper_config=TINY, **overrides), pretrained=False).eval()


def test_short_context_uses_only_configured_tail():
    model = tiny_model(max_source_positions=100)
    x = torch.randn(2, 80, 800)
    changed = x.clone()
    changed[:, :, :600] = 100
    with torch.no_grad():
        torch.testing.assert_close(model(x)["p_eot"], model(changed)["p_eot"], rtol=0, atol=0)
        torch.testing.assert_close(model(x)["p_eot"], model(x[:, :, -200:])["p_eot"], rtol=0, atol=0)
