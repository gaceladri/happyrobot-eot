import torch

from conftest import tiny_model


def test_short_context_uses_only_configured_tail():
    model = tiny_model(max_source_positions=100)
    x = torch.randn(2, 80, 800)
    changed = x.clone()
    changed[:, :, :600] = 100
    with torch.no_grad():
        torch.testing.assert_close(model(x)["p_eot"], model(changed)["p_eot"], rtol=0, atol=0)
        torch.testing.assert_close(model(x)["p_eot"], model(x[:, :, -200:])["p_eot"], rtol=0, atol=0)
