import pytest
import torch

from conftest import tiny_model
from eot.modeling.model import EOTConfig, EOTModel
from eot.modeling.soup import average_models


def _params(model):
    return {k: v.clone() for k, v in model.state_dict().items() if v.is_floating_point()}


def test_alpha_endpoints_reproduce_parents_and_midpoint_is_mean():
    left, right = tiny_model(1), tiny_model(2)
    lp, rp = _params(left), _params(right)
    mid = _params(average_models(tiny_model(1), tiny_model(2), 0.5))
    for key in lp:
        torch.testing.assert_close(mid[key], (lp[key] + rp[key]) / 2)
    torch.testing.assert_close(_params(average_models(tiny_model(1), tiny_model(2), 0.0)), lp)
    torch.testing.assert_close(_params(average_models(tiny_model(1), tiny_model(2), 1.0)), rp)


def test_mismatched_configs_or_alpha_are_rejected():
    with pytest.raises(ValueError, match="alpha"):
        average_models(tiny_model(1), tiny_model(2), 1.5)
    other = EOTModel(EOTConfig(use_fvad=True, whisper_config=tiny_model().cfg.whisper_config), pretrained=False)
    with pytest.raises(ValueError, match="config"):
        average_models(tiny_model(1), other, 0.5)
