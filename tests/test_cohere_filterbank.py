"""``CohereFilterbank`` reproduces the former inline backbone front-end bit for bit and round-trips its sidecar payload."""

from __future__ import annotations

import pytest
import torch

from eot.modeling.cohere_filterbank import PAYLOAD_FORMAT, CohereFilterbank


def _reference_frontend(fb: CohereFilterbank, wave: torch.Tensor) -> torch.Tensor:
    """The ``CohereTranscribeBackbone.frontend`` body as it was before the extraction (kept as the numerical oracle)."""
    x = wave.float()
    n = x.shape[-1]
    n_out, n_valid = 1 + n // fb.hop, (n + 2 * (fb.n_fft // 2) - fb.n_fft) // fb.hop
    g = torch.Generator(device="cpu")
    g.manual_seed(n)
    x = x + fb.dither * torch.randn((n,), dtype=x.dtype, generator=g)
    x = torch.cat((x[:, :1], x[:, 1:] - float(fb.preemph) * x[:, :-1]), dim=1)
    st = torch.stft(
        x, n_fft=fb.n_fft, hop_length=fb.hop, win_length=fb.win, center=True, window=fb.window, return_complex=True, pad_mode="constant"
    )
    mag = torch.sqrt(torch.view_as_real(st).pow(2).sum(-1)).pow(2.0)
    log = torch.log(torch.matmul(fb.mel_filters, mag) + 2.0**-24)
    v = log[..., :n_valid]
    mean = v.mean(dim=2, keepdim=True)
    std = torch.sqrt(((v - mean) ** 2).sum(dim=2, keepdim=True) / (n_valid - 1.0))
    std = std.masked_fill(std.isnan(), 0.0) + 1e-5
    out = (log - mean) / std
    if n_out > n_valid:
        out = torch.cat([out[..., :n_valid], torch.zeros_like(out[..., n_valid:])], dim=2)
    return out


@pytest.fixture
def filterbank() -> CohereFilterbank:
    generator = torch.Generator().manual_seed(7)
    return CohereFilterbank(
        n_fft=512,
        hop=160,
        win=400,
        preemph=0.97,
        dither=1e-5,
        window=torch.hann_window(400, periodic=False).to(torch.bfloat16).float(),
        mel_filters=torch.rand(128, 257, generator=generator).to(torch.bfloat16).float(),
    ).eval()


@pytest.mark.parametrize("n_samples", [64_000, 16_000, 1_601])
def test_matches_the_former_inline_frontend_bit_for_bit(filterbank, n_samples):
    wave = torch.randn(2, n_samples, generator=torch.Generator().manual_seed(n_samples))
    out = filterbank(wave)
    assert out.shape == (2, 128, filterbank.n_frames(n_samples)[0])
    assert torch.equal(out, _reference_frontend(filterbank, wave))


def test_payload_round_trip_and_format_check(filterbank):
    payload = filterbank.to_payload()
    assert payload["format"] == PAYLOAD_FORMAT
    rebuilt = CohereFilterbank.from_payload(payload).eval()
    wave = torch.randn(1, 8_000, generator=torch.Generator().manual_seed(0))
    assert torch.equal(rebuilt(wave), filterbank(wave))
    assert rebuilt.state_dict() == {}  # non-persistent buffers only: composing it never changes a checkpoint's keys
    with pytest.raises(ValueError, match="format"):
        CohereFilterbank.from_payload({**payload, "format": "other"})
