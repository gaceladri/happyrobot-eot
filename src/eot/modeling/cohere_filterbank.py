"""Cohere Transcribe log-mel front-end as a self-contained torch module (no transformers / Hub imports).

Re-implements the repository's ``FilterbankFeatures.forward`` on tensors: 128 log-mel at 100 Hz (25 ms / 10 ms,
n_fft 512, pre-emphasis 0.97, slaney mels, ``log(x + 2^-24)``), per-feature normalisation over the whole window
and a deterministic dither seeded by the sample count. ``CohereTranscribeBackbone`` composes it for training and
``check_frontend`` compares it with the remote extractor; the TensorRT service rebuilds it from the ``frontend.pt``
sidecar (``from_payload``) so serving never constructs the encoder.
"""

from __future__ import annotations

import torch
from torch import nn

PAYLOAD_FORMAT = "L041_COHERE_FRONTEND_V1"


class CohereFilterbank(nn.Module):
    """``[B, N]`` float32 16 kHz -> ``[B, n_mels, 1 + N // hop]``; fp32 regardless of autocast.

    ``window`` and ``mel_filters`` are non-persistent buffers: the extractor's own copies live in the checkpoint,
    so composing this module never changes a backbone's ``state_dict`` keys.
    """

    DITHER = 1e-5
    LOG_ZERO_GUARD = 2.0**-24

    def __init__(
        self,
        *,
        n_fft: int,
        hop: int,
        win: int,
        preemph: float | None,
        dither: float,
        window: torch.Tensor,
        mel_filters: torch.Tensor,
    ):
        super().__init__()
        self.n_fft, self.hop, self.win = int(n_fft), int(hop), int(win)
        self.preemph = None if preemph is None else float(preemph)
        self.dither = float(dither)
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("mel_filters", mel_filters, persistent=False)  # [n_mels, 1 + n_fft // 2]

    @classmethod
    def from_payload(cls, payload: dict) -> CohereFilterbank:
        if payload.get("format") != PAYLOAD_FORMAT:
            raise ValueError(f"unknown frontend payload format {payload.get('format')!r}; expected {PAYLOAD_FORMAT}")
        return cls(**{k: payload[k] for k in ("n_fft", "hop", "win", "preemph", "dither", "window", "mel_filters")})

    def to_payload(self) -> dict:
        return {
            "format": PAYLOAD_FORMAT,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "win": self.win,
            "preemph": self.preemph,
            "dither": self.dither,
            "window": self.window.detach().cpu().clone(),
            "mel_filters": self.mel_filters.detach().cpu().clone(),
        }

    def n_frames(self, n_samples: int) -> tuple[int, int]:
        """``(STFT frames produced, valid frames)``: center=True gives ``1 + N // hop``; the extractor counts ``N // hop`` as valid."""
        valid = (n_samples + 2 * (self.n_fft // 2) - self.n_fft) // self.hop
        return 1 + n_samples // self.hop, valid

    def _dither_noise(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        # The extractor seeds its dither by the number of samples: deterministic, and identical on CPU and GPU.
        g = torch.Generator(device="cpu")
        g.manual_seed(n)
        return torch.randn((n,), dtype=dtype, generator=g)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wave.device.type, enabled=False):
            x = wave.float()
            n = x.shape[-1]
            n_out, n_valid = self.n_frames(n)
            if self.dither > 0:
                x = x + self.dither * self._dither_noise(n, x.dtype).to(x.device)
            if self.preemph is not None:
                x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            st = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop,
                win_length=self.win,
                center=True,
                window=self.window,
                return_complex=True,
                pad_mode="constant",
            )
            mag = torch.sqrt(torch.view_as_real(st).pow(2).sum(-1)).pow(2.0)
            log = torch.log(torch.matmul(self.mel_filters, mag) + self.LOG_ZERO_GUARD)
            v = log[..., :n_valid]
            mean = v.mean(dim=2, keepdim=True)
            std = torch.sqrt(((v - mean) ** 2).sum(dim=2, keepdim=True) / (n_valid - 1.0))
            std = std.masked_fill(std.isnan(), 0.0) + self.DITHER
            out = (log - mean) / std
            if n_out > n_valid:
                out = torch.cat([out[..., :n_valid], torch.zeros_like(out[..., n_valid:])], dim=2)
            return out
