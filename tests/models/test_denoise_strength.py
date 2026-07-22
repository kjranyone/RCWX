"""Denoiser strength behavior without loading pretrained weights."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from rcwx.audio.denoise import MLDenoiser

denoise_module = importlib.import_module("rcwx.audio.denoise")


class _HalfAmplitudeModel(torch.nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio * 0.5


@pytest.mark.parametrize(
    ("strength", "expected_scale"),
    [(0.5, 0.75), (1.0, 0.5), (1.5, 0.375), (2.0, 0.25)],
)
def test_ml_strength_blends_first_and_second_pass(strength: float, expected_scale: float):
    denoiser = MLDenoiser(device="cpu")
    denoiser._model = _HalfAmplitudeModel()
    denoiser._loaded = True
    audio = np.linspace(-0.8, 0.8, 32, dtype=np.float32)

    output = denoiser.process(audio, strength=strength)

    np.testing.assert_allclose(output, audio * expected_scale, atol=1e-6)


def test_spectral_strength_scales_gate_parameters(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class _FakeSpectralDenoiser:
        def __init__(self, sample_rate, config):
            captured["sample_rate"] = sample_rate
            captured["config"] = config

        def enable_auto_learn(self):
            pass

        def process(self, audio):
            return audio

    monkeypatch.setattr(denoise_module, "SpectralGateDenoiser", _FakeSpectralDenoiser)
    audio = np.ones(32, dtype=np.float32)

    output = denoise_module.denoise(audio, method="spectral", strength=1.5)

    assert output is audio
    assert captured["sample_rate"] == 16000
    assert captured["config"].threshold_db == 9.0
    assert captured["config"].reduction_db == -36.0
