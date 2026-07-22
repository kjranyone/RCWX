"""Regression tests for the device-side streaming output resampler."""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import resample_poly

from rcwx.pipeline.inference import RVCPipeline


def test_streaming_output_resampler_preserves_length_and_waveform() -> None:
    pipeline = RVCPipeline.__new__(RVCPipeline)
    pipeline.sample_rate = 40000
    pipeline.device = "cpu"
    pipeline._streaming_output_resamplers = {}

    phase = torch.arange(2400, dtype=torch.float32) / pipeline.sample_rate
    source = torch.sin(2 * torch.pi * 440 * phase).unsqueeze(0)
    output = pipeline._resample_streaming_output(source, 48000)

    assert output.shape == (1, 2880)
    assert 48000 in pipeline._streaming_output_resamplers

    reference = resample_poly(source.numpy()[0], 6, 5).astype(np.float32)
    correlation = np.corrcoef(output.numpy()[0], reference)[0, 1]
    assert correlation > 0.999
