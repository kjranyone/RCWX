"""SwiftF0 F0 extraction model wrapper.

SwiftF0 is a lightweight ONNX-based pitch estimator (95.8K params, MIT license)
that runs on CPU via ONNX Runtime. Ideal for parallel extraction with HuBERT
on XPU since it doesn't compete for GPU resources.

Reference: https://github.com/lars76/swift-f0
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SWIFTF0_AVAILABLE = False
try:
    from swift_f0 import SwiftF0 as _SwiftF0Detector
    _SWIFTF0_AVAILABLE = True
except ImportError:
    pass


def is_swiftf0_available() -> bool:
    """Check if SwiftF0 is available."""
    return _SWIFTF0_AVAILABLE


class SwiftF0:
    """SwiftF0 F0 extraction model.

    A very fast CPU-based alternative using ONNX Runtime.
    Internal hop is 256 samples (16ms @ 16kHz); output is resampled
    to match RCWX's standard 160-sample hop (10ms @ 16kHz, 100fps).
    """

    INTERNAL_HOP = 256  # SwiftF0's native hop length at 16kHz
    TARGET_HOP = 160    # RCWX standard F0 hop (100fps at 16kHz)

    def __init__(
        self,
        hop_length: int = 160,
        confidence_threshold: float = 0.5,
    ):
        if not _SWIFTF0_AVAILABLE:
            raise ImportError(
                "swift-f0 is not installed. Install with: pip install swift-f0"
            )

        self.hop_length = hop_length
        self.sample_rate = 16000
        self.confidence_threshold = confidence_threshold

        logger.info(
            f"Loading SwiftF0 (ONNX/CPU, confidence_threshold={confidence_threshold})"
        )
        self.detector = _SwiftF0Detector(confidence_threshold=confidence_threshold)
        logger.info("SwiftF0 model loaded")

    @torch.no_grad()
    def infer(
        self,
        audio: torch.Tensor,
        threshold: float = 0.5,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
    ) -> torch.Tensor:
        """Extract F0 from audio.

        Args:
            audio: Audio tensor [B, T] or [T] at 16kHz
            threshold: Voicing confidence threshold
            f0_min: Minimum F0 frequency in Hz
            f0_max: Maximum F0 frequency in Hz

        Returns:
            F0 tensor [B, T_frames] where unvoiced frames are 0
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        batch_size = audio.shape[0]
        device = audio.device
        n_samples = audio.shape[1]

        # Expected output frames at RCWX's 160-hop rate
        n_target_frames = n_samples // self.TARGET_HOP

        results = []
        for b in range(batch_size):
            audio_np = audio[b].detach().cpu().float().numpy()

            # Silence detection (same as FCPE)
            if np.max(np.abs(audio_np)) < 1e-6:
                results.append(np.zeros(n_target_frames, dtype=np.float32))
                continue

            # Run SwiftF0 detection
            try:
                result = self.detector.detect_from_array(
                    audio_np, sample_rate=self.sample_rate
                )
            except Exception as e:
                logger.warning(f"SwiftF0 inference failed: {e}")
                results.append(np.zeros(n_target_frames, dtype=np.float32))
                continue

            # Apply voicing mask + frequency range filter
            f0 = result.pitch_hz.copy()
            voiced = result.voicing & (f0 >= f0_min) & (f0 <= f0_max)

            # Re-apply threshold if different from constructor
            if abs(threshold - self.confidence_threshold) > 1e-6:
                voiced = voiced & (result.confidence >= threshold)

            f0[~voiced] = 0.0

            # Resample from 256-hop frames to 160-hop frames
            n_swift_frames = len(f0)
            if n_swift_frames == 0:
                results.append(np.zeros(n_target_frames, dtype=np.float32))
                continue

            if n_swift_frames == n_target_frames:
                f0_resampled = f0
            else:
                # Use linear interpolation via torch for consistency
                f0_t = torch.from_numpy(f0).float().unsqueeze(0).unsqueeze(0)
                f0_resampled_t = torch.nn.functional.interpolate(
                    f0_t, size=n_target_frames, mode="linear", align_corners=False
                )
                f0_resampled = f0_resampled_t.squeeze().numpy()

                # Zero out frames that were unvoiced in the original
                # (interpolation can smear voiced/unvoiced boundaries)
                if n_target_frames > 0 and n_swift_frames > 0:
                    voiced_t = torch.from_numpy(
                        voiced.astype(np.float32)
                    ).unsqueeze(0).unsqueeze(0)
                    voiced_resampled = torch.nn.functional.interpolate(
                        voiced_t, size=n_target_frames, mode="nearest"
                    ).squeeze().numpy()
                    f0_resampled[voiced_resampled < 0.5] = 0.0

            results.append(f0_resampled.astype(np.float32))

        f0_batch = np.stack(results, axis=0)
        return torch.from_numpy(f0_batch).to(device=device)
