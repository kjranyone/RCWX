"""FCPE F0 extraction model wrapper.

FCPE (Fast Context-based Pitch Estimation) is a lightweight F0 estimator
suitable for real-time applications with lower latency than RMVPE.

Reference: https://github.com/CNChTu/FCPE
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Check if torchfcpe is available
_FCPE_AVAILABLE = False
try:
    from torchfcpe import spawn_bundled_infer_model
    _FCPE_AVAILABLE = True
except ImportError:
    pass


def is_fcpe_available() -> bool:
    """Check if FCPE is available."""
    return _FCPE_AVAILABLE


class FCPE:
    """FCPE F0 extraction model.

    A fast, lightweight alternative to RMVPE for real-time voice conversion.
    Requires ~5x less computation and works with shorter audio chunks.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        hop_length: int = 160,
    ):
        """Initialize FCPE model.

        Args:
            device: Device to run on (cpu, cuda, xpu)
            dtype: Data type for inference
            hop_length: Hop size in samples (default 160 for 16kHz = 10ms)
        """
        if not _FCPE_AVAILABLE:
            raise ImportError(
                "torchfcpe is not installed. Install with: pip install torchfcpe"
            )

        self.device = device
        self.dtype = dtype
        self.hop_length = hop_length
        self.sample_rate = 16000

        # Load bundled model
        logger.info(f"Loading FCPE model on {device}")
        self.model = spawn_bundled_infer_model(device=device)
        self.model.eval()  # Set to evaluation mode for deterministic inference
        logger.info("FCPE model loaded")

    @torch.no_grad()
    def infer(
        self,
        audio: torch.Tensor,
        threshold: float = 0.006,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
    ) -> torch.Tensor:
        """Extract F0 from audio.

        Args:
            audio: Audio tensor [B, T] or [T] at 16kHz
            threshold: Voicing threshold (default 0.006, lower = more sensitive)
            f0_min: Minimum F0 frequency in Hz
            f0_max: Maximum F0 frequency in Hz

        Returns:
            F0 tensor [B, T_frames] where unvoiced frames are 0
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # FCPE expects [B, T, 1] input
        audio = audio.to(self.device).float()
        if audio.dim() == 2:
            audio = audio.unsqueeze(-1)  # [B, T] -> [B, T, 1]

        # Stabilize input range for FCPE mel extractor
        # Large input gain can drive mel features out of expected range and cause errors.
        with torch.no_grad():
            peak = torch.max(torch.abs(audio))
            if peak > 1.0:
                audio = audio * (0.97 / peak)
            else:
                audio = torch.clamp(audio, -1.0, 1.0)

        # Run inference
        # decoder_mode options: 'local_argmax', 'global_argmax'
        try:
            f0 = self.model.infer(
                audio,
                sr=self.sample_rate,
                decoder_mode='local_argmax',
                threshold=threshold,
                f0_min=f0_min,
                f0_max=f0_max,
            )
        except Exception as e:
            logger.warning(f"FCPE inference failed: {e}")
            return torch.zeros(audio.shape[0], audio.shape[1] // self.hop_length, device=audio.device, dtype=self.dtype)

        # f0 shape: [B, T_frames, 1] -> [B, T_frames]
        if f0.dim() == 3:
            f0 = f0.squeeze(-1)

        # Safety: Replace NaN/Inf with 0 (unvoiced)
        # FCPE can sometimes output NaN values for certain audio conditions
        if torch.any(torch.isnan(f0)) or torch.any(torch.isinf(f0)):
            logger.warning(f"FCPE output contains NaN/Inf values, replacing with 0 (unvoiced)")
            f0 = torch.where(torch.isnan(f0) | torch.isinf(f0), torch.zeros_like(f0), f0)

        return f0.to(self.dtype)

    def to(self, device: str) -> "FCPE":
        """Move model to device."""
        self.device = device
        # torchfcpe model handles device internally
        return self
