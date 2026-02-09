"""RVC Synthesizer model loader and utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn

from rcwx.models.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsidNono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsidNono,
)

logger = logging.getLogger(__name__)


def detect_model_type(checkpoint: dict) -> Tuple[int, bool, int]:
    """
    Detect the RVC model type from checkpoint.

    Args:
        checkpoint: Model checkpoint dictionary

    Returns:
        Tuple of (version, has_f0, speaker_embed_dim)
        - version: 1 for 256-dim features, 2 for 768-dim features
        - has_f0: True if model uses F0 (pitch)
        - speaker_embed_dim: Speaker embedding dimension
    """
    config = checkpoint.get("config", [])
    weight = checkpoint.get("weight", {})

    # Debug: log all checkpoint keys
    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
    logger.info(f"Config: {config}")

    # First check explicit version key (e.g., "v1", "v2")
    version_str = checkpoint.get("version", "")
    if version_str:
        logger.info(f"Found version key: {version_str}")
        if "2" in str(version_str):
            version = 2
        elif "1" in str(version_str):
            version = 1
        else:
            version = 2  # Default to v2
    # Fallback: detect from weights
    elif "enc_p.pre.weight" in weight:
        shape = weight["enc_p.pre.weight"].shape
        in_channels = shape[1]
        logger.info(f"enc_p.pre.weight shape: {shape}, in_channels={in_channels}")
        version = 2 if in_channels == 768 else 1
    else:
        # Fallback: check config length
        logger.info(f"No enc_p.pre.weight, using config length={len(config)}")
        version = 2 if len(config) > 18 else 1

    # Check explicit f0 key first (1=has F0, 0=no F0)
    f0_flag = checkpoint.get("f0")
    if f0_flag is not None:
        has_f0 = bool(int(f0_flag))
    else:
        # Fallback: detect by checking for decoder source module
        has_f0 = "dec.m_source.l_linear.weight" in weight

    # Get speaker embedding dimension
    if "emb_g.weight" in weight:
        speaker_embed_dim = weight["emb_g.weight"].shape[0]
    else:
        speaker_embed_dim = 1

    logger.info(
        f"Detected model: version={version}, has_f0={has_f0}, "
        f"speaker_embed_dim={speaker_embed_dim}"
    )

    return version, has_f0, speaker_embed_dim


def get_model_class(version: int, has_f0: bool) -> Type[nn.Module]:
    """
    Get the appropriate model class based on version and F0 support.

    Args:
        version: Model version (1 or 2)
        has_f0: Whether model uses F0

    Returns:
        Model class
    """
    if version == 1:
        if has_f0:
            return SynthesizerTrnMs256NSFsid
        else:
            return SynthesizerTrnMs256NSFsidNono
    else:  # version == 2
        if has_f0:
            return SynthesizerTrnMs768NSFsid
        else:
            return SynthesizerTrnMs768NSFsidNono


# Default model configuration
DEFAULT_CONFIG = {
    "spec_channels": 1025,
    "segment_size": 32,
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [10, 10, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "gin_channels": 256,
}


class SynthesizerLoader:
    """
    Loader for RVC synthesizer models.

    Handles loading, detection, and configuration of RVC models.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        use_compile: bool = False,
    ):
        """
        Initialize the synthesizer loader.

        Args:
            model_path: Path to the .pth model file
            device: Device to load model on
            dtype: Data type for model weights
            use_compile: Whether to use torch.compile
        """
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.use_compile = use_compile

        self.model: Optional[nn.Module] = None
        self.version: int = 2
        self.has_f0: bool = True
        self.speaker_id: int = 0
        self.sample_rate: int = 40000

    def load(self) -> nn.Module:
        """
        Load the synthesizer model.

        Returns:
            Loaded model
        """
        logger.info(f"Loading synthesizer from: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=True,
        )

        # Detect model type
        self.version, self.has_f0, speaker_embed_dim = detect_model_type(checkpoint)

        # Get configuration
        config = checkpoint.get("config", [])
        if config:
            # Parse config list to dict
            model_config = self._parse_config(config)
        else:
            model_config = DEFAULT_CONFIG.copy()

        # Update speaker embedding dimension
        model_config["spk_embed_dim"] = speaker_embed_dim

        # Set is_half based on dtype for NSF source module
        model_config["is_half"] = self.dtype == torch.float16

        # Get sample rate from config or default
        sr = checkpoint.get("sr", 40000)
        # Handle string sample rates like "40k", "48k"
        if isinstance(sr, str):
            sr = sr.lower().replace("k", "000")
            sr = int(sr)
        self.sample_rate = sr
        model_config["sr"] = self.sample_rate

        # Get model class
        model_cls = get_model_class(self.version, self.has_f0)

        # Create model
        self.model = model_cls(**model_config)

        # Load weights
        weight = checkpoint.get("weight", checkpoint)
        self.model.load_state_dict(weight, strict=False)

        # Move to device and set dtype
        self.model.to(self.device).to(self.dtype)
        self.model.eval()

        # Remove weight normalization for inference
        self.model.remove_weight_norm()

        # Optionally compile model
        if self.use_compile:
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        logger.info(
            f"Model loaded: v{self.version}, f0={self.has_f0}, sr={self.sample_rate}"
        )

        return self.model

    def _parse_config(self, config: list) -> dict:
        """Parse config list to dictionary."""
        # Config list order (from RVC WebUI):
        # [spec_channels, segment_size, inter_channels, hidden_channels,
        #  filter_channels, n_heads, n_layers, kernel_size, p_dropout,
        #  resblock, resblock_kernel_sizes, resblock_dilation_sizes,
        #  upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
        #  spk_embed_dim, gin_channels, sr]
        keys = [
            "spec_channels",
            "segment_size",
            "inter_channels",
            "hidden_channels",
            "filter_channels",
            "n_heads",
            "n_layers",
            "kernel_size",
            "p_dropout",
            "resblock",
            "resblock_kernel_sizes",
            "resblock_dilation_sizes",
            "upsample_rates",
            "upsample_initial_channel",
            "upsample_kernel_sizes",
            "spk_embed_dim",
            "gin_channels",
            "sr",
        ]

        result = DEFAULT_CONFIG.copy()
        for i, key in enumerate(keys):
            if i < len(config):
                result[key] = config[i]

        return result

    @torch.no_grad()
    def infer(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        speaker_id: Optional[torch.Tensor] = None,
        noise_scale: float = 0.66666,
    ) -> torch.Tensor:
        """
        Run inference on the loaded model.

        Args:
            features: HuBERT features [B, T, C]
            feature_lengths: Feature lengths [B]
            pitch: Pitch indices (for F0 models) [B, T]
            pitchf: Pitch values in Hz (for F0 models) [B, T]
            speaker_id: Speaker ID tensor [B]
            noise_scale: VAE noise coefficient (0=deterministic, 0.66666=default)

        Returns:
            Generated audio [B, T_out]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Model expects features in [B, T, C] format (no transpose needed)
        # Original RVC uses Linear embedding which takes [B, T, C]

        # Default speaker ID
        if speaker_id is None:
            speaker_id = torch.zeros(
                features.shape[0],
                dtype=torch.long,
                device=features.device,
            )

        if self.has_f0:
            if pitch is None or pitchf is None:
                raise ValueError("F0 model requires pitch and pitchf inputs")
            output = self.model.infer(
                features,
                feature_lengths,
                pitch,
                pitchf,
                speaker_id,
                noise_scale=noise_scale,
            )
        else:
            output = self.model.infer(
                features,
                feature_lengths,
                speaker_id,
                noise_scale=noise_scale,
            )

        return output.squeeze(1)


def load_index(index_path: str) -> Optional[object]:
    """
    Load FAISS index for voice similarity search.

    Args:
        index_path: Path to .index file

    Returns:
        FAISS index or None if not available
    """
    try:
        import faiss

        index_path = Path(index_path)
        if not index_path.exists():
            logger.warning(f"Index file not found: {index_path}")
            return None

        logger.info(f"Loading index from: {index_path}")
        index = faiss.read_index(str(index_path))
        return index
    except ImportError:
        logger.warning("faiss-cpu not installed, index loading disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        return None
