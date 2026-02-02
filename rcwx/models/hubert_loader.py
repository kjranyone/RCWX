"""HuBERT model loader that can load RVC's hubert_base.pt into transformers model."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig

logger = logging.getLogger(__name__)


def _setup_fake_fairseq():
    """Setup fake fairseq modules for loading checkpoints."""
    if 'fairseq' in sys.modules:
        return

    fairseq_module = types.ModuleType('fairseq')
    fairseq_data = types.ModuleType('fairseq.data')
    fairseq_data_dict = types.ModuleType('fairseq.data.dictionary')

    class Dictionary:
        def __init__(self, *args, **kwargs):
            self.symbols = []
            self.count = []
            self.indices = {}
        def __getstate__(self):
            return self.__dict__
        def __setstate__(self, state):
            self.__dict__.update(state)

    fairseq_data_dict.Dictionary = Dictionary
    fairseq_data.dictionary = fairseq_data_dict
    fairseq_module.data = fairseq_data

    sys.modules['fairseq'] = fairseq_module
    sys.modules['fairseq.data'] = fairseq_data
    sys.modules['fairseq.data.dictionary'] = fairseq_data_dict


def _map_fairseq_to_transformers(fairseq_state_dict: dict) -> dict:
    """Map fairseq HuBERT state dict keys to transformers format."""
    new_state_dict = {}

    for key, value in fairseq_state_dict.items():
        new_key = key

        # Skip non-model weights
        if key in ['mask_emb', 'label_embs_concat']:
            new_key = 'masked_spec_embed' if key == 'mask_emb' else None
            if new_key is None:
                continue

        # Feature extractor conv layers
        elif key.startswith('feature_extractor.conv_layers'):
            parts = key.split('.')
            layer_idx = parts[2]
            sub_idx = parts[3]
            rest = '.'.join(parts[4:])

            if sub_idx == '0':  # Conv weight
                new_key = f'feature_extractor.conv_layers.{layer_idx}.conv.{rest}'
            elif sub_idx == '2':  # GroupNorm/LayerNorm
                new_key = f'feature_extractor.conv_layers.{layer_idx}.layer_norm.{rest}'
            else:
                continue

        # Feature projection
        elif key.startswith('post_extract_proj.'):
            rest = key[len('post_extract_proj.'):]
            new_key = f'feature_projection.projection.{rest}'

        # Layer norm after feature extraction (before encoder)
        elif key == 'layer_norm.weight':
            new_key = 'feature_projection.layer_norm.weight'
        elif key == 'layer_norm.bias':
            new_key = 'feature_projection.layer_norm.bias'

        # Positional conv embedding - map weight_norm parameters directly
        # Transformers uses parametrization with dim=2 (same as fairseq)
        # weight_g shape: [1, 1, kernel_size] -> original0
        # weight_v shape: [out_channels, in_channels//groups, kernel_size] -> original1
        elif key.startswith('encoder.pos_conv.0.'):
            rest = key[len('encoder.pos_conv.0.'):]
            if rest == 'bias':
                new_key = 'encoder.pos_conv_embed.conv.bias'
            elif rest == 'weight_g':
                new_key = 'encoder.pos_conv_embed.conv.parametrizations.weight.original0'
            elif rest == 'weight_v':
                new_key = 'encoder.pos_conv_embed.conv.parametrizations.weight.original1'

        # Encoder layer norm
        elif key == 'encoder.layer_norm.weight':
            new_key = 'encoder.layer_norm.weight'
        elif key == 'encoder.layer_norm.bias':
            new_key = 'encoder.layer_norm.bias'

        # Transformer encoder layers
        elif key.startswith('encoder.layers.'):
            # encoder.layers.X.self_attn.Y -> encoder.layers.X.attention.Y
            # encoder.layers.X.self_attn_layer_norm.Y -> encoder.layers.X.layer_norm.Y
            # encoder.layers.X.fc1.Y -> encoder.layers.X.feed_forward.intermediate_dense.Y
            # encoder.layers.X.fc2.Y -> encoder.layers.X.feed_forward.output_dense.Y
            # encoder.layers.X.final_layer_norm.Y -> encoder.layers.X.final_layer_norm.Y
            new_key = key.replace('.self_attn.', '.attention.')
            new_key = new_key.replace('.self_attn_layer_norm.', '.layer_norm.')
            new_key = new_key.replace('.fc1.', '.feed_forward.intermediate_dense.')
            new_key = new_key.replace('.fc2.', '.feed_forward.output_dense.')

        # Final projection (for ContentVec)
        elif key.startswith('final_proj.'):
            # Keep as is - we'll handle this separately
            new_key = key

        new_state_dict[new_key] = value

    return new_state_dict


class HuBERTLoader:
    """
    Loader for HuBERT models that handles both transformers and RVC formats.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.model: Optional[HubertModel] = None
        self.final_proj: Optional[nn.Linear] = None

        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]) -> None:
        """Load HuBERT model."""
        model_path = Path(model_path) if model_path else None

        # Check if it's RVC's hubert_base.pt format
        if model_path and model_path.exists() and model_path.suffix == '.pt':
            self._load_rvc_hubert(model_path)
        else:
            self._load_transformers_hubert(model_path)

    def _load_transformers_hubert(self, model_path: Optional[Path]) -> None:
        """Load HuBERT from transformers."""
        logger.info("Loading HuBERT from transformers (lengyue233/content-vec-best)")
        self.model = HubertModel.from_pretrained("lengyue233/content-vec-best")
        self.model.to(self.device).to(self.dtype)
        self.model.eval()

        # Initialize final_proj (768 -> 256) for v1 models
        self.final_proj = nn.Linear(768, 256)
        nn.init.xavier_uniform_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        self.final_proj.to(self.device).to(self.dtype)
        self.final_proj.eval()  # Set to evaluation mode

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.final_proj.parameters():
            param.requires_grad = False

    def _load_rvc_hubert(self, model_path: Path) -> None:
        """Load HuBERT from RVC's hubert_base.pt format."""
        logger.info(f"Loading HuBERT from RVC format: {model_path}")

        # Setup fake fairseq for unpickling
        _setup_fake_fairseq()

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        fairseq_state_dict = checkpoint['model']

        # Create transformers model
        config = HubertConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
        self.model = HubertModel(config)

        # Map weights
        transformers_state_dict = _map_fairseq_to_transformers(fairseq_state_dict)

        # Handle final_proj separately
        final_proj_weight = fairseq_state_dict.get('final_proj.weight')
        final_proj_bias = fairseq_state_dict.get('final_proj.bias')

        # Remove final_proj from state dict (not part of HubertModel)
        transformers_state_dict.pop('final_proj.weight', None)
        transformers_state_dict.pop('final_proj.bias', None)

        # Load weights
        missing, unexpected = self.model.load_state_dict(transformers_state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

        self.model.to(self.device).to(self.dtype)
        self.model.eval()

        # Create final_proj for v1 models
        self.final_proj = nn.Linear(768, 256)
        if final_proj_weight is not None:
            self.final_proj.weight.data.copy_(final_proj_weight)
            if final_proj_bias is not None:
                self.final_proj.bias.data.copy_(final_proj_bias)
            logger.info("Loaded final_proj from checkpoint")
        else:
            nn.init.xavier_uniform_(self.final_proj.weight)
            nn.init.zeros_(self.final_proj.bias)

        self.final_proj.to(self.device).to(self.dtype)
        self.final_proj.eval()  # Set to evaluation mode

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.final_proj.parameters():
            param.requires_grad = False

        logger.info("HuBERT loaded from RVC format successfully")

    @torch.no_grad()
    def extract(
        self,
        audio: torch.Tensor,
        output_layer: int = 12,
        output_dim: int = 768,
    ) -> torch.Tensor:
        """
        Extract features from audio.

        Args:
            audio: Audio tensor [B, T] at 16kHz
            output_layer: Which layer to extract from (1-indexed)
            output_dim: Output dimension (768 for v2, 256 for v1)

        Returns:
            Features tensor [B, T', C]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device).float()

        outputs = self.model(audio, output_hidden_states=True)
        features = outputs.hidden_states[output_layer]

        if output_dim == 256:
            features = self.final_proj(features)

        return features

    def forward(self, audio: torch.Tensor, output_dim: int = 768) -> torch.Tensor:
        return self.extract(audio, output_dim=output_dim)
