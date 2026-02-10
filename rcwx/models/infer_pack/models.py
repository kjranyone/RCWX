"""RVC Synthesizer models - ported from RVC WebUI."""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from rcwx.models.infer_pack import attentions, commons, modules


class TextEncoder(nn.Module):
    """Text/Feature encoder - matches original RVC architecture.

    IMPORTANT: Uses nn.Linear for phone embedding (not Conv1d).
    Input features should be [B, T, C] format.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # Original RVC uses Linear embedding, not Conv1d
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=10,  # RVC default
        )

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phone: Phone/HuBERT features [B, T, C]
            phone_lengths: Feature lengths [B]
            pitch: Pitch indices [B, T] (optional)

        Returns:
            x, m, logs, x_mask
        """
        # Embed phone features: [B, T, C] -> [B, T, hidden]
        x = self.emb_phone(phone)

        # Add pitch embedding if available
        if pitch is not None and hasattr(self, "emb_pitch"):
            x = x + self.emb_pitch(pitch)

        # Scale and activate (original RVC pattern)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)

        # Transpose for encoder: [B, T, hidden] -> [B, hidden, T]
        x = x.transpose(1, 2)

        # Create mask after transpose (mask is [B, 1, T])
        x_mask = torch.unsqueeze(
            commons.sequence_mask(phone_lengths, x.size(2)), 1
        ).to(x.dtype)

        # Encoder and projection
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    """Posterior encoder (VAE encoder)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = (
            torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1)
            .to(x.dtype)
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self) -> None:
        self.enc.remove_weight_norm()


class SineGen(nn.Module):
    """Sine wave generator for Neural Source Filter.

    Matches original RVC WebUI implementation using sub-frame phase
    accumulation (_f02sine) instead of interpolation-based approach.
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
        flag_for_pulse: bool = False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        # "legacy" keeps prior interpolation-based implementation.
        # "subframe" enables the newer phase-accumulation implementation.
        self.phase_mode = os.getenv("RCWX_SINEGEN_MODE", "legacy").lower()

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(
        self, f0: torch.Tensor, upp: int
    ) -> torch.Tensor:
        """Convert F0 to sine waveforms using sub-frame phase accumulation.

        This matches the original RVC WebUI algorithm exactly.
        Uses explicit float32 for phase accumulation to avoid precision
        loss under float16 autocast.

        Args:
            f0: [B, T, 1] F0 values per frame
            upp: upsample factor (samples per frame)

        Returns:
            sines: [B, T*upp, dim] sine waveforms
        """
        # Sub-sample indices within each frame: [1, 2, ..., upp]
        a = torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
        # Phase increment for each sub-sample: [B, T, upp]
        rad = f0 / self.sampling_rate * a

        # Track inter-frame phase continuity in float32 for precision
        # rad2: fractional phase at the end of each frame (last sub-sample)
        rad2 = torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5
        # Accumulated phase offset across frames
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        # Add accumulated offset: first frame gets 0, subsequent frames get drift
        rad += F.pad(rad_acc, (0, 0, 1, 0), mode="constant")

        # Flatten to waveform rate: [B, T*upp, 1]
        rad = rad.reshape(f0.shape[0], -1, 1)

        # Multiply by harmonic numbers for overtones: [B, T*upp, dim]
        b = torch.arange(
            1, self.dim + 1, dtype=f0.dtype, device=f0.device
        ).reshape(1, 1, -1)
        rad = rad * b

        # Random initial phase per harmonic (fundamental starts at 0)
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad = rad + rand_ini

        sines = torch.sin(2 * np.pi * rad)
        return sines

    def _f02sine_legacy(
        self, f0: torch.Tensor, upp: int
    ) -> torch.Tensor:
        """Legacy interpolation-based sine generation used by prior RCWX path."""
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        f0_buf[:, :, 0] = f0[:, :, 0]
        for idx in range(self.harmonic_num):
            f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

        rad_values = (f0_buf / self.sampling_rate) % 1
        rand_ini = torch.rand(
            f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device, dtype=f0_buf.dtype
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        tmp_over_one = torch.cumsum(rad_values, 1)
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(
            tmp_over_one.transpose(2, 1),
            scale_factor=float(upp),
            mode="linear",
            align_corners=True,
        ).transpose(2, 1)
        rad_values = F.interpolate(
            rad_values.transpose(2, 1),
            scale_factor=float(upp),
            mode="nearest",
        ).transpose(2, 1)
        tmp_over_one %= 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        return torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

    def forward(
        self,
        f0: torch.Tensor,
        upp: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # f0: [B, T] -> [B, T, 1]
            f0 = f0.unsqueeze(-1)

            if self.phase_mode == "subframe":
                sine_waves = self._f02sine(f0, upp)
            else:
                sine_waves = self._f02sine_legacy(f0, upp)
            sine_waves = sine_waves * self.sine_amp

            uv = self._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """Harmonic + Noise source module."""

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
        is_half: bool = True,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
        upp: int = 1,
    ) -> torch.Tensor:
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        # Match dtype to linear layer weights (handles float16/bfloat16/float32)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class GeneratorNSF(nn.Module):
    """Neural Source Filter decoder (F0 model)."""

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        is_half: bool = False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = nn.Upsample(scale_factor=np.prod(upsample_rates))

        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr, harmonic_num=0, is_half=is_half
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_cls = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        c_cur,
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = int(np.prod(upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(commons.init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = np.prod(upsample_rates)

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        n_res: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        if n_res is not None:
            # Match original RVC: interpolate both har_source and x to target length
            n = int(n_res.item()) if isinstance(n_res, torch.Tensor) else int(n_res)
            if n * self.upp != har_source.shape[-1]:
                har_source = F.interpolate(
                    har_source, size=n * self.upp, mode="linear"
                )
            if n != x.shape[-1]:
                x = F.interpolate(x, size=n, mode="linear")

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class Generator(nn.Module):
    """Standard decoder (No-F0 model)."""

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_cls = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(commons.init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        n_res: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class ResidualCouplingBlock(nn.Module):
    """Residual coupling block for normalizing flow."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.flows:
            if hasattr(layer, "remove_weight_norm"):
                layer.remove_weight_norm()


class SynthesizerTrnMs256NSFsid(nn.Module):
    """RVC v1 model with F0 (256-dim features)."""

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.sr = sr

        self.enc_p = TextEncoder(
            256,  # 256-dim HuBERT features
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=True,
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=kwargs.get("is_half", False),
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def remove_weight_norm(self) -> None:
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    @torch.no_grad()
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        skip_head: int = 0,
        return_length: int = 0,
        return_length2: int = 0,
        noise_scale: float = 0.66666,
    ) -> torch.Tensor:
        """
        Args:
            phone: HuBERT features [B, T, C] - NOT transposed
            phone_lengths: Feature lengths [B]
            pitch: Pitch indices [B, T]
            pitchf: Pitch in Hz [B, T]
            sid: Speaker ID [B]
            noise_scale: VAE noise coefficient (0=deterministic, 0.66666=default)
        """
        g = self.emb_g(sid).unsqueeze(-1)
        # TextEncoder expects [B, T, C] format
        x, m_p, logs_p, x_mask = self.enc_p(phone, phone_lengths, pitch)

        # Apply skip_head and return_length for streaming
        # Match original RVC: provide 24 extra context frames to flow model
        if skip_head > 0 or return_length > 0:
            head = skip_head
            length = return_length if return_length > 0 else x.shape[2] - skip_head
            # Flow context: start up to 24 frames earlier for receptive field
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            # Slice encoder outputs with flow context
            m_p = m_p[:, :, flow_head : flow_head + dec_head + length]
            logs_p = logs_p[:, :, flow_head : flow_head + dec_head + length]
            x_mask = x_mask[:, :, flow_head : flow_head + dec_head + length]
            # Sample and run flow with extra context
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            # Slice flow output to target region (discard context)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
            pitchf = pitchf[:, head : head + length]
        else:
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)

        n_res = return_length2 if return_length2 > 0 else None
        o = self.dec(z * x_mask, pitchf, g=g, n_res=n_res)
        return o


class SynthesizerTrnMs768NSFsid(SynthesizerTrnMs256NSFsid):
    """RVC v2 model with F0 (768-dim features)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the encoder with 768-dim input
        self.enc_p = TextEncoder(
            768,  # 768-dim HuBERT features
            self.inter_channels,
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
            f0=True,
        )


class SynthesizerTrnMs256NSFsidNono(nn.Module):
    """RVC v1 model without F0 (256-dim features)."""

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int = None,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim

        self.enc_p = TextEncoder(
            256,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=False,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def remove_weight_norm(self) -> None:
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    @torch.no_grad()
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        sid: torch.Tensor,
        skip_head: int = 0,
        return_length: int = 0,
        return_length2: int = 0,
        noise_scale: float = 0.66666,
    ) -> torch.Tensor:
        """
        Args:
            phone: HuBERT features [B, T, C] - NOT transposed
            phone_lengths: Feature lengths [B]
            sid: Speaker ID [B]
            noise_scale: VAE noise coefficient (0=deterministic, 0.66666=default)
        """
        g = self.emb_g(sid).unsqueeze(-1)
        # TextEncoder expects [B, T, C] format, no pitch for non-F0 models
        x, m_p, logs_p, x_mask = self.enc_p(phone, phone_lengths)

        # Apply skip_head and return_length for streaming
        # Match original RVC: provide 24 extra context frames to flow model
        if skip_head > 0 or return_length > 0:
            head = skip_head
            length = return_length if return_length > 0 else x.shape[2] - skip_head
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m_p = m_p[:, :, flow_head : flow_head + dec_head + length]
            logs_p = logs_p[:, :, flow_head : flow_head + dec_head + length]
            x_mask = x_mask[:, :, flow_head : flow_head + dec_head + length]
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
        else:
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)

        n_res = return_length2 if return_length2 > 0 else None
        o = self.dec(z * x_mask, g=g, n_res=n_res)
        return o


class SynthesizerTrnMs768NSFsidNono(SynthesizerTrnMs256NSFsidNono):
    """RVC v2 model without F0 (768-dim features)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the encoder with 768-dim input
        self.enc_p = TextEncoder(
            768,
            self.inter_channels,
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
            f0=False,
        )
