"""Synthesizer signal probe — analytical diagnosis of hoarse voice.

Hooks into every stage of the RVC synthesizer to measure signal statistics.
Requires a model to run.

Probed stages:
  1. TextEncoder output: m_p, logs_p (VAE prior mean/log-std)
  2. VAE sampling: z_p (latent with noise)
  3. Flow reverse: z (decoded latent)
  4. SineGen: harmonic source waveform
  5. SourceModule: merged harmonic source (after linear+tanh)
  6. GeneratorNSF upsampling stages: x at each layer
  7. Pre-tanh signal (conv_post output)
  8. Final output (post-tanh)
"""

from __future__ import annotations

import logging
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(r"C:\lib\github\grand2-products\RCWX\model\kurumi\kurumi.pth")


# ---------------------------------------------------------------------------
# Probe infrastructure
# ---------------------------------------------------------------------------

class SynthProbe:
    """Captures intermediate tensors via forward hooks."""

    def __init__(self):
        self.captures: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hooks = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # SineGen returns (sine_waves, uv, noise)
                # SourceModuleHnNSF returns (sine_merge, None, None)
                t = output[0]
            else:
                t = output
            if isinstance(t, torch.Tensor):
                self.captures[name] = t.detach().float().cpu()
        return hook

    def attach(self, synthesizer):
        """Attach hooks to all key stages."""
        # SynthesizerLoader wraps the actual model in .model
        model = getattr(synthesizer, "model", synthesizer)

        # 1. TextEncoder (enc_p)
        self._hooks.append(
            model.enc_p.register_forward_hook(self._make_enc_p_hook())
        )

        # 2. Flow
        self._hooks.append(
            model.flow.register_forward_hook(self._make_hook("flow_output"))
        )

        # 3. SineGen (inside dec.m_source.l_sin_gen)
        self._hooks.append(
            model.dec.m_source.l_sin_gen.register_forward_hook(
                self._make_hook("sinegen_output")
            )
        )

        # 4. SourceModuleHnNSF (dec.m_source)
        self._hooks.append(
            model.dec.m_source.register_forward_hook(
                self._make_hook("source_module_output")
            )
        )

        # 5. conv_pre
        self._hooks.append(
            model.dec.conv_pre.register_forward_hook(
                self._make_hook("dec_conv_pre")
            )
        )

        # 6. Each upsampling stage + noise_conv
        for i, (up, nc) in enumerate(
            zip(model.dec.ups, model.dec.noise_convs)
        ):
            self._hooks.append(
                up.register_forward_hook(self._make_hook(f"dec_ups_{i}"))
            )
            self._hooks.append(
                nc.register_forward_hook(self._make_hook(f"dec_noise_conv_{i}"))
            )

        # 7. conv_post (pre-tanh)
        self._hooks.append(
            model.dec.conv_post.register_forward_hook(
                self._make_hook("dec_conv_post_pre_tanh")
            )
        )

        # 8. GeneratorNSF output (post-tanh)
        self._hooks.append(
            model.dec.register_forward_hook(self._make_hook("dec_output"))
        )

    def _make_enc_p_hook(self):
        """Special hook for TextEncoder that returns (x, m_p, logs_p, x_mask)."""
        def hook(module, input, output):
            x, m_p, logs_p, x_mask = output
            self.captures["enc_p_x"] = x.detach().float().cpu()
            self.captures["enc_p_m_p"] = m_p.detach().float().cpu()
            self.captures["enc_p_logs_p"] = logs_p.detach().float().cpu()
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def report(self) -> str:
        """Generate human-readable report of all captured signals."""
        lines = []
        lines.append("=" * 80)
        lines.append("SYNTHESIZER SIGNAL PROBE REPORT")
        lines.append("=" * 80)

        for name, tensor in self.captures.items():
            t = tensor
            lines.append(f"\n--- {name} ---")
            lines.append(f"  shape: {list(t.shape)}")
            lines.append(f"  dtype: {t.dtype}")

            # Flatten for stats
            flat = t.reshape(-1)
            lines.append(f"  mean:  {flat.mean().item():+.6f}")
            lines.append(f"  std:   {flat.std().item():.6f}")
            lines.append(f"  min:   {flat.min().item():+.6f}")
            lines.append(f"  max:   {flat.max().item():+.6f}")
            lines.append(f"  rms:   {torch.sqrt((flat ** 2).mean()).item():.6f}")

            # Saturation analysis (for audio-range signals)
            abs_flat = flat.abs()
            lines.append(f"  |x|>0.9:  {(abs_flat > 0.9).float().mean().item() * 100:.2f}%")
            lines.append(f"  |x|>0.95: {(abs_flat > 0.95).float().mean().item() * 100:.2f}%")
            lines.append(f"  |x|>0.99: {(abs_flat > 0.99).float().mean().item() * 100:.2f}%")

            # For logs_p specifically: show exp(logs_p) = actual std dev
            if "logs_p" in name:
                exp_logs = torch.exp(t.reshape(-1))
                lines.append(f"  -- exp(logs_p) = actual std dev --")
                lines.append(f"  exp mean: {exp_logs.mean().item():.6f}")
                lines.append(f"  exp std:  {exp_logs.std().item():.6f}")
                lines.append(f"  exp min:  {exp_logs.min().item():.6f}")
                lines.append(f"  exp max:  {exp_logs.max().item():.6f}")

            # For pre-tanh: show what tanh compression does
            if "pre_tanh" in name:
                post_tanh = torch.tanh(flat)
                compression = (flat.abs() - post_tanh.abs())
                lines.append(f"  -- tanh compression analysis --")
                lines.append(f"  mean compression: {compression.mean().item():.6f}")
                lines.append(f"  max compression:  {compression.max().item():.6f}")
                lines.append(f"  tanh output rms:  {torch.sqrt((post_tanh ** 2).mean()).item():.6f}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test: probe with different noise_scale values
# ---------------------------------------------------------------------------

def test_synth_probe_sweep():
    """Run synthesizer with multiple noise_scale values and compare signals."""
    if not MODEL_PATH.exists():
        print(f"SKIP: model not found at {MODEL_PATH}")
        return

    from rcwx.pipeline.inference import RVCPipeline

    pipeline = RVCPipeline(str(MODEL_PATH), device="xpu", dtype=torch.float16, use_compile=False)
    pipeline.load()

    synth = pipeline.synthesizer

    # Generate test input: 1 second of 200Hz sine at 16kHz
    sr = 16000
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    audio = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)

    noise_scales = [0.0, 0.1, 0.2, 0.4, 0.66666]

    for ns in noise_scales:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# noise_scale = {ns}")
        logger.info(f"{'#' * 60}")

        probe = SynthProbe()
        probe.attach(synth)

        try:
            output = pipeline.infer(
                audio,
                input_sr=sr,
                pitch_shift=0,
                f0_method="rmvpe",
                noise_scale=ns,
                f0_lowpass_cutoff_hz=16.0,
            )
            print(probe.report())

            # Summary line
            pre_tanh = probe.captures.get("dec_conv_post_pre_tanh")
            dec_out = probe.captures.get("dec_output")
            if pre_tanh is not None and dec_out is not None:
                pt_flat = pre_tanh.reshape(-1)
                do_flat = dec_out.reshape(-1)
                logger.info(
                    f"SUMMARY ns={ns}: "
                    f"pre_tanh rms={torch.sqrt((pt_flat**2).mean()).item():.4f} "
                    f"max={pt_flat.abs().max().item():.4f} "
                    f"|>0.95|={((pt_flat.abs() > 0.95).float().mean().item() * 100):.1f}% | "
                    f"output rms={torch.sqrt((do_flat**2).mean()).item():.4f} "
                    f"|>0.95|={((do_flat.abs() > 0.95).float().mean().item() * 100):.1f}%"
                )
        finally:
            probe.remove()


def test_synth_probe_streaming():
    """Probe streaming inference to check if chunked processing differs."""
    if not MODEL_PATH.exists():
        print(f"SKIP: model not found at {MODEL_PATH}")
        return

    from rcwx.pipeline.inference import RVCPipeline

    pipeline = RVCPipeline(str(MODEL_PATH), device="xpu", dtype=torch.float16, use_compile=False)
    pipeline.load()

    synth = pipeline.synthesizer

    # Simulate streaming chunk: 160ms + 100ms overlap = 260ms
    sr = 16000
    chunk_samples = 320 * 13  # 260ms = 13 HuBERT frames
    overlap_samples = 320 * 5  # 100ms overlap
    t = np.arange(chunk_samples) / sr
    audio = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)

    probe = SynthProbe()
    probe.attach(synth)

    try:
        output = pipeline.infer_streaming(
            audio,
            overlap_samples=overlap_samples,
            noise_scale=0.4,
            f0_lowpass_cutoff_hz=16.0,
        )
        logger.info("\n--- Streaming mode probe ---")
        print(probe.report())
    finally:
        probe.remove()


def test_pre_tanh_histogram():
    """Analyze pre-tanh signal distribution to quantify saturation."""
    if not MODEL_PATH.exists():
        print(f"SKIP: model not found at {MODEL_PATH}")
        return

    from rcwx.pipeline.inference import RVCPipeline

    pipeline = RVCPipeline(str(MODEL_PATH), device="xpu", dtype=torch.float16, use_compile=False)
    pipeline.load()

    synth = pipeline.synthesizer

    sr = 16000
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    audio = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)

    probe = SynthProbe()
    probe.attach(synth)

    try:
        output = pipeline.infer(
            audio, input_sr=sr, noise_scale=0.4, f0_lowpass_cutoff_hz=16.0,
        )

        pre_tanh = probe.captures.get("dec_conv_post_pre_tanh")
        if pre_tanh is None:
            logger.warning("No pre_tanh capture")
            return

        flat = pre_tanh.reshape(-1)
        abs_flat = flat.abs()

        # Histogram bins
        bins = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0]
        logger.info("\n--- Pre-tanh |x| histogram ---")
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            count = ((abs_flat >= lo) & (abs_flat < hi)).sum().item()
            pct = count / len(abs_flat) * 100
            bar = "#" * int(pct / 2)
            logger.info(f"  [{lo:5.1f}, {hi:5.1f}): {pct:6.2f}% {bar}")

        # Tanh distortion estimate
        post_tanh = torch.tanh(flat)
        thd = torch.sqrt(((flat - post_tanh) ** 2).mean()) / torch.sqrt((flat ** 2).mean())
        logger.info(f"\n  Tanh distortion (NMSE): {thd.item():.4f}")
        logger.info(f"  Signal >1.0: {(abs_flat > 1.0).float().mean().item() * 100:.2f}%")
        logger.info(f"  Signal >2.0: {(abs_flat > 2.0).float().mean().item() * 100:.2f}%")

    finally:
        probe.remove()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_synth_probe_sweep,
        test_synth_probe_streaming,
        test_pre_tanh_histogram,
    ]
    for t in tests:
        name = t.__name__
        print(f"\n{'=' * 60}")
        print(f"Running {name}...")
        print(f"{'=' * 60}")
        try:
            t()
            print(f"  DONE")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
