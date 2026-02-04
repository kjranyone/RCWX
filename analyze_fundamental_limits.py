"""Fundamental analysis: Where does batch-chunk divergence occur?

This script quantifies degradation at each processing stage to identify
the fundamental bottleneck preventing us from reaching 0.93 correlation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.audio.resample import resample, StatefulResampler


def load_audio(path: str, max_sec: float = 5.0) -> np.ndarray:
    """Load audio (shorter for faster analysis)."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio[:, 0]
    max_samples = int(sr * max_sec)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    if sr != 48000:
        audio = resample(audio, sr, 48000)
    return audio.astype(np.float32)


def correlation(a, b):
    """Simple correlation coefficient."""
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compare_features(feat1, feat2):
    """Compare two feature tensors."""
    if feat1.shape != feat2.shape:
        min_len = min(feat1.shape[1], feat2.shape[1])
        feat1 = feat1[:, :min_len]
        feat2 = feat2[:, :min_len]

    # Flatten and compute correlation
    f1 = feat1.cpu().numpy().flatten()
    f2 = feat2.cpu().numpy().flatten()
    return correlation(f1, f2)


def analyze_resampling_stage(pipeline, audio_48k):
    """Stage 1: Analyze resampling precision."""
    print("\n" + "="*80)
    print("STAGE 1: Resampling (48kHz → 16kHz)")
    print("="*80)

    # Batch: single resample
    batch_resampler = StatefulResampler(48000, 16000)
    batch_16k = batch_resampler.resample_chunk(audio_48k)

    # Chunk: simulate chunking
    chunk_resampler = StatefulResampler(48000, 16000)
    chunk_size = int(48000 * 0.35)  # 350ms chunks
    chunks_16k = []

    pos = 0
    while pos < len(audio_48k):
        chunk = audio_48k[pos:pos + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunks_16k.append(chunk_resampler.resample_chunk(chunk))
        pos += chunk_size

    chunk_16k = np.concatenate(chunks_16k)

    # Compare
    corr = correlation(batch_16k, chunk_16k)
    print(f"  Batch length:  {len(batch_16k)} samples")
    print(f"  Chunk length:  {len(chunk_16k)} samples")
    print(f"  Correlation:   {corr:.6f}")
    print(f"  → Resampling is {'PERFECT' if corr > 0.9999 else 'IMPERFECT'}")

    return batch_16k, chunk_16k, corr


def analyze_hubert_stage(pipeline, batch_16k, chunk_16k):
    """Stage 2: Analyze HuBERT feature extraction."""
    print("\n" + "="*80)
    print("STAGE 2: HuBERT Feature Extraction")
    print("="*80)

    # Batch HuBERT
    batch_audio_t = torch.from_numpy(batch_16k).float().unsqueeze(0).to(pipeline.device)
    with torch.no_grad():
        batch_features = pipeline.hubert.extract(batch_audio_t, output_layer=pipeline.hubert_output_layer)

    # Chunk HuBERT (simulate with padding)
    chunk_size_16k = int(16000 * 0.35)
    context_16k = int(16000 * 0.119)

    chunk_features_list = []
    pos = 0
    chunk_idx = 0

    while pos < len(chunk_16k):
        # Extract chunk with context
        if chunk_idx == 0:
            # First chunk: no left context
            start = 0
            end = pos + chunk_size_16k
        else:
            # Subsequent chunks: left context
            start = max(0, pos - context_16k)
            end = pos + chunk_size_16k

        chunk_audio = chunk_16k[start:end]

        # Pad with reflection
        t_pad = 320  # 1 HuBERT hop
        chunk_audio_padded = np.pad(chunk_audio, (t_pad, t_pad), mode='reflect')

        chunk_audio_t = torch.from_numpy(chunk_audio_padded).float().unsqueeze(0).to(pipeline.device)
        with torch.no_grad():
            chunk_feat = pipeline.hubert.extract(chunk_audio_t, output_layer=pipeline.hubert_output_layer)

        # Trim context from features (if not first chunk)
        if chunk_idx > 0:
            # Context corresponds to ~context_16k/320 frames (HuBERT hop = 320)
            context_frames = context_16k // 320
            if chunk_feat.shape[1] > context_frames:
                chunk_feat = chunk_feat[:, context_frames:]

        chunk_features_list.append(chunk_feat)
        pos += chunk_size_16k
        chunk_idx += 1

    chunk_features = torch.cat(chunk_features_list, dim=1)

    # Compare
    corr = compare_features(batch_features, chunk_features)
    print(f"  Batch shape:   {batch_features.shape}")
    print(f"  Chunk shape:   {chunk_features.shape}")
    print(f"  Correlation:   {corr:.6f}")
    print(f"  → HuBERT chunking causes {'MINOR' if corr > 0.99 else 'SIGNIFICANT'} degradation")

    return batch_features, chunk_features, corr


def analyze_synthesizer_stage(pipeline, batch_features, chunk_features):
    """Stage 3: Analyze synthesizer output."""
    print("\n" + "="*80)
    print("STAGE 3: Synthesizer (Features → Audio)")
    print("="*80)

    # For fair comparison, use same feature lengths
    min_len = min(batch_features.shape[1], chunk_features.shape[1])
    batch_feat = batch_features[:, :min_len]
    chunk_feat = chunk_features[:, :min_len]

    # Interpolate to 100fps (2x)
    batch_feat_100fps = torch.nn.functional.interpolate(
        batch_feat.permute(0, 2, 1),
        size=min_len * 2,
        mode='linear',
        align_corners=True
    ).permute(0, 2, 1)

    chunk_feat_100fps = torch.nn.functional.interpolate(
        chunk_feat.permute(0, 2, 1),
        size=min_len * 2,
        mode='linear',
        align_corners=True
    ).permute(0, 2, 1)

    # Generate F0 (dummy - same for both)
    f0_len = batch_feat_100fps.shape[1]
    f0 = torch.ones(1, f0_len, device=pipeline.device) * 200.0  # 200Hz dummy F0

    # Synthesize
    with torch.no_grad():
        batch_audio = pipeline.synthesizer.infer(batch_feat_100fps, f0, f0)
        chunk_audio = pipeline.synthesizer.infer(chunk_feat_100fps, f0, f0)

    batch_audio_np = batch_audio[0].cpu().numpy()
    chunk_audio_np = chunk_audio[0].cpu().numpy()

    # Compare
    corr = correlation(batch_audio_np, chunk_audio_np)
    print(f"  Batch length:  {len(batch_audio_np)} samples")
    print(f"  Chunk length:  {len(chunk_audio_np)} samples")
    print(f"  Correlation:   {corr:.6f}")
    print(f"  → Synthesizer inherits feature differences")

    return batch_audio_np, chunk_audio_np, corr


def main():
    print("="*80)
    print("FUNDAMENTAL BOTTLENECK ANALYSIS")
    print("="*80)
    print("\nGoal: Identify where batch-chunk divergence occurs")
    print("Current: 0.9103-0.9127 correlation")
    print("Target:  0.93+ correlation")
    print("Gap:     0.0197-0.0273 improvement needed")

    # Load
    print("\nLoading model and audio...")
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
    audio_48k = load_audio("sample_data/seki.wav", max_sec=5.0)
    print(f"Audio: {len(audio_48k)/48000:.2f}s @ 48kHz")

    # Stage-by-stage analysis
    results = {}

    # Stage 1: Resampling
    batch_16k, chunk_16k, corr1 = analyze_resampling_stage(pipeline, audio_48k)
    results['resampling'] = corr1

    # Stage 2: HuBERT
    batch_feat, chunk_feat, corr2 = analyze_hubert_stage(pipeline, batch_16k, chunk_16k)
    results['hubert'] = corr2

    # Stage 3: Synthesizer
    batch_synth, chunk_synth, corr3 = analyze_synthesizer_stage(pipeline, batch_feat, chunk_feat)
    results['synthesizer'] = corr3

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Correlation at Each Stage")
    print("="*80)
    print(f"  1. Resampling:    {results['resampling']:.6f} {'✓ PERFECT' if results['resampling'] > 0.9999 else '✗ DEGRADED'}")
    print(f"  2. HuBERT:        {results['hubert']:.6f} {'✓ EXCELLENT' if results['hubert'] > 0.99 else '✗ DEGRADED'}")
    print(f"  3. Synthesizer:   {results['synthesizer']:.6f}")
    print("="*80)

    # Analysis
    print("\nANALYSIS:")

    if results['resampling'] < 0.9999:
        print("  ⚠ Resampling introduces phase errors")
        print("    → StatefulResampler may accumulate errors across chunks")

    if results['hubert'] < 0.99:
        print("  ⚠ HuBERT is the PRIMARY bottleneck")
        print("    → Chunk boundaries cause feature discontinuities")
        print("    → Context padding helps but is insufficient")
        print("    → Transformer's global attention cannot be replicated in chunks")
    else:
        print("  ✓ HuBERT chunking is highly effective")

    if results['synthesizer'] < results['hubert']:
        print("  ⚠ Synthesizer amplifies feature differences")
    else:
        print("  ✓ Synthesizer faithfully reproduces features")

    print("\nCONCLUSION:")
    if results['hubert'] < 0.99:
        print("  The fundamental limit is HuBERT's non-causal architecture.")
        print("  Improvements needed:")
        print("    1. Better feature cache blending (learned vs linear)")
        print("    2. Larger context windows (>119ms)")
        print("    3. Causal HuBERT variant (long-term)")
    else:
        print("  HuBERT chunking is not the bottleneck.")
        print("  Look at other stages (F0, SOLA, length adjustment).")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
