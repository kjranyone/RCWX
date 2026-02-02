"""Analyze HuBERT feature differences between batch and streaming processing."""

import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.device import get_device
from rcwx.models.hubert_loader import HuBERTLoader
from rcwx.pipeline.inference import RVCPipeline


def analyze_hubert_features():
    """Compare HuBERT features between batch and chunked processing."""
    print("=" * 60)
    print("HuBERT Feature Analysis: Batch vs Streaming")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load HuBERT
    models_dir = Path.home() / ".cache" / "rcwx" / "models"
    hubert = HuBERTLoader(models_dir, device)

    # Generate test audio (1 second sine wave at 440Hz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3

    # Add some variation to make it more speech-like
    audio += np.sin(2 * np.pi * 880 * t) * 0.1
    audio += np.sin(2 * np.pi * 220 * t) * 0.15

    print(f"\nTest audio: {len(audio)} samples ({duration}s at {sample_rate}Hz)")

    # Batch processing
    print("\n--- Batch Processing ---")
    audio_tensor = torch.from_numpy(audio).float().to(device)
    batch_feat = hubert.extract(audio_tensor)
    print(f"Batch features shape: {batch_feat.shape}")
    print(f"  Mean: {batch_feat.mean().item():.6f}")
    print(f"  Std:  {batch_feat.std().item():.6f}")
    print(f"  Min:  {batch_feat.min().item():.6f}")
    print(f"  Max:  {batch_feat.max().item():.6f}")

    # Chunked processing (simulating streaming)
    print("\n--- Chunked Processing ---")
    chunk_sec = 0.10  # 100ms chunks
    context_sec = 0.10  # 100ms context
    chunk_samples = int(sample_rate * chunk_sec)
    context_samples = int(sample_rate * context_sec)

    all_chunk_feats = []
    main_pos = 0
    chunk_idx = 0

    while main_pos < len(audio):
        # Build chunk with context
        if chunk_idx == 0:
            # First chunk: no left context
            chunk_start = 0
            chunk_end = min(chunk_samples, len(audio))
            chunk = audio[chunk_start:chunk_end]
            context_offset = 0
        else:
            # Subsequent chunks: with left context
            chunk_start = max(0, main_pos - context_samples)
            chunk_end = min(main_pos + chunk_samples, len(audio))
            chunk = audio[chunk_start:chunk_end]
            context_offset = main_pos - chunk_start

        if len(chunk) < 160:  # minimum for HuBERT
            break

        # Extract features
        chunk_tensor = torch.from_numpy(chunk).float().to(device)
        feat = hubert.extract(chunk_tensor)

        # Calculate expected frames for main part
        main_samples = min(chunk_samples, len(audio) - main_pos)
        if chunk_idx > 0:
            # Trim context frames
            context_frames = int(context_offset / 320)  # HuBERT hop = 320
            if context_frames < feat.shape[1]:
                feat = feat[:, context_frames:, :]

        all_chunk_feats.append(feat)
        main_pos += chunk_samples
        chunk_idx += 1

    # Concatenate all chunk features
    if all_chunk_feats:
        chunk_feat = torch.cat(all_chunk_feats, dim=1)
        print(f"Chunk features shape: {chunk_feat.shape}")
        print(f"  Mean: {chunk_feat.mean().item():.6f}")
        print(f"  Std:  {chunk_feat.std().item():.6f}")
        print(f"  Min:  {chunk_feat.min().item():.6f}")
        print(f"  Max:  {chunk_feat.max().item():.6f}")

        # Compare
        print("\n--- Comparison ---")
        min_frames = min(batch_feat.shape[1], chunk_feat.shape[1])
        batch_aligned = batch_feat[:, :min_frames, :]
        chunk_aligned = chunk_feat[:, :min_frames, :]

        # Per-frame correlation
        correlations = []
        for i in range(min_frames):
            b = batch_aligned[0, i, :].cpu().numpy()
            c = chunk_aligned[0, i, :].cpu().numpy()
            corr = np.corrcoef(b, c)[0, 1]
            correlations.append(corr)

        correlations = np.array(correlations)
        print(f"Per-frame correlation:")
        print(f"  Mean: {np.mean(correlations):.4f}")
        print(f"  Min:  {np.min(correlations):.4f}")
        print(f"  Max:  {np.max(correlations):.4f}")

        # MSE
        mse = torch.nn.functional.mse_loss(batch_aligned, chunk_aligned).item()
        print(f"MSE: {mse:.6f}")

        # Cosine similarity
        batch_flat = batch_aligned.reshape(-1).cpu()
        chunk_flat = chunk_aligned.reshape(-1).cpu()
        cos_sim = torch.nn.functional.cosine_similarity(
            batch_flat.unsqueeze(0), chunk_flat.unsqueeze(0)
        ).item()
        print(f"Cosine Similarity: {cos_sim:.4f}")

        # Frame-by-frame analysis at chunk boundaries
        print("\n--- Boundary Analysis ---")
        hop_frames = int(chunk_samples / 320)  # frames per chunk
        for boundary_idx in range(1, min(4, chunk_idx)):
            frame = boundary_idx * hop_frames
            if frame < min_frames:
                b = batch_aligned[0, frame, :].cpu().numpy()
                c = chunk_aligned[0, frame, :].cpu().numpy()
                corr = np.corrcoef(b, c)[0, 1]
                mse_frame = np.mean((b - c) ** 2)
                print(f"  Boundary {boundary_idx} (frame {frame}): corr={corr:.4f}, mse={mse_frame:.6f}")


def analyze_energy_normalization():
    """Analyze output energy differences."""
    print("\n" + "=" * 60)
    print("Output Energy Analysis")
    print("=" * 60)

    # Load test results if available
    test_output_dir = Path(__file__).parent.parent / "test_output"
    batch_wav = test_output_dir / "batch_output.wav"
    stream_wav = test_output_dir / "streaming_output.wav"

    if not batch_wav.exists() or not stream_wav.exists():
        print("Test outputs not found. Run test_chunking_modes_comparison.py first.")
        return

    import scipy.io.wavfile as wav

    sr1, batch_audio = wav.read(batch_wav)
    sr2, stream_audio = wav.read(stream_wav)

    # Normalize to float
    if batch_audio.dtype == np.int16:
        batch_audio = batch_audio.astype(np.float32) / 32768.0
    if stream_audio.dtype == np.int16:
        stream_audio = stream_audio.astype(np.float32) / 32768.0

    # Align lengths
    min_len = min(len(batch_audio), len(stream_audio))
    batch_audio = batch_audio[:min_len]
    stream_audio = stream_audio[:min_len]

    # RMS energy
    batch_rms = np.sqrt(np.mean(batch_audio ** 2))
    stream_rms = np.sqrt(np.mean(stream_audio ** 2))

    print(f"Batch RMS:  {batch_rms:.4f}")
    print(f"Stream RMS: {stream_rms:.4f}")
    print(f"Ratio (stream/batch): {stream_rms/batch_rms:.4f}")

    # Per-segment analysis
    segment_sec = 0.5
    segment_samples = int(sr1 * segment_sec)
    n_segments = min_len // segment_samples

    print(f"\nPer-segment energy ratio ({segment_sec}s segments):")
    ratios = []
    for i in range(n_segments):
        start = i * segment_samples
        end = start + segment_samples
        b_rms = np.sqrt(np.mean(batch_audio[start:end] ** 2))
        s_rms = np.sqrt(np.mean(stream_audio[start:end] ** 2))
        if b_rms > 0.001:  # avoid division by zero
            ratio = s_rms / b_rms
            ratios.append(ratio)
            print(f"  Segment {i}: {ratio:.4f}")

    if ratios:
        print(f"\nOverall: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")


if __name__ == "__main__":
    analyze_hubert_features()
    analyze_energy_normalization()
