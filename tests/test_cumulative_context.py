"""Test cumulative context approach for HuBERT processing."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.io import wavfile

from rcwx.audio.resample import resample
from rcwx.device import get_device
from rcwx.pipeline.inference import RVCPipeline


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def test_cumulative_context():
    """
    Cumulative context approach:
    - Chunk 1: process [chunk1], extract features for chunk1
    - Chunk 2: process [chunk1 + chunk2], extract features for chunk2
    - Chunk 3: process [chunk1 + chunk2 + chunk3], extract features for chunk3
    ...

    This ensures each chunk sees all previous audio as context,
    matching batch processing's context availability.
    """
    print("=" * 70)
    print("Cumulative Context HuBERT Test")
    print("=" * 70)

    model_path = Path("model/kurumi/kurumi.pth")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    device = get_device()
    print(f"Device: {device}")

    pipeline = RVCPipeline(
        model_path=str(model_path),
        device=device,
        dtype=torch.float32,
        use_f0=True,
        use_compile=False,
    )
    pipeline.load()

    if hasattr(pipeline.hubert, 'model'):
        pipeline.hubert.model.float()

    # Load test audio (3 seconds for faster testing)
    audio_path = Path("sample_data/kakita.wav")
    audio_16k = load_audio(audio_path, target_sr=16000)[:48000]  # 3 seconds
    print(f"Audio: {len(audio_16k)/16000:.1f}s @ 16kHz")

    # Batch processing (ground truth)
    print("\n--- Batch Processing (Ground Truth) ---")
    audio_tensor = torch.from_numpy(audio_16k).float().to(device).unsqueeze(0)
    with torch.no_grad():
        batch_feat = pipeline.hubert.extract(audio_tensor, output_layer=12, output_dim=768)
    print(f"Batch features: {batch_feat.shape}")

    # Cumulative context processing
    chunk_sec = 0.10  # 100ms chunks
    chunk_samples = int(16000 * chunk_sec)
    hop_size = 320  # HuBERT hop

    print(f"\n--- Cumulative Context Processing (chunk={chunk_sec}s) ---")

    cumulative_feats = []
    pos = 0
    chunk_idx = 0
    prev_feat_len = 0

    while pos < len(audio_16k):
        chunk_end = min(pos + chunk_samples, len(audio_16k))

        # Cumulative audio: from start to current chunk end
        cumulative_audio = audio_16k[:chunk_end]

        # Extract features for entire cumulative audio
        audio_tensor = torch.from_numpy(cumulative_audio).float().to(device).unsqueeze(0)
        with torch.no_grad():
            feat = pipeline.hubert.extract(audio_tensor, output_layer=12, output_dim=768)

        # Extract only the NEW frames (frames not from previous chunks)
        new_frames = feat.shape[1] - prev_feat_len
        if new_frames > 0:
            chunk_feat = feat[:, prev_feat_len:, :]
            cumulative_feats.append(chunk_feat)
            if chunk_idx < 3:
                print(f"  Chunk {chunk_idx}: audio=0:{chunk_end}, feat={feat.shape[1]}, new={new_frames}")

        prev_feat_len = feat.shape[1]
        pos = chunk_end
        chunk_idx += 1

    # Concatenate
    cumulative_feat = torch.cat(cumulative_feats, dim=1)
    print(f"Cumulative features: {cumulative_feat.shape}")

    # Compare with batch
    min_frames = min(batch_feat.shape[1], cumulative_feat.shape[1])
    batch_aligned = batch_feat[:, :min_frames, :].cpu()
    cumul_aligned = cumulative_feat[:, :min_frames, :].cpu()

    # Per-frame correlation
    frame_corrs = []
    high_corr_frames = []
    low_corr_frames = []
    for i in range(min_frames):
        b = batch_aligned[0, i, :].numpy()
        c = cumul_aligned[0, i, :].numpy()
        if np.std(b) > 1e-6 and np.std(c) > 1e-6:
            corr = np.corrcoef(b, c)[0, 1]
            if not np.isnan(corr):
                frame_corrs.append(corr)
                if corr > 0.9:
                    high_corr_frames.append(i)
                elif corr < 0.1:
                    low_corr_frames.append(i)

    frame_corrs = np.array(frame_corrs)

    print(f"\n  High correlation frames (>0.9): {len(high_corr_frames)}/{min_frames}")
    print(f"  Low correlation frames (<0.1): {len(low_corr_frames)}/{min_frames}")
    if high_corr_frames:
        print(f"  High corr frame indices: {high_corr_frames[:10]}{'...' if len(high_corr_frames) > 10 else ''}")
    if low_corr_frames:
        print(f"  Low corr frame indices: {low_corr_frames[:10]}{'...' if len(low_corr_frames) > 10 else ''}")
    print(f"\nCumulative vs Batch Comparison:")
    print(f"  Frame correlation: mean={frame_corrs.mean():.4f}, min={frame_corrs.min():.4f}, max={frame_corrs.max():.4f}")

    # MSE
    mse = torch.nn.functional.mse_loss(batch_aligned, cumul_aligned).item()
    print(f"  MSE: {mse:.6f}")

    # Cosine similarity
    batch_flat = batch_aligned.reshape(-1)
    cumul_flat = cumul_aligned.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        batch_flat.unsqueeze(0), cumul_flat.unsqueeze(0)
    ).item()
    print(f"  Cosine Sim: {cos_sim:.4f}")

    # Check if they're identical (within floating point tolerance)
    max_diff = (batch_aligned - cumul_aligned).abs().max().item()
    print(f"  Max absolute diff: {max_diff:.8f}")

    if max_diff < 1e-5:
        print("\n*** SUCCESS: Cumulative context matches batch processing! ***")
    else:
        print(f"\n  Note: Small differences likely due to HuBERT's sliding window behavior")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_cumulative_context()
