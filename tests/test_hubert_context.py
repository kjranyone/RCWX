"""Test HuBERT feature quality with different context sizes."""

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


def test_context_sizes():
    """Compare HuBERT features with different context sizes."""
    print("=" * 70)
    print("HuBERT Context Size Analysis")
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

    # Ensure HuBERT is in float32
    if hasattr(pipeline.hubert, 'model'):
        pipeline.hubert.model.float()

    # Load test audio (5 seconds)
    audio_path = Path("sample_data/kakita.wav")
    audio_16k = load_audio(audio_path, target_sr=16000)[:80000]  # 5 seconds
    print(f"Audio: {len(audio_16k)/16000:.1f}s @ 16kHz")

    # Batch processing (ground truth)
    print("\n--- Batch Processing (Ground Truth) ---")
    audio_tensor = torch.from_numpy(audio_16k).float().to(device).unsqueeze(0)
    with torch.no_grad():
        batch_feat = pipeline.hubert.extract(audio_tensor, output_layer=12, output_dim=768)
    print(f"Batch features: {batch_feat.shape}")

    # Test different context sizes
    chunk_sec = 0.10  # 100ms main chunk
    chunk_samples = int(16000 * chunk_sec)
    hop_size = 320  # HuBERT hop

    context_sizes = [0.0, 0.10, 0.20, 0.50, 1.0, 2.0]

    for context_sec in context_sizes:
        context_samples = int(16000 * context_sec)

        print(f"\n--- Context: {context_sec}s ({context_samples} samples) ---")

        chunk_feats = []
        main_pos = 0
        chunk_idx = 0

        while main_pos < len(audio_16k):
            # Build chunk with context
            chunk_start = max(0, main_pos - context_samples)
            chunk_end = min(main_pos + chunk_samples, len(audio_16k))
            chunk = audio_16k[chunk_start:chunk_end]

            if len(chunk) < 320:
                break

            # Extract features
            chunk_tensor = torch.from_numpy(chunk).float().to(device).unsqueeze(0)
            with torch.no_grad():
                feat = pipeline.hubert.extract(chunk_tensor, output_layer=12, output_dim=768)

            # Calculate how many frames to keep (main portion only)
            # context_offset = number of samples of context we actually have
            context_offset = main_pos - chunk_start
            context_frames = context_offset // hop_size

            # Trim context frames from the start
            if context_frames > 0 and context_frames < feat.shape[1]:
                feat = feat[:, context_frames:, :]

            # Also trim excess frames from the end to match chunk_sec worth
            main_frames = chunk_samples // hop_size
            if feat.shape[1] > main_frames:
                feat = feat[:, :main_frames, :]

            chunk_feats.append(feat)
            main_pos += chunk_samples
            chunk_idx += 1

        # Concatenate
        if not chunk_feats:
            print("No features extracted")
            continue

        chunk_feat = torch.cat(chunk_feats, dim=1)

        # Compare with batch
        min_frames = min(batch_feat.shape[1], chunk_feat.shape[1])
        batch_aligned = batch_feat[:, :min_frames, :].cpu()
        chunk_aligned = chunk_feat[:, :min_frames, :].cpu()

        # Per-frame correlation
        frame_corrs = []
        for i in range(min_frames):
            b = batch_aligned[0, i, :].numpy()
            c = chunk_aligned[0, i, :].numpy()
            if np.std(b) > 1e-6 and np.std(c) > 1e-6:
                corr = np.corrcoef(b, c)[0, 1]
                if not np.isnan(corr):
                    frame_corrs.append(corr)

        if frame_corrs:
            frame_corrs = np.array(frame_corrs)
            print(f"  Frames: {chunk_feat.shape[1]} (batch: {batch_feat.shape[1]})")
            print(f"  Correlation: mean={frame_corrs.mean():.4f}, min={frame_corrs.min():.4f}, max={frame_corrs.max():.4f}")

            # MSE
            mse = torch.nn.functional.mse_loss(batch_aligned, chunk_aligned).item()
            print(f"  MSE: {mse:.6f}")

            # Cosine similarity
            batch_flat = batch_aligned.reshape(-1)
            chunk_flat = chunk_aligned.reshape(-1)
            cos_sim = torch.nn.functional.cosine_similarity(
                batch_flat.unsqueeze(0), chunk_flat.unsqueeze(0)
            ).item()
            print(f"  Cosine Sim: {cos_sim:.4f}")

    print("\n" + "=" * 70)
    print("Conclusion: Larger context should improve correlation with batch processing")
    print("=" * 70)


if __name__ == "__main__":
    test_context_sizes()
