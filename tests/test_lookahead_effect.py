"""Test the effect of lookahead on HuBERT feature quality."""

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
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def test_lookahead():
    """
    Test how lookahead (future context) affects HuBERT feature quality.

    Approach: For each chunk at position t, process audio [0:t+lookahead]
    and extract features for the chunk at t.
    """
    print("=" * 70)
    print("Lookahead Effect on HuBERT Features")
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

    # Load test audio (3 seconds)
    audio_path = Path("sample_data/kakita.wav")
    audio_16k = load_audio(audio_path, target_sr=16000)[:48000]  # 3 seconds
    print(f"Audio: {len(audio_16k)/16000:.1f}s @ 16kHz")

    # Batch processing (ground truth)
    audio_tensor = torch.from_numpy(audio_16k).float().to(device).unsqueeze(0)
    with torch.no_grad():
        batch_feat = pipeline.hubert.extract(audio_tensor, output_layer=12, output_dim=768)
    print(f"Batch features: {batch_feat.shape}")

    # Test different lookahead values
    chunk_sec = 0.10
    chunk_samples = int(16000 * chunk_sec)
    hop_size = 320

    lookahead_values = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]

    print("\n" + "-" * 70)
    print(f"{'Lookahead':>10} | {'Corr Mean':>10} | {'Corr Min':>10} | {'High(>0.9)':>10} | {'Latency':>10}")
    print("-" * 70)

    for lookahead_sec in lookahead_values:
        lookahead_samples = int(16000 * lookahead_sec)

        feats = []
        pos = 0
        prev_feat_len = 0

        while pos < len(audio_16k):
            chunk_end = min(pos + chunk_samples, len(audio_16k))

            # Include lookahead (future context)
            audio_end = min(chunk_end + lookahead_samples, len(audio_16k))
            audio_with_lookahead = audio_16k[:audio_end]

            # Extract features
            audio_tensor = torch.from_numpy(audio_with_lookahead).float().to(device).unsqueeze(0)
            with torch.no_grad():
                feat = pipeline.hubert.extract(audio_tensor, output_layer=12, output_dim=768)

            # Extract only frames for current chunk (not lookahead)
            # frame_end = frames up to chunk_end
            frame_end = (chunk_end - 1) // hop_size + 1
            new_frames = frame_end - prev_feat_len

            if new_frames > 0 and prev_feat_len < feat.shape[1]:
                chunk_feat = feat[:, prev_feat_len:min(frame_end, feat.shape[1]), :]
                feats.append(chunk_feat)

            prev_feat_len = frame_end
            pos = chunk_end

        if not feats:
            print(f"{lookahead_sec:>10.1f}s | No features")
            continue

        cumul_feat = torch.cat(feats, dim=1)

        # Compare with batch
        min_frames = min(batch_feat.shape[1], cumul_feat.shape[1])
        batch_aligned = batch_feat[:, :min_frames, :].cpu()
        cumul_aligned = cumul_feat[:, :min_frames, :].cpu()

        # Per-frame correlation
        frame_corrs = []
        high_corr = 0
        for i in range(min_frames):
            b = batch_aligned[0, i, :].numpy()
            c = cumul_aligned[0, i, :].numpy()
            if np.std(b) > 1e-6 and np.std(c) > 1e-6:
                corr = np.corrcoef(b, c)[0, 1]
                if not np.isnan(corr):
                    frame_corrs.append(corr)
                    if corr > 0.9:
                        high_corr += 1

        if frame_corrs:
            frame_corrs = np.array(frame_corrs)
            latency_ms = (chunk_sec + lookahead_sec) * 1000
            print(f"{lookahead_sec:>10.1f}s | {frame_corrs.mean():>10.4f} | {frame_corrs.min():>10.4f} | {high_corr:>10}/{min_frames} | {latency_ms:>8.0f}ms")

    print("-" * 70)
    print("\nConclusion:")
    print("  - Lookahead provides future context to HuBERT")
    print("  - Larger lookahead = higher correlation with batch")
    print("  - Trade-off: increased latency")


if __name__ == "__main__":
    test_lookahead()
