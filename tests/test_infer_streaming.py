"""Test infer_streaming() vs infer() output comparison.

Verifies that the new streaming inference produces correct output
by comparing against batch inference (gold standard).

Key insight: The VITS synthesizer uses VAE sampling (torch.randn_like * 0.66666),
making each call non-deterministic. Use noise_scale=0 for deterministic comparison.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_test_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target_sr."""
    from rcwx.audio.resample import resample
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = resample(data, sr, target_sr)
    return data


def test_batch_vs_streaming():
    """Compare batch infer() with infer_streaming() on same audio."""
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline

    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("No model configured. Run GUI first.")
        return

    model_path = config.last_model_path
    logger.info(f"Model: {model_path}")

    # Load pipeline
    pipeline = RVCPipeline(model_path, device=config.device, use_compile=False)
    pipeline.load()
    model_sr = pipeline.sample_rate
    logger.info(f"Model sample rate: {model_sr}")

    # Load test audio
    test_file = Path(__file__).parent.parent / "sample_data" / "sustained_voice.wav"
    if not test_file.exists():
        test_file = Path(__file__).parent.parent / "sample_data" / "seki.wav"
    audio_16k = load_test_audio(str(test_file), 16000)
    logger.info(f"Test audio: {test_file.name}, {len(audio_16k)} samples ({len(audio_16k)/16000:.2f}s)")

    # Use 1 second of audio
    audio_16k = audio_16k[:16000]
    hop = 320
    aligned_len = (len(audio_16k) // hop) * hop
    audio_aligned = audio_16k[:aligned_len]

    # === Test 1: Deterministic comparison (noise_scale=0) ===
    logger.info("\n=== Test 1: Deterministic batch vs streaming (noise_scale=0) ===")
    pipeline.clear_cache()
    batch_output = pipeline.infer(
        audio_16k,
        input_sr=16000,
        pitch_shift=0,
        f0_method="fcpe",
        use_feature_cache=False,
        voice_gate_mode="off",
        noise_scale=0.0,
        use_parallel_extraction=False,
    )
    pipeline.clear_cache()
    streaming_output = pipeline.infer_streaming(
        audio_16k=audio_aligned,
        overlap_samples=0,
        pitch_shift=0,
        f0_method="fcpe",
        voice_gate_mode="off",
        noise_scale=0.0,
        use_parallel_extraction=False,
    )
    logger.info(f"Batch output: {len(batch_output)} samples")
    logger.info(f"Streaming output: {len(streaming_output)} samples")

    min_len = min(len(batch_output), len(streaming_output))
    if min_len > 0:
        corr = np.corrcoef(batch_output[:min_len], streaming_output[:min_len])[0, 1]
        rmse = np.sqrt(np.mean((batch_output[:min_len] - streaming_output[:min_len])**2))
        logger.info(f"Deterministic: corr={corr:.6f}, rmse={rmse:.6f}")
        assert corr > 0.95, f"Deterministic correlation too low: {corr:.6f}"
        logger.info("PASS: Deterministic comparison")

    # === Test 2: With voice gate (deterministic) ===
    logger.info("\n=== Test 2: Deterministic with voice gate ===")
    pipeline.clear_cache()
    batch_gate = pipeline.infer(
        audio_16k, input_sr=16000, pitch_shift=0, f0_method="fcpe",
        use_feature_cache=False, voice_gate_mode="expand", noise_scale=0.0,
        use_parallel_extraction=False,
    )
    pipeline.clear_cache()
    stream_gate = pipeline.infer_streaming(
        audio_aligned, overlap_samples=0, pitch_shift=0, f0_method="fcpe",
        voice_gate_mode="expand", noise_scale=0.0, use_parallel_extraction=False,
    )
    min_len = min(len(batch_gate), len(stream_gate))
    corr_gate = np.corrcoef(batch_gate[:min_len], stream_gate[:min_len])[0, 1]
    logger.info(f"With voice gate: corr={corr_gate:.6f}")
    assert corr_gate > 0.95, f"Voice gate correlation too low: {corr_gate:.6f}"
    logger.info("PASS: Voice gate comparison")

    # === Test 3: Production mode (normal noise) ===
    logger.info("\n=== Test 3: Production mode (noise_scale=0.667) ===")
    pipeline.clear_cache()
    batch_noisy = pipeline.infer(
        audio_16k, input_sr=16000, pitch_shift=0, f0_method="fcpe",
        use_feature_cache=False, voice_gate_mode="expand",
        use_parallel_extraction=False,
    )
    pipeline.clear_cache()
    stream_noisy = pipeline.infer_streaming(
        audio_aligned, overlap_samples=0, pitch_shift=0, f0_method="fcpe",
        voice_gate_mode="expand", use_parallel_extraction=False,
    )
    min_len = min(len(batch_noisy), len(stream_noisy))
    corr_noisy = np.corrcoef(batch_noisy[:min_len], stream_noisy[:min_len])[0, 1]
    logger.info(f"Production mode: corr={corr_noisy:.6f} (expected ~0.8 due to VAE noise)")
    # With normal VAE noise, correlation should be > 0.5 (perceptually similar)
    assert corr_noisy > 0.5, f"Production correlation too low: {corr_noisy:.6f}"
    logger.info("PASS: Production mode")

    # === Test 4: f0_method=none (overload protection path) ===
    logger.info("\n=== Test 4: f0_method=none (overload protection) ===")
    pipeline.clear_cache()
    stream_nof0 = pipeline.infer_streaming(
        audio_aligned, overlap_samples=0, pitch_shift=0, f0_method="none",
        voice_gate_mode="off", noise_scale=0.0, use_parallel_extraction=False,
    )
    rms_nof0 = np.sqrt(np.mean(stream_nof0**2))
    logger.info(f"f0=none output: {len(stream_nof0)} samples, rms={rms_nof0:.6f}")
    assert len(stream_nof0) > 0, "f0=none produced empty output"
    logger.info("PASS: f0_method=none")

    # === Test 5: Chunked streaming with overlap ===
    logger.info("\n=== Test 5: Chunked streaming with overlap (deterministic) ===")
    pipeline.clear_cache()

    chunk_sec = 0.15
    overlap_sec = 0.10
    chunk_samples = (int(16000 * chunk_sec) // hop) * hop
    overlap_samples = (int(16000 * overlap_sec) // hop) * hop

    logger.info(f"Chunk: {chunk_samples} samples ({chunk_samples/16000*1000:.0f}ms)")
    logger.info(f"Overlap: {overlap_samples} samples ({overlap_samples/16000*1000:.0f}ms)")

    chunks_output = []
    pos = 0
    chunk_idx = 0

    while pos < len(audio_aligned):
        new_hop = audio_aligned[pos : pos + chunk_samples]
        if len(new_hop) < hop:
            break
        aligned_new = (len(new_hop) // hop) * hop
        new_hop = new_hop[:aligned_new]
        if len(new_hop) == 0:
            break

        if chunk_idx == 0:
            if overlap_samples > 0 and len(new_hop) > overlap_samples:
                reflection = new_hop[:overlap_samples][::-1].copy()
                chunk_16k = np.concatenate([reflection, new_hop])
                ovl = overlap_samples
            else:
                chunk_16k = new_hop
                ovl = 0
        else:
            ovl_start = max(0, pos - overlap_samples)
            overlap_audio = audio_aligned[ovl_start:pos]
            aligned_ovl = (len(overlap_audio) // hop) * hop
            if aligned_ovl > 0:
                overlap_audio = overlap_audio[-aligned_ovl:]
                chunk_16k = np.concatenate([overlap_audio, new_hop])
                ovl = len(overlap_audio)
            else:
                chunk_16k = new_hop
                ovl = 0

        out = pipeline.infer_streaming(
            audio_16k=chunk_16k, overlap_samples=ovl, pitch_shift=0,
            f0_method="fcpe", noise_scale=0.0, voice_gate_mode="off",
            use_parallel_extraction=False,
        )
        chunks_output.append(out)
        logger.info(f"  Chunk {chunk_idx}: in={len(chunk_16k)}, overlap={ovl}, out={len(out)} samples")

        pos += chunk_samples
        chunk_idx += 1

    if chunks_output:
        streaming_chunked = np.concatenate(chunks_output)
        logger.info(f"\nChunked output: {len(streaming_chunked)} samples ({chunk_idx} chunks)")

        # Check for discontinuities at chunk boundaries
        diff = np.abs(np.diff(streaming_chunked))
        large_jumps = np.where(diff > 0.3)[0]
        logger.info(f"Discontinuities (>0.3): {len(large_jumps)}")
        if len(large_jumps) > 0:
            logger.warning(f"  at sample positions: {large_jumps[:10]}")

        # Check for silence / zero output
        rms = np.sqrt(np.mean(streaming_chunked**2))
        logger.info(f"Output RMS: {rms:.6f}")
        assert rms > 0.001, "ERROR: Output is near-silent!"
        logger.info("PASS: Chunked streaming output is non-silent")

    # === Test 6: Output length consistency ===
    logger.info("\n=== Test 6: Output length per chunk ===")
    pipeline.clear_cache()

    expected_output_per_hop = chunk_samples * model_sr // 16000
    logger.info(f"Expected output per hop: {expected_output_per_hop} samples")

    length_errors = []
    pos = 0
    for i in range(5):
        new_hop = audio_aligned[pos : pos + chunk_samples]
        if len(new_hop) < chunk_samples:
            break

        if i == 0:
            reflection = new_hop[:overlap_samples][::-1].copy()
            chunk = np.concatenate([reflection, new_hop])
            ovl = overlap_samples
        else:
            ovl_start = max(0, pos - overlap_samples)
            overlap_audio = audio_aligned[ovl_start:pos]
            aligned_ovl = (len(overlap_audio) // hop) * hop
            overlap_audio = overlap_audio[-aligned_ovl:]
            chunk = np.concatenate([overlap_audio, new_hop])
            ovl = len(overlap_audio)

        out = pipeline.infer_streaming(
            audio_16k=chunk, overlap_samples=ovl, pitch_shift=0,
            f0_method="fcpe", noise_scale=0.0, voice_gate_mode="off",
            use_parallel_extraction=False,
        )
        diff = len(out) - expected_output_per_hop
        length_errors.append(diff)
        logger.info(f"  Chunk {i}: output={len(out)}, expected={expected_output_per_hop}, diff={diff}")
        pos += chunk_samples

    if length_errors:
        max_err = max(abs(e) for e in length_errors)
        logger.info(f"Max length error: {max_err} samples ({max_err/model_sr*1000:.1f}ms)")

    # === Save outputs ===
    out_dir = Path("test_output/streaming_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_dir / "batch.wav"), model_sr, batch_output)
    wavfile.write(str(out_dir / "streaming_no_overlap.wav"), model_sr, streaming_output)
    if chunks_output:
        wavfile.write(str(out_dir / "streaming_chunked.wav"), model_sr, streaming_chunked)
    logger.info(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    test_batch_vs_streaming()
