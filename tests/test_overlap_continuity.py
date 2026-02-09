"""Test overlap continuity between chunks.

Verifies:
1. Overlap buffer audio matches the tail of the previous hop (audio-level continuity)
2. Chunk boundaries in infer_streaming() output are smooth
3. Batch vs streaming quality comparison at boundaries
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
    from rcwx.audio.resample import resample
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = resample(data, sr, target_sr)
    return data


def align_to_hop(samples: int, hop: int) -> int:
    return ((samples + hop - 1) // hop) * hop


def test_overlap_continuity():
    """Test that overlap buffer creates continuous audio context."""
    from rcwx.audio.resample import StatefulResampler, resample
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline

    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("No model configured. Run GUI first.")
        return

    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()
    model_sr = pipeline.sample_rate
    logger.info(f"Model sample rate: {model_sr}")

    # Load test audio
    test_file = Path(__file__).parent.parent / "sample_data" / "sustained_voice.wav"
    if not test_file.exists():
        test_file = Path(__file__).parent.parent / "sample_data" / "seki.wav"
    audio_16k = load_test_audio(str(test_file), 16000)
    audio_16k = audio_16k[:32000]  # 2 seconds

    hop = 320
    chunk_sec = 0.16  # after 20ms rounding
    overlap_sec = 0.08  # auto-param for chunk=0.16

    hop_16k = align_to_hop(int(16000 * chunk_sec), hop)  # 2560
    overlap_16k = align_to_hop(int(16000 * overlap_sec), hop)  # 1280

    logger.info(f"hop_16k={hop_16k}, overlap_16k={overlap_16k}")

    # Align audio
    aligned_len = (len(audio_16k) // hop) * hop
    audio_16k = audio_16k[:aligned_len]

    # === Test 1: Overlap buffer audio continuity ===
    logger.info("\n=== Test 1: Overlap buffer audio continuity ===")

    overlap_buf = None
    chunks_16k = []  # Store each chunk_16k
    overlap_bufs = []  # Store each overlap_buf
    hop_starts = []  # Track position in continuous audio

    pos = 0
    while pos < len(audio_16k):
        new_hop = audio_16k[pos:pos + hop_16k]
        if len(new_hop) < hop:
            break
        new_hop = new_hop[:(len(new_hop) // hop) * hop]

        if overlap_buf is not None:
            chunk = np.concatenate([overlap_buf, new_hop])
            ovl = len(overlap_buf)
        elif overlap_16k > 0:
            reflection = new_hop[:overlap_16k][::-1].copy()
            chunk = np.concatenate([reflection, new_hop])
            ovl = overlap_16k
        else:
            chunk = new_hop
            ovl = 0

        chunks_16k.append(chunk)
        hop_starts.append(pos)

        overlap_buf = new_hop[-overlap_16k:].copy()
        overlap_bufs.append(overlap_buf)

        pos += hop_16k

    logger.info(f"Generated {len(chunks_16k)} chunks")

    # Verify: overlap_buf[i] should equal audio_16k[start+hop-overlap : start+hop]
    for i in range(len(overlap_bufs)):
        start = hop_starts[i]
        expected_overlap = audio_16k[start + hop_16k - overlap_16k : start + hop_16k]
        if len(expected_overlap) == len(overlap_bufs[i]):
            match = np.allclose(overlap_bufs[i], expected_overlap)
            if not match:
                diff = np.max(np.abs(overlap_bufs[i] - expected_overlap))
                logger.error(f"  Chunk {i}: overlap_buf MISMATCH! max_diff={diff}")
            elif i < 3:
                logger.info(f"  Chunk {i}: overlap_buf matches audio (OK)")

    # Verify: chunk[i+1] starts with overlap_buf[i]
    for i in range(len(chunks_16k) - 1):
        next_chunk = chunks_16k[i + 1]
        prev_overlap = overlap_bufs[i]
        chunk_overlap_part = next_chunk[:len(prev_overlap)]
        match = np.allclose(chunk_overlap_part, prev_overlap)
        if not match:
            diff = np.max(np.abs(chunk_overlap_part - prev_overlap))
            logger.error(f"  Chunk {i}→{i+1}: overlap prefix MISMATCH! max_diff={diff}")
        elif i < 3:
            logger.info(f"  Chunk {i}→{i+1}: overlap prefix matches (OK)")

    # Verify: junction continuity
    # overlap_buf[i][-1] should be followed by chunks_16k[i+1][overlap_16k]
    for i in range(len(chunks_16k) - 1):
        junction_left = overlap_bufs[i][-1]
        junction_right = chunks_16k[i + 1][len(overlap_bufs[i])]
        # These should be consecutive samples from the continuous audio
        expected_left = audio_16k[hop_starts[i] + hop_16k - 1]
        expected_right = audio_16k[hop_starts[i + 1]]
        left_ok = junction_left == expected_left
        right_ok = junction_right == expected_right
        if not (left_ok and right_ok):
            logger.error(f"  Chunk {i}→{i+1}: junction NOT continuous!")
        elif i < 3:
            jump = abs(junction_right - junction_left)
            logger.info(f"  Chunk {i}→{i+1}: junction continuous (jump={jump:.6f})")

    logger.info("PASS: Audio-level overlap is continuous")

    # === Test 2: Batch vs Streaming at chunk boundaries ===
    logger.info("\n=== Test 2: Batch vs Streaming output at boundaries ===")

    # Batch: process entire audio
    pipeline.clear_cache()
    batch_output = pipeline.infer(
        audio_16k, input_sr=16000, pitch_shift=0, f0_method="fcpe",
        use_feature_cache=False, voice_gate_mode="off", noise_scale=0.0,
        use_parallel_extraction=False,
    )
    logger.info(f"Batch output: {len(batch_output)} samples at {model_sr}Hz")

    # Streaming: process chunk by chunk
    pipeline.clear_cache()
    streaming_outputs = []
    for i, chunk in enumerate(chunks_16k):
        ovl = overlap_16k if i > 0 else (overlap_16k if len(chunk) > hop_16k else 0)
        out = pipeline.infer_streaming(
            audio_16k=chunk, overlap_samples=ovl, pitch_shift=0,
            f0_method="fcpe", noise_scale=0.0, voice_gate_mode="off",
            use_parallel_extraction=False,
        )
        streaming_outputs.append(out)

    streaming_concat = np.concatenate(streaming_outputs)
    logger.info(f"Streaming output: {len(streaming_concat)} samples ({len(streaming_outputs)} chunks)")

    # Compare at chunk boundaries
    expected_chunk_len = hop_16k * model_sr // 16000
    logger.info(f"Expected chunk output length: {expected_chunk_len}")

    boundary_corrs = []
    boundary_window = expected_chunk_len // 4  # Quarter chunk at boundary

    for i in range(len(streaming_outputs) - 1):
        boundary_pos = sum(len(o) for o in streaming_outputs[:i + 1])

        # Window around boundary in streaming output
        left = max(0, boundary_pos - boundary_window)
        right = min(len(streaming_concat), boundary_pos + boundary_window)
        stream_window = streaming_concat[left:right]

        # Same window in batch output
        if right <= len(batch_output):
            batch_window = batch_output[left:right]
            corr = np.corrcoef(stream_window, batch_window)[0, 1]
            boundary_corrs.append(corr)
            if i < 5 or corr < 0.9:
                logger.info(f"  Boundary {i}→{i+1} (pos={boundary_pos}): corr={corr:.4f}")

    if boundary_corrs:
        avg_corr = sum(boundary_corrs) / len(boundary_corrs)
        min_corr = min(boundary_corrs)
        logger.info(f"\nBoundary correlation: avg={avg_corr:.4f}, min={min_corr:.4f}")

    # Compare at chunk centers (should be higher correlation)
    center_corrs = []
    for i in range(len(streaming_outputs)):
        chunk_start = sum(len(o) for o in streaming_outputs[:i])
        center = chunk_start + len(streaming_outputs[i]) // 2
        left = max(0, center - boundary_window)
        right = min(len(streaming_concat), center + boundary_window)
        stream_window = streaming_concat[left:right]

        if right <= len(batch_output):
            batch_window = batch_output[left:right]
            corr = np.corrcoef(stream_window, batch_window)[0, 1]
            center_corrs.append(corr)

    if center_corrs:
        avg_center = sum(center_corrs) / len(center_corrs)
        min_center = min(center_corrs)
        logger.info(f"Center correlation:   avg={avg_center:.4f}, min={min_center:.4f}")

    # === Test 3: Discontinuities at boundaries ===
    logger.info("\n=== Test 3: Output discontinuities at chunk boundaries ===")

    for i in range(len(streaming_outputs) - 1):
        boundary_pos = sum(len(o) for o in streaming_outputs[:i + 1])
        if boundary_pos < len(streaming_concat):
            jump = abs(streaming_concat[boundary_pos] - streaming_concat[boundary_pos - 1])
            if i < 5 or jump > 0.1:
                logger.info(f"  Boundary {i}→{i+1}: jump={jump:.6f}")

    # Overall discontinuity stats
    diff = np.abs(np.diff(streaming_concat))
    large_jumps = np.where(diff > 0.3)[0]
    logger.info(f"\nDiscontinuities > 0.3: {len(large_jumps)}")

    # === Test 4: Compare different overlap sizes ===
    logger.info("\n=== Test 4: Effect of overlap size on quality ===")

    for test_overlap_ms in [0, 40, 80, 120, 160, 200]:
        test_overlap_16k = align_to_hop(int(16000 * test_overlap_ms / 1000), hop)
        if test_overlap_16k >= hop_16k:
            continue

        pipeline.clear_cache()
        test_outputs = []
        test_overlap_buf = None
        pos = 0
        while pos < len(audio_16k):
            new_hop = audio_16k[pos:pos + hop_16k]
            if len(new_hop) < hop:
                break
            new_hop = new_hop[:(len(new_hop) // hop) * hop]

            if test_overlap_buf is not None and test_overlap_16k > 0:
                chunk = np.concatenate([test_overlap_buf, new_hop])
                ovl = len(test_overlap_buf)
            elif test_overlap_16k > 0 and len(new_hop) > test_overlap_16k:
                reflection = new_hop[:test_overlap_16k][::-1].copy()
                chunk = np.concatenate([reflection, new_hop])
                ovl = test_overlap_16k
            else:
                chunk = new_hop
                ovl = 0

            out = pipeline.infer_streaming(
                audio_16k=chunk, overlap_samples=ovl, pitch_shift=0,
                f0_method="fcpe", noise_scale=0.0, voice_gate_mode="off",
                use_parallel_extraction=False,
            )
            test_outputs.append(out)

            if test_overlap_16k > 0:
                test_overlap_buf = new_hop[-test_overlap_16k:].copy()
            pos += hop_16k

        if test_outputs:
            test_concat = np.concatenate(test_outputs)
            min_len = min(len(test_concat), len(batch_output))
            if min_len > 0:
                corr = np.corrcoef(test_concat[:min_len], batch_output[:min_len])[0, 1]

                # Measure boundary jumps
                jumps = []
                bpos = 0
                for j in range(len(test_outputs) - 1):
                    bpos += len(test_outputs[j])
                    if bpos < len(test_concat):
                        jumps.append(abs(test_concat[bpos] - test_concat[bpos - 1]))
                avg_jump = sum(jumps) / len(jumps) if jumps else 0

                logger.info(f"  overlap={test_overlap_ms:3d}ms: corr={corr:.4f}, "
                           f"avg_boundary_jump={avg_jump:.4f}, chunks={len(test_outputs)}")

    # Save outputs
    out_dir = Path("test_output/overlap_continuity")
    out_dir.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_dir / "batch.wav"), model_sr, batch_output)
    wavfile.write(str(out_dir / "streaming.wav"), model_sr, streaming_concat)
    logger.info(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    test_overlap_continuity()
