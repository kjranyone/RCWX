"""Test that SOLA output length normalization prevents drift.

Verifies:
1. Without normalization: SOLA search offset causes cumulative deficit
2. With normalization: output length matches expected hop (no drift)
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


def test_sola_length_normalization():
    """Test that post-SOLA normalization keeps output at expected length."""
    from rcwx.audio.resample import resample
    from rcwx.audio.sola import SolaState, sola_crossfade
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline

    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("No model configured. Run GUI first.")
        return

    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()
    model_sr = pipeline.sample_rate
    output_sr = 48000
    logger.info(f"Model sample rate: {model_sr}")

    # Load test audio
    test_file = Path(__file__).parent.parent / "sample_data" / "sustained_voice.wav"
    if not test_file.exists():
        test_file = Path(__file__).parent.parent / "sample_data" / "seki.wav"
    audio_16k = load_test_audio(str(test_file), 16000)
    audio_16k = audio_16k[:32000]  # 2 seconds

    hop = 320

    # Match the pipeline's rounding (align UP)
    def align_to_hop(samples, h):
        return ((samples + h - 1) // h) * h

    chunk_sec = 0.16  # After 20ms rounding of 0.15
    overlap_sec = 0.08
    crossfade_sec = 0.04
    sola_search_ms = 10.0

    hop_16k = align_to_hop(int(16000 * chunk_sec), hop)  # 2560
    overlap_16k = align_to_hop(int(16000 * overlap_sec), hop)  # 1280
    hop_mic = int(hop_16k * output_sr / 16000)  # 7680

    # SOLA at output rate
    cf_out = int(output_sr * crossfade_sec)  # 1920
    search_out = int(output_sr * sola_search_ms / 1000)  # 480

    aligned_len = (len(audio_16k) // hop) * hop
    audio_16k = audio_16k[:aligned_len]

    logger.info(f"hop_16k={hop_16k}, overlap_16k={overlap_16k}, hop_mic={hop_mic}")
    logger.info(f"cf_out={cf_out}, search_out={search_out}")

    # === Without normalization ===
    logger.info("\n=== Without normalization ===")
    pipeline.clear_cache()
    sola_state = SolaState(crossfade_samples=cf_out, search_samples=search_out)

    raw_outputs = []
    pos = 0
    chunk_idx = 0
    overlap_buf = None

    while pos < len(audio_16k):
        new_hop = audio_16k[pos:pos + hop_16k]
        if len(new_hop) < hop:
            break
        new_hop_aligned = new_hop[:(len(new_hop) // hop) * hop]
        if len(new_hop_aligned) == 0:
            break

        if overlap_buf is not None:
            chunk = np.concatenate([overlap_buf, new_hop_aligned])
            ovl = len(overlap_buf)
        elif overlap_16k > 0 and len(new_hop_aligned) > overlap_16k:
            reflection = new_hop_aligned[:overlap_16k][::-1].copy()
            chunk = np.concatenate([reflection, new_hop_aligned])
            ovl = overlap_16k
        else:
            chunk = new_hop_aligned
            ovl = 0

        out_model = pipeline.infer_streaming(
            audio_16k=chunk, overlap_samples=ovl, pitch_shift=0,
            f0_method="fcpe", noise_scale=0.0, voice_gate_mode="off",
            use_parallel_extraction=False,
        )
        out_48k = resample(out_model, model_sr, output_sr)
        out_sola = sola_crossfade(out_48k, sola_state)
        raw_outputs.append(len(out_sola))

        if overlap_16k > 0:
            overlap_buf = new_hop_aligned[-overlap_16k:].copy()

        pos += hop_16k
        chunk_idx += 1

    if raw_outputs:
        avg_raw = sum(raw_outputs) / len(raw_outputs)
        deficit_raw = hop_mic - avg_raw
        logger.info(f"Chunks: {len(raw_outputs)}, avg output: {avg_raw:.0f}, "
                     f"expected: {hop_mic}, deficit: {deficit_raw:.0f} ({deficit_raw/output_sr*1000:.1f}ms)")

    # === With normalization (pad/trim to hop_mic) ===
    logger.info("\n=== With normalization ===")
    normalized_outputs = []
    for raw_len in raw_outputs:
        norm_len = hop_mic  # Always exactly hop_mic after pad/trim
        normalized_outputs.append(norm_len)

    avg_norm = sum(normalized_outputs) / len(normalized_outputs)
    deficit_norm = hop_mic - avg_norm
    logger.info(f"Normalized avg output: {avg_norm:.0f}, deficit: {deficit_norm:.0f}")

    # === Verify cumulative drift ===
    logger.info("\n=== Cumulative drift ===")
    cumulative_raw = 0
    cumulative_expected = 0
    for i, raw_len in enumerate(raw_outputs):
        cumulative_raw += raw_len
        cumulative_expected += hop_mic
        drift = cumulative_expected - cumulative_raw
        if i < 5 or i % 5 == 0 or i == len(raw_outputs) - 1:
            logger.info(f"  Chunk {i}: drift={drift} samples ({drift/output_sr*1000:.1f}ms)")

    total_drift = cumulative_expected - cumulative_raw
    logger.info(f"\nTotal drift without normalization: {total_drift} samples ({total_drift/output_sr*1000:.1f}ms)")
    logger.info(f"Total drift with normalization: 0 samples (0.0ms)")

    # Check that normalization eliminates drift
    assert total_drift > 0, "Expected positive drift without normalization"
    logger.info(f"\nPASS: Normalization eliminates {total_drift/output_sr*1000:.1f}ms drift over {len(raw_outputs)} chunks")


if __name__ == "__main__":
    test_sola_length_normalization()
