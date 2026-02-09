"""Diagnose chunk boundary seams using actual debug audio.

Simulates the exact realtime_unified.py pipeline flow:
  48kHz input → StatefulResampler 48k→16k → overlap assembly →
  infer_streaming() → StatefulResampler model→48k → SOLA → output
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


def align_to_hop(samples: int, hop: int) -> int:
    return ((samples + hop - 1) // hop) * hop


def run(input_wav: str):
    from rcwx.audio.resample import StatefulResampler, resample
    from rcwx.audio.sola import SolaState, sola_crossfade
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline

    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("No model configured.")
        return

    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()
    model_sr = pipeline.sample_rate

    # Load input
    sr_in, audio_in = wavfile.read(input_wav)
    if audio_in.dtype == np.int16:
        audio_in = audio_in.astype(np.float32) / 32768.0
    if audio_in.ndim > 1:
        audio_in = audio_in.mean(axis=1)
    logger.info(f"Input: {input_wav}, sr={sr_in}, {len(audio_in)} samples ({len(audio_in)/sr_in:.2f}s)")

    # --- Pipeline params (match auto-params for chunk=160ms) ---
    mic_sr = sr_in
    output_sr = 48000
    hubert_hop = 320

    chunk_sec = 0.16
    overlap_sec = 0.08
    crossfade_sec = 0.04
    sola_search_ms = 10.0

    hop_16k = align_to_hop(int(16000 * chunk_sec), hubert_hop)  # 2560
    overlap_16k = align_to_hop(int(16000 * overlap_sec), hubert_hop)  # 1280
    hop_mic = int(hop_16k * mic_sr / 16000)

    # SOLA extra: same calculation as realtime_unified.py
    crossfade_samples_out = int(output_sr * crossfade_sec)  # 1920
    search_samples_out = int(output_sr * sola_search_ms / 1000)  # 480
    sola_extra_out = crossfade_samples_out + search_samples_out  # 2400
    sola_extra_model = int(sola_extra_out * model_sr / output_sr)  # 2000

    logger.info(f"hop_16k={hop_16k}, overlap_16k={overlap_16k}, hop_mic={hop_mic}")
    logger.info(f"chunk_sec={chunk_sec}, overlap_sec={overlap_sec}")
    logger.info(f"sola_extra_model={sola_extra_model} (crossfade={crossfade_samples_out}, search={search_samples_out})")

    out_dir = Path("test_output/seam_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # A) Batch inference (gold standard)
    # ============================
    logger.info("\n=== A) Batch inference ===")
    audio_16k_full = resample(audio_in, sr_in, 16000)
    pipeline.clear_cache()
    batch_output = pipeline.infer(
        audio_16k_full, input_sr=16000, pitch_shift=0, f0_method="fcpe",
        use_feature_cache=False, voice_gate_mode="off",
        use_parallel_extraction=False,
    )
    batch_48k = resample(batch_output, model_sr, output_sr)
    logger.info(f"Batch: {len(batch_output)} @ {model_sr}Hz → {len(batch_48k)} @ {output_sr}Hz")
    wavfile.write(str(out_dir / "batch_48k.wav"), output_sr, batch_48k)

    # ============================
    # B) Streaming (simulate realtime_unified.py exactly)
    # ============================
    logger.info("\n=== B) Streaming pipeline (simulating realtime_unified.py) ===")
    pipeline.clear_cache()

    input_resampler = StatefulResampler(mic_sr, 16000)
    output_resampler = StatefulResampler(model_sr, output_sr)
    sola_state = SolaState(
        crossfade_samples=int(output_sr * crossfade_sec),
        search_samples=int(output_sr * sola_search_ms / 1000),
    )

    overlap_buf = None
    input_buf = audio_in.copy()

    chunks_raw = []         # raw infer output (model_sr)
    chunks_resampled = []   # after output resampler (48kHz)
    chunks_after_sola = []  # after SOLA
    chunks_final = []       # after length normalization

    hop_samples_mic = hop_mic
    chunk_idx = 0
    pos = 0

    while pos + hop_samples_mic <= len(input_buf):
        hop_audio = input_buf[pos:pos + hop_samples_mic]
        pos += hop_samples_mic

        # Stage 2: Resample 48k → 16k
        hop_16k_data = input_resampler.resample_chunk(hop_audio)

        # Stage 4: Align to HuBERT boundary
        aligned = align_to_hop(len(hop_16k_data), hubert_hop)
        if len(hop_16k_data) < aligned:
            hop_16k_data = np.pad(hop_16k_data, (0, aligned - len(hop_16k_data)))
        elif len(hop_16k_data) > aligned:
            hop_16k_data = hop_16k_data[:aligned]

        # Assemble [overlap | new_hop]
        if overlap_buf is not None:
            chunk_16k = np.concatenate([overlap_buf, hop_16k_data])
            ovl_samples = len(overlap_buf)
        elif overlap_16k > 0 and len(hop_16k_data) > overlap_16k:
            reflection = hop_16k_data[:overlap_16k][::-1].copy()
            chunk_16k = np.concatenate([reflection, hop_16k_data])
            ovl_samples = overlap_16k
        else:
            chunk_16k = hop_16k_data
            ovl_samples = 0

        # Store overlap for next
        if overlap_16k > 0:
            overlap_buf = hop_16k_data[-overlap_16k:].copy()

        # Stage 5: Inference (with sola_extra like realtime_unified.py)
        output_model = pipeline.infer_streaming(
            audio_16k=chunk_16k,
            overlap_samples=ovl_samples,
            pitch_shift=0,
            f0_method="fcpe",
            voice_gate_mode="off",
            use_parallel_extraction=False,
            sola_extra_samples=sola_extra_model,
        )
        chunks_raw.append(output_model.copy())

        # Stage 6: Resample model_sr → 48kHz
        output_48k = output_resampler.resample_chunk(output_model)
        chunks_resampled.append(output_48k.copy())

        # Stage 7: SOLA
        output_sola = sola_crossfade(output_48k, sola_state)
        chunks_after_sola.append(output_sola.copy())

        # No length normalization — RingOutputBuffer absorbs surplus
        # (matches realtime_unified.py behavior)
        chunks_final.append(output_sola.copy())

        if chunk_idx < 5:
            logger.info(
                f"  Chunk {chunk_idx}: in_16k={len(chunk_16k)}, ovl={ovl_samples}, "
                f"raw={len(output_model)}, resamp={len(output_48k)}, "
                f"sola={len(output_sola)}"
            )
        chunk_idx += 1

    logger.info(f"Total chunks: {chunk_idx}")

    # ============================
    # C) Analysis
    # ============================
    logger.info("\n=== C) Analysis ===")

    # Save raw concatenated (no SOLA, no normalization)
    raw_concat = np.concatenate([resample(c, model_sr, output_sr) for c in chunks_raw])
    wavfile.write(str(out_dir / "streaming_raw_concat.wav"), output_sr, raw_concat)

    # Save resampled concatenated (no SOLA)
    resamp_concat = np.concatenate(chunks_resampled)
    wavfile.write(str(out_dir / "streaming_no_sola.wav"), output_sr, resamp_concat)

    # Save final (with SOLA + normalization)
    final_concat = np.concatenate(chunks_final)
    wavfile.write(str(out_dir / "streaming_final.wav"), output_sr, final_concat)

    # --- Boundary analysis on RAW output (before SOLA) ---
    logger.info("\n--- Raw output boundaries (before SOLA) ---")
    boundary_pos = 0
    for i in range(min(len(chunks_resampled) - 1, 10)):
        boundary_pos += len(chunks_resampled[i])
        if boundary_pos < len(resamp_concat):
            left = resamp_concat[boundary_pos - 1]
            right = resamp_concat[boundary_pos]
            jump = abs(right - left)

            # Local RMS around boundary (±50 samples)
            w = 50
            local = resamp_concat[max(0, boundary_pos - w):boundary_pos + w]
            local_rms = np.sqrt(np.mean(local**2)) if len(local) > 0 else 0

            logger.info(
                f"  Boundary {i}→{i+1}: jump={jump:.4f}, "
                f"left={left:.4f}, right={right:.4f}, local_rms={local_rms:.4f}"
            )

    # --- Boundary analysis on FINAL output (after SOLA + normalization) ---
    logger.info("\n--- Final output boundaries (after SOLA) ---")
    for i in range(min(len(chunks_final) - 1, 10)):
        bp = (i + 1) * hop_samples_mic
        if bp < len(final_concat):
            left = final_concat[bp - 1]
            right = final_concat[bp]
            jump = abs(right - left)

            w = 50
            local = final_concat[max(0, bp - w):bp + w]
            local_rms = np.sqrt(np.mean(local**2)) if len(local) > 0 else 0

            logger.info(
                f"  Boundary {i}→{i+1}: jump={jump:.4f}, "
                f"left={left:.4f}, right={right:.4f}, local_rms={local_rms:.4f}"
            )

    # --- Correlation with batch at boundaries ---
    logger.info("\n--- Batch vs Streaming correlation ---")
    min_len = min(len(final_concat), len(batch_48k))
    if min_len > 0:
        overall_corr = np.corrcoef(final_concat[:min_len], batch_48k[:min_len])[0, 1]
        logger.info(f"Overall correlation: {overall_corr:.4f}")

    # Per-chunk correlation with batch
    chunk_start = 0
    for i in range(min(len(chunks_final), 10)):
        chunk_end = chunk_start + len(chunks_final[i])
        if chunk_end <= len(batch_48k):
            c = np.corrcoef(chunks_final[i], batch_48k[chunk_start:chunk_end])[0, 1]
            logger.info(f"  Chunk {i} corr with batch: {c:.4f}")
        chunk_start = chunk_end

    # --- Save individual chunks for inspection ---
    for i in range(min(5, len(chunks_raw))):
        wavfile.write(str(out_dir / f"chunk_{i:02d}_raw.wav"), model_sr, chunks_raw[i])
        wavfile.write(str(out_dir / f"chunk_{i:02d}_final.wav"), output_sr, chunks_final[i])

    # --- Save 3-chunk boundary regions for close listening ---
    for i in range(min(len(chunks_final) - 1, 5)):
        # 2 chunks around boundary
        pair = np.concatenate([chunks_final[i], chunks_final[i + 1]])
        wavfile.write(str(out_dir / f"boundary_{i:02d}_{i+1:02d}.wav"), output_sr, pair)

    logger.info(f"\nAll outputs saved to {out_dir}/")
    logger.info("Key files to compare:")
    logger.info("  batch_48k.wav           - gold standard (batch)")
    logger.info("  streaming_no_sola.wav   - streaming without SOLA")
    logger.info("  streaming_final.wav     - streaming with SOLA + normalization")
    logger.info("  boundary_XX_YY.wav      - zoom into boundary regions")


if __name__ == "__main__":
    import sys
    wav = sys.argv[1] if len(sys.argv) > 1 else "debug_audio/01_input_raw.wav"
    run(wav)
