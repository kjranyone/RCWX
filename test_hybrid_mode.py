"""Test hybrid mode specifically."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))

from rcwx.audio.resample import resample
from rcwx.device import get_device, get_dtype
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Test hybrid mode."""
    # Configuration
    model_path = "sample_data/hogaraka/hogarakav2.pth"

    # Load and prepare audio
    test_audio_path = "sample_data/seki.wav"
    sr, audio_48k = wavfile.read(test_audio_path)

    # Convert to mono and float32
    if audio_48k.dtype == np.int16:
        audio_48k = audio_48k.astype(np.float32) / 32768.0
    elif audio_48k.dtype == np.int32:
        audio_48k = audio_48k.astype(np.float32) / 2147483648.0

    if len(audio_48k.shape) > 1:
        audio_48k = audio_48k[:, 0]

    # Resample to 48kHz if needed
    if sr != 48000:
        audio_48k = resample(audio_48k, sr, 48000)

    logger.info(f"Test audio: {len(audio_48k)} samples @ 48kHz ({len(audio_48k) / 48000:.2f}s)")

    # Use only first 5 seconds for faster testing
    test_duration_samples = int(5 * 48000)
    if len(audio_48k) > test_duration_samples:
        audio_48k = audio_48k[:test_duration_samples]
        logger.info(f"Using first {test_duration_samples / 48000:.2f}s for testing")

    # Test with hybrid mode
    chunk_sec = 0.35

    configs_to_test = [
        ("wokada", "w-okada mode"),
        ("rvc_webui", "RVC WebUI mode"),
        ("hybrid", "Hybrid mode"),
    ]

    for chunking_mode, mode_name in configs_to_test:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing: {mode_name}")
        logger.info(f"{'=' * 60}")

        # Load fresh pipeline for each mode to avoid cache contamination
        device_str = get_device("xpu")
        logger.info(f"Loading model: model_path")

        test_pipeline = RVCPipeline(
            model_path=model_path,
            device=device_str,
            dtype="float16",
            use_compile=False,
        )
        test_pipeline.load()

        # Clear feature cache
        test_pipeline.clear_cache()
        logger.info("Feature cache cleared for this mode")

        # Create config for this mode
        config = RealtimeConfig(
            chunking_mode=chunking_mode,
            chunk_sec=chunk_sec,
            mic_sample_rate=48000,
            input_sample_rate=16000,
            output_sample_rate=48000,
            f0_method="fcpe",
            use_f0=True,
            use_feature_cache=True,
            use_sola=True,
            context_sec=0.05,
            crossfade_sec=0.05,
            prebuffer_chunks=0,  # Disable prebuffering for testing
            max_queue_size=1000,  # Large queue for testing full audio
        )

        # Create voice changer
        changer = RealtimeVoiceChanger(
            pipeline=test_pipeline,
            config=config,
        )

        # Process audio - feed all input first to ensure full chunks
        block_size = int(48000 * chunk_sec / 4)
        outputs = []

        # Feed all input chunks first
        for i in range(0, len(audio_48k), block_size):
            block = audio_48k[i : i + block_size]
            changer.process_input_chunk(block)

        # Process all queued chunks
        while changer.process_next_chunk():
            pass  # process_next_chunk queues output internally

        # Flush final SOLA buffer if needed
        changer.flush_final_sola_buffer()

        # Drain all output from queue
        try:
            while True:
                audio = changer._output_queue.get_nowait()
                outputs.append(audio)
        except Exception:
            pass

        # Combine outputs
        streaming_output = np.concatenate(outputs)

        # Get batch reference (use same config as streaming)
        logger.info(f"\nComputing batch reference...")
        batch_output = test_pipeline.infer(
            audio_48k,
            input_sr=16000,  # Same as streaming (resamples to 16kHz first)
            pitch_shift=0,
            f0_method="fcpe",
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=False,
        )

        # Align lengths
        min_len = min(len(batch_output), len(streaming_output))
        batch_output = batch_output[:min_len]
        streaming_output = streaming_output[:min_len]

        # Compute metrics
        correlation = np.corrcoef(batch_output, streaming_output)[0, 1]
        mae = np.mean(np.abs(batch_output - streaming_output))
        rmse = np.sqrt(np.mean((batch_output - streaming_output) ** 2))
        energy_ratio = np.sum(streaming_output**2) / np.sum(batch_output**2)

        logger.info(f"\nResults for {mode_name}:")
        logger.info(f"  Batch length:      {len(batch_output)} samples")
        logger.info(f"  Streaming length:  {len(streaming_output)} samples")
        logger.info(f"  Correlation:       {correlation:.6f}")
        logger.info(f"  MAE:               {mae:.6f}")
        logger.info(f"  RMSE:              {rmse:.6f}")
        logger.info(f"  Energy ratio:      {energy_ratio:.6f}")

        # Check discontinuities
        threshold = 0.2
        batch_diff = np.abs(np.diff(batch_output))
        stream_diff = np.abs(np.diff(streaming_output))

        batch_disc = int(np.sum(batch_diff > threshold))
        stream_disc = int(np.sum(stream_diff > threshold))

        logger.info(f"  Batch discontinuities:  {batch_disc}")
        logger.info(f"  Streaming disc:       {stream_disc}")
        logger.info(f"  Increase:            {stream_disc - batch_disc:+d}")

        # Save output
        output_path = f"test_output/hybrid_test_{chunking_mode}.wav"
        wavfile.write(output_path, 48000, (streaming_output * 32767).astype(np.int16))
        logger.info(f"  Saved to: {output_path}")

        # Reset pipeline for next test
        test_pipeline.clear_cache()

        logger.info("\nAll tests completed!")


if __name__ == "__main__":
    main()
