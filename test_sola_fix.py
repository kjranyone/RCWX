"""Test SOLA fix - verify output length matches input."""
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))

from rcwx.audio.resample import resample
from rcwx.audio.crossfade import flush_sola_buffer
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_test_audio(path: Path, target_sr: int = 48000, duration_sec: float = 5.0) -> np.ndarray:
    """Load and resample audio file."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    samples = int(target_sr * duration_sec)
    return audio[:samples].astype(np.float32)


def process_streaming(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    use_sola: bool,
    chunk_sec: float = 0.096,
) -> np.ndarray:
    """Process audio with streaming."""
    rt_config = RealtimeConfig(
        mic_sample_rate=48000,
        output_sample_rate=48000,
        chunk_sec=chunk_sec,
        context_sec=0.05,
        lookahead_sec=0.0,
        crossfade_sec=0.05,
        use_sola=use_sola,
        prebuffer_chunks=2,
        buffer_margin=0.5,
        pitch_shift=0,
        use_f0=True,
        f0_method="rmvpe",
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=True,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    expected_duration_sec = len(audio) / 48000
    expected_chunks = int(expected_duration_sec / rt_config.chunk_sec) + 2
    max_output_samples = expected_chunks * int(rt_config.output_sample_rate * rt_config.chunk_sec) * 2
    changer.output_buffer.set_max_latency(max_output_samples)

    input_block_size = int(48000 * 0.02)
    output_block_size = int(rt_config.output_sample_rate * 0.02)
    outputs = []

    pos = 0
    while pos < len(audio):
        block = audio[pos : pos + input_block_size]
        if len(block) < input_block_size:
            block = np.pad(block, (0, input_block_size - len(block)))
        pos += input_block_size
        changer.process_input_chunk(block)
        while changer.process_next_chunk():
            pass
        changer.get_output_chunk(0)

    while changer.process_next_chunk():
        changer.get_output_chunk(0)

    # Flush remaining SOLA buffer if SOLA is enabled
    if use_sola and changer._sola_state is not None:
        final_buffer = flush_sola_buffer(changer._sola_state)
        if len(final_buffer) > 0:
            changer.output_buffer.add(final_buffer)
            logger.info(f"Flushed final SOLA buffer: {len(final_buffer)} samples")

    while changer.output_buffer.available > 0:
        output_block = changer.get_output_chunk(output_block_size)
        outputs.append(output_block)

    if outputs:
        return np.concatenate(outputs)
    else:
        return np.array([], dtype=np.float32)


def main():
    logger.info("="*70)
    logger.info("SOLA w-okada Mode Test")
    logger.info("="*70)

    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("ERROR: No model configured.")
        return False

    logger.info(f"Loading model: {config.last_model_path}")
    pipeline = RVCPipeline(
        config.last_model_path,
        device=config.device,
        use_compile=False,
    )
    pipeline.load()

    test_file = Path("sample_data/seki.wav")
    if not test_file.exists():
        logger.error(f"ERROR: Test file not found: {test_file}")
        return False

    logger.info(f"Loading test audio: {test_file}")
    audio = load_test_audio(test_file, target_sr=48000, duration_sec=5.0)
    input_duration = len(audio) / 48000
    logger.info(f"Input: {input_duration:.2f}s ({len(audio)} samples)")

    # Test SOLA ON
    logger.info("\n--- Testing SOLA ON ---")
    output_sola_on = process_streaming(pipeline, audio, use_sola=True, chunk_sec=0.096)
    output_duration_on = len(output_sola_on) / 48000
    logger.info(f"Output: {output_duration_on:.2f}s ({len(output_sola_on)} samples)")

    # Test SOLA OFF
    logger.info("\n--- Testing SOLA OFF ---")
    output_sola_off = process_streaming(pipeline, audio, use_sola=False, chunk_sec=0.096)
    output_duration_off = len(output_sola_off) / 48000
    logger.info(f"Output: {output_duration_off:.2f}s ({len(output_sola_off)} samples)")

    # Save outputs
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    wavfile.write(output_dir / "sola_on_fixed.wav", 48000, (output_sola_on * 32767).astype(np.int16))
    wavfile.write(output_dir / "sola_off_fixed.wav", 48000, (output_sola_off * 32767).astype(np.int16))

    # Check results
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    logger.info(f"Input:           {input_duration:.2f}s")
    logger.info(f"SOLA ON output:  {output_duration_on:.2f}s ({output_duration_on/input_duration*100:.1f}%)")
    logger.info(f"SOLA OFF output: {output_duration_off:.2f}s ({output_duration_off/input_duration*100:.1f}%)")

    # Expected: both should be close to input duration (allowing for some variation)
    tolerance = 0.5  # 0.5 seconds tolerance
    sola_on_ok = abs(output_duration_on - input_duration) < tolerance
    sola_off_ok = abs(output_duration_off - input_duration) < tolerance

    if sola_on_ok and sola_off_ok:
        logger.info("\n✅ PASS: Both SOLA ON and OFF produce correct output length!")
        logger.info("SOLA bug has been fixed.")
        return True
    else:
        if not sola_on_ok:
            logger.error(f"\n❌ FAIL: SOLA ON output too short/long (diff={output_duration_on - input_duration:.2f}s)")
        if not sola_off_ok:
            logger.error(f"\n❌ FAIL: SOLA OFF output too short/long (diff={output_duration_off - input_duration:.2f}s)")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
