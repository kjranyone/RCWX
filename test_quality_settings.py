"""Test different settings to find the best audio quality configuration."""
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))

from rcwx.audio.resample import resample
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

    # Take first N seconds
    samples = int(target_sr * duration_sec)
    return audio[:samples].astype(np.float32)


def process_with_settings(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    chunk_sec: float,
    use_feature_cache: bool,
    use_sola: bool,
    voice_gate_mode: str,
) -> tuple[np.ndarray, dict]:
    """Process audio with specific settings."""
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
        voice_gate_mode=voice_gate_mode,
        use_feature_cache=use_feature_cache,
    )

    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True

    # Set large output buffer
    expected_duration_sec = len(audio) / 48000
    expected_chunks = int(expected_duration_sec / rt_config.chunk_sec) + 2
    max_output_samples = expected_chunks * int(rt_config.output_sample_rate * rt_config.chunk_sec) * 2
    changer.output_buffer.set_max_latency(max_output_samples)

    # Feed audio
    input_block_size = int(48000 * 0.02)
    output_block_size = int(rt_config.output_sample_rate * 0.02)
    outputs = []

    pos = 0
    chunks_processed = 0

    while pos < len(audio):
        block = audio[pos : pos + input_block_size]
        if len(block) < input_block_size:
            block = np.pad(block, (0, input_block_size - len(block)))
        pos += input_block_size

        changer.process_input_chunk(block)

        while changer.process_next_chunk():
            chunks_processed += 1

        changer.get_output_chunk(0)

    # Flush remaining
    while changer.process_next_chunk():
        chunks_processed += 1
        changer.get_output_chunk(0)

    # Collect output
    total_output_retrieved = 0
    while changer.output_buffer.available > 0:
        output_block = changer.get_output_chunk(output_block_size)
        outputs.append(output_block)
        total_output_retrieved += len(output_block)

    stats = {
        "chunks_processed": chunks_processed,
        "buffer_underruns": changer.stats.buffer_underruns,
        "buffer_overruns": changer.stats.buffer_overruns,
    }

    if outputs:
        return np.concatenate(outputs), stats
    else:
        return np.array([], dtype=np.float32), stats


def compare_to_reference(test_output: np.ndarray, ref_output: np.ndarray) -> dict:
    """Compare test output to reference."""
    min_len = min(len(test_output), len(ref_output))
    if min_len == 0:
        return {"error": "Empty output"}

    test_trim = test_output[:min_len]
    ref_trim = ref_output[:min_len]

    diff = test_trim - ref_trim
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    correlation = np.corrcoef(test_trim, ref_trim)[0, 1]

    test_energy = np.sqrt(np.mean(test_trim**2))
    ref_energy = np.sqrt(np.mean(ref_trim**2))
    energy_ratio = test_energy / ref_energy if ref_energy > 0 else 0

    return {
        "correlation": correlation,
        "mae": mae,
        "rmse": rmse,
        "energy_ratio": energy_ratio,
        "length": min_len,
    }


def main():
    logger.info("="*70)
    logger.info("Quality Settings Test")
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
    duration = len(audio) / 48000
    logger.info(f"Audio: {duration:.2f}s @ 48kHz")

    # Test configurations
    test_configs = [
        # Baseline (current default)
        {
            "name": "Baseline (all ON)",
            "chunk_sec": 0.096,
            "use_feature_cache": True,
            "use_sola": True,
            "voice_gate_mode": "expand",
        },
        # Voice Gate variants
        {
            "name": "Voice Gate OFF",
            "chunk_sec": 0.096,
            "use_feature_cache": True,
            "use_sola": True,
            "voice_gate_mode": "off",
        },
        {
            "name": "Voice Gate STRICT",
            "chunk_sec": 0.096,
            "use_feature_cache": True,
            "use_sola": True,
            "voice_gate_mode": "strict",
        },
        # Feature Cache OFF
        {
            "name": "Feature Cache OFF",
            "chunk_sec": 0.096,
            "use_feature_cache": False,
            "use_sola": True,
            "voice_gate_mode": "expand",
        },
        # SOLA OFF
        {
            "name": "SOLA OFF",
            "chunk_sec": 0.096,
            "use_feature_cache": True,
            "use_sola": False,
            "voice_gate_mode": "expand",
        },
        # Larger chunk
        {
            "name": "Chunk 150ms",
            "chunk_sec": 0.150,
            "use_feature_cache": True,
            "use_sola": True,
            "voice_gate_mode": "expand",
        },
        # All OFF (minimal processing)
        {
            "name": "All OFF (minimal)",
            "chunk_sec": 0.096,
            "use_feature_cache": False,
            "use_sola": False,
            "voice_gate_mode": "off",
        },
    ]

    results = []
    reference_output = None

    for i, cfg in enumerate(test_configs):
        logger.info(f"\n{'='*70}")
        logger.info(f"Test {i+1}/{len(test_configs)}: {cfg['name']}")
        logger.info(f"{'='*70}")
        logger.info(f"  chunk_sec: {cfg['chunk_sec']}")
        logger.info(f"  use_feature_cache: {cfg['use_feature_cache']}")
        logger.info(f"  use_sola: {cfg['use_sola']}")
        logger.info(f"  voice_gate_mode: {cfg['voice_gate_mode']}")

        try:
            output, stats = process_with_settings(
                pipeline,
                audio,
                chunk_sec=cfg["chunk_sec"],
                use_feature_cache=cfg["use_feature_cache"],
                use_sola=cfg["use_sola"],
                voice_gate_mode=cfg["voice_gate_mode"],
            )

            logger.info(f"\nOutput: {len(output)} samples")
            logger.info(f"Buffer underruns: {stats['buffer_underruns']}")
            logger.info(f"Buffer overruns: {stats['buffer_overruns']}")

            # Save output
            output_dir = Path("test_output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{i+1:02d}_{cfg['name'].replace(' ', '_')}.wav"
            wavfile.write(output_file, 48000, (output * 32767).astype(np.int16))
            logger.info(f"Saved: {output_file}")

            # Use first output as reference
            if i == 0:
                reference_output = output
                result = {
                    "name": cfg["name"],
                    "correlation": 1.0,
                    "mae": 0.0,
                    "energy_ratio": 1.0,
                    "underruns": stats["buffer_underruns"],
                }
            else:
                metrics = compare_to_reference(output, reference_output)
                logger.info(f"\nQuality vs Baseline:")
                logger.info(f"  Correlation: {metrics.get('correlation', 0):.4f}")
                logger.info(f"  MAE: {metrics.get('mae', 0):.4f}")
                logger.info(f"  Energy ratio: {metrics.get('energy_ratio', 0):.4f}")

                result = {
                    "name": cfg["name"],
                    "correlation": metrics.get("correlation", 0),
                    "mae": metrics.get("mae", 0),
                    "energy_ratio": metrics.get("energy_ratio", 0),
                    "underruns": stats["buffer_underruns"],
                }

            results.append(result)

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY - Quality Comparison")
    logger.info(f"{'='*70}")
    logger.info(f"{'Config':<25} {'Corr':<8} {'MAE':<8} {'Energy':<8} {'Under':<7}")
    logger.info("-" * 70)

    for r in results:
        logger.info(
            f"{r['name']:<25} {r['correlation']:<8.4f} {r['mae']:<8.4f} "
            f"{r['energy_ratio']:<8.4f} {r['underruns']:<7}"
        )

    logger.info(f"\n出力ファイル: test_output/*.wav")
    logger.info("各設定での音声を聞き比べて、最も良い音質の設定を確認してください。")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
