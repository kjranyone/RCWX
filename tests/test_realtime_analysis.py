"""
Realtime Voice Changer Analysis Framework

GUIと冪等な検証環境を提供し、各処理ステップの中間結果を保存・解析する。
問題箇所を特定し、解析的に改善を進めるためのツール。

Usage:
    uv run python tests/test_realtime_analysis.py
    uv run python tests/test_realtime_analysis.py --visualize
    uv run python tests/test_realtime_analysis.py --chunk-sec 0.15 --f0-method fcpe
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeConfig, RealtimeVoiceChanger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkAnalysis:
    """Single chunk analysis result."""
    chunk_idx: int
    input_samples: int
    output_samples: int
    input_rms: float
    output_rms: float
    inference_ms: float
    sola_offset: Optional[int] = None
    sola_correlation: Optional[float] = None
    discontinuity_count: int = 0
    max_discontinuity: float = 0.0


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    config: dict
    total_chunks: int
    total_input_samples: int
    total_output_samples: int
    chunks: list[ChunkAnalysis] = field(default_factory=list)

    # Global metrics
    correlation_vs_batch: float = 0.0
    mae_vs_batch: float = 0.0
    rmse_vs_batch: float = 0.0
    total_discontinuities: int = 0
    energy_ratio: float = 1.0

    # Timing
    total_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    max_inference_ms: float = 0.0


def load_test_audio(path: Path, target_sr: int = 48000) -> np.ndarray:
    """Load and resample audio file to target sample rate."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def create_gui_equivalent_config(
    config: RCWXConfig,
    chunk_sec: float = 0.15,
    f0_method: str = "fcpe",
    chunking_mode: str = "wokada",
    context_sec: float = 0.10,
    crossfade_sec: float = 0.05,
    use_sola: bool = True,
    mic_sample_rate: int = 48000,
    output_sample_rate: int = 48000,
) -> RealtimeConfig:
    """
    Create RealtimeConfig that matches GUI settings.

    This ensures test results are idempotent with GUI processing.
    """
    return RealtimeConfig(
        mic_sample_rate=mic_sample_rate,
        output_sample_rate=output_sample_rate,
        chunk_sec=chunk_sec,
        context_sec=context_sec,
        crossfade_sec=crossfade_sec,
        use_sola=use_sola,
        prebuffer_chunks=0,  # No prebuffer for testing
        pitch_shift=0,
        use_f0=True,
        f0_method=f0_method,
        chunking_mode=chunking_mode,
        rvc_overlap_sec=crossfade_sec,
        index_rate=0.0,
        voice_gate_mode="off",  # Disable for analysis
        use_feature_cache=True,
        use_parallel_extraction=True,
        denoise_enabled=False,
        use_energy_normalization=False,
        use_peak_normalization=False,
    )


def process_batch_reference(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    f0_method: str = "fcpe",
    output_sample_rate: int = 48000,
) -> np.ndarray:
    """
    Process entire audio in one shot (gold standard reference).
    """
    pipeline.clear_cache()

    # Resample to 16kHz for processing
    audio_16k = resample(audio, 48000, 16000)

    # Single inference
    output = pipeline.infer(
        audio_16k,
        input_sr=16000,
        pitch_shift=0,
        f0_method=f0_method,
        index_rate=0.0,
        voice_gate_mode="off",
        use_feature_cache=False,
    )

    # Resample to output rate
    if pipeline.sample_rate != output_sample_rate:
        output = resample(output, pipeline.sample_rate, output_sample_rate)

    return output


def process_streaming_with_analysis(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    rt_config: RealtimeConfig,
    save_intermediates: bool = True,
    output_dir: Optional[Path] = None,
) -> tuple[np.ndarray, AnalysisResult]:
    """
    Process audio in streaming mode with detailed analysis.

    Returns:
        Tuple of (output_audio, analysis_result)
    """
    import time

    if output_dir is None:
        output_dir = Path("test_output/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create RealtimeVoiceChanger
    changer = RealtimeVoiceChanger(pipeline, config=rt_config)
    pipeline.clear_cache()

    # Initialize
    changer._recalculate_buffers()
    changer._running = True

    # Prepare for analysis
    analysis = AnalysisResult(
        config=asdict(rt_config),
        total_chunks=0,
        total_input_samples=len(audio),
        total_output_samples=0,
    )

    # Set large output buffer for testing
    mic_sample_rate = rt_config.mic_sample_rate
    expected_duration_sec = len(audio) / mic_sample_rate
    expected_chunks = int(expected_duration_sec / rt_config.chunk_sec) + 2
    max_output_samples = expected_chunks * int(rt_config.output_sample_rate * rt_config.chunk_sec) * 2
    changer.output_buffer.set_max_latency(max_output_samples)

    # Simulate streaming
    input_block_size = int(mic_sample_rate * 0.02)  # 20ms blocks
    output_block_size = int(rt_config.output_sample_rate * 0.02)

    outputs = []
    intermediate_outputs = []
    pos = 0
    chunks_processed = 0

    logger.info(f"Processing {len(audio)/mic_sample_rate:.2f}s audio in {rt_config.chunking_mode} mode")
    logger.info(f"chunk_sec={rt_config.chunk_sec}, context_sec={rt_config.context_sec}, crossfade_sec={rt_config.crossfade_sec}")

    while pos < len(audio):
        # Feed input block
        block = audio[pos : pos + input_block_size]
        if len(block) < input_block_size:
            block = np.pad(block, (0, input_block_size - len(block)))
        pos += input_block_size

        changer.process_input_chunk(block)

        # Process chunks and collect analysis
        while True:
            start_time = time.perf_counter()
            processed = changer.process_next_chunk()
            inference_time = (time.perf_counter() - start_time) * 1000

            if not processed:
                break

            chunks_processed += 1

            # Get output for this chunk
            chunk_output = changer.get_output_chunk(0)

            # Create chunk analysis
            chunk_analysis = ChunkAnalysis(
                chunk_idx=chunks_processed - 1,
                input_samples=changer.mic_chunk_samples,
                output_samples=changer.output_buffer.available,
                input_rms=float(np.sqrt(np.mean(block**2))),
                output_rms=float(np.sqrt(np.mean(chunk_output**2))) if len(chunk_output) > 0 else 0.0,
                inference_ms=inference_time,
            )

            # Check for discontinuities in this chunk's output
            if len(chunk_output) > 1:
                diff = np.abs(np.diff(chunk_output))
                jumps = diff > 0.2
                chunk_analysis.discontinuity_count = int(np.sum(jumps))
                chunk_analysis.max_discontinuity = float(np.max(diff)) if len(diff) > 0 else 0.0

            analysis.chunks.append(chunk_analysis)

            # Save intermediate if requested
            if save_intermediates and chunks_processed <= 10:
                intermediate_outputs.append(chunk_output.copy())

    # Flush final buffer
    changer.flush_final_sola_buffer()
    changer.get_output_chunk(0)

    # Collect all output
    while changer.output_buffer.available > 0:
        output_block = changer.get_output_chunk(output_block_size)
        outputs.append(output_block)

    # Concatenate outputs
    if outputs:
        final_output = np.concatenate(outputs)
    else:
        final_output = np.array([], dtype=np.float32)

    # Update analysis
    analysis.total_chunks = chunks_processed
    analysis.total_output_samples = len(final_output)

    if analysis.chunks:
        analysis.total_inference_ms = sum(c.inference_ms for c in analysis.chunks)
        analysis.avg_inference_ms = analysis.total_inference_ms / len(analysis.chunks)
        analysis.max_inference_ms = max(c.inference_ms for c in analysis.chunks)
        analysis.total_discontinuities = sum(c.discontinuity_count for c in analysis.chunks)

    # Save intermediates
    if save_intermediates and intermediate_outputs:
        for i, chunk in enumerate(intermediate_outputs[:5]):
            if len(chunk) > 0:
                wavfile.write(
                    output_dir / f"chunk_{i:03d}.wav",
                    rt_config.output_sample_rate,
                    (chunk * 32767).astype(np.int16),
                )

    logger.info(f"Processed {chunks_processed} chunks, output {len(final_output)} samples")

    return final_output, analysis


def compare_outputs(batch: np.ndarray, streaming: np.ndarray, analysis: AnalysisResult) -> AnalysisResult:
    """Compare batch and streaming outputs, updating analysis."""
    min_len = min(len(batch), len(streaming))

    if min_len == 0:
        return analysis

    batch_trim = batch[:min_len]
    streaming_trim = streaming[:min_len]

    # Calculate metrics
    diff = batch_trim - streaming_trim
    analysis.mae_vs_batch = float(np.mean(np.abs(diff)))
    analysis.rmse_vs_batch = float(np.sqrt(np.mean(diff**2)))
    analysis.correlation_vs_batch = float(np.corrcoef(batch_trim, streaming_trim)[0, 1])

    # Energy ratio
    batch_energy = np.sqrt(np.mean(batch_trim**2))
    streaming_energy = np.sqrt(np.mean(streaming_trim**2))
    analysis.energy_ratio = float(streaming_energy / batch_energy) if batch_energy > 0 else 0.0

    return analysis


def count_discontinuities(audio: np.ndarray, threshold: float = 0.2) -> tuple[int, np.ndarray, np.ndarray]:
    """Count audio discontinuities (clicks/pops)."""
    if len(audio) < 2:
        return 0, np.array([]), np.array([])
    diff = np.abs(np.diff(audio))
    jumps = np.where(diff > threshold)[0]
    jump_values = diff[jumps]
    return len(jumps), jumps, jump_values


def visualize_results(
    audio_input: np.ndarray,
    batch_output: np.ndarray,
    streaming_output: np.ndarray,
    analysis: AnalysisResult,
    output_dir: Path,
    sample_rate: int = 48000,
):
    """Generate visualization of results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping visualization")
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    # Time axis
    t_input = np.arange(len(audio_input)) / sample_rate
    t_batch = np.arange(len(batch_output)) / sample_rate
    t_streaming = np.arange(len(streaming_output)) / sample_rate

    # 1. Input waveform
    axes[0].plot(t_input, audio_input, 'b-', linewidth=0.5)
    axes[0].set_title('Input Audio')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(0, max(t_input[-1] if len(t_input) else 1, 0.1))

    # 2. Batch vs Streaming comparison
    axes[1].plot(t_batch, batch_output, 'g-', linewidth=0.5, label='Batch (Reference)', alpha=0.7)
    axes[1].plot(t_streaming, streaming_output, 'r-', linewidth=0.5, label='Streaming', alpha=0.7)
    axes[1].set_title(f'Batch vs Streaming (Correlation: {analysis.correlation_vs_batch:.4f})')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right')

    # 3. Difference
    min_len = min(len(batch_output), len(streaming_output))
    if min_len > 0:
        diff = batch_output[:min_len] - streaming_output[:min_len]
        t_diff = np.arange(min_len) / sample_rate
        axes[2].plot(t_diff, diff, 'purple', linewidth=0.5)
        axes[2].set_title(f'Difference (RMSE: {analysis.rmse_vs_batch:.6f})')
        axes[2].set_ylabel('Amplitude')

    # 4. Inference time per chunk
    if analysis.chunks:
        chunk_times = [c.inference_ms for c in analysis.chunks]
        chunk_indices = [c.chunk_idx for c in analysis.chunks]
        axes[3].bar(chunk_indices, chunk_times, color='blue', alpha=0.7)
        axes[3].axhline(y=analysis.avg_inference_ms, color='r', linestyle='--', label=f'Avg: {analysis.avg_inference_ms:.1f}ms')
        axes[3].set_title('Inference Time per Chunk')
        axes[3].set_xlabel('Chunk Index')
        axes[3].set_ylabel('Time (ms)')
        axes[3].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_plot.png', dpi=150)
    plt.close()

    logger.info(f"Saved visualization to {output_dir / 'analysis_plot.png'}")


def run_analysis(
    test_file: Path,
    chunk_sec: float = 0.15,
    f0_method: str = "fcpe",
    chunking_mode: str = "wokada",
    context_sec: float = 0.10,
    crossfade_sec: float = 0.05,
    use_sola: bool = True,
    visualize: bool = False,
    max_duration_sec: float = 5.0,
) -> AnalysisResult:
    """Run complete analysis."""
    output_dir = Path("test_output/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("No model configured. Run GUI first to select a model.")
        return None

    # Load pipeline
    logger.info(f"Loading model: {config.last_model_path}")
    pipeline = RVCPipeline(
        config.last_model_path,
        device=config.device,
        use_compile=False,  # Disable for deterministic results
    )
    pipeline.load()

    # Load test audio
    logger.info(f"Loading test audio: {test_file}")
    audio = load_test_audio(test_file, target_sr=48000)

    # Limit duration
    max_samples = int(48000 * max_duration_sec)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        logger.info(f"Trimmed audio to {max_duration_sec}s")

    # Create GUI-equivalent config
    rt_config = create_gui_equivalent_config(
        config,
        chunk_sec=chunk_sec,
        f0_method=f0_method,
        chunking_mode=chunking_mode,
        context_sec=context_sec,
        crossfade_sec=crossfade_sec,
        use_sola=use_sola,
    )

    # Process batch (reference)
    logger.info("=" * 60)
    logger.info("Processing BATCH (reference)")
    logger.info("=" * 60)
    batch_output = process_batch_reference(
        pipeline, audio, f0_method=f0_method, output_sample_rate=48000
    )

    # Process streaming
    logger.info("=" * 60)
    logger.info("Processing STREAMING")
    logger.info("=" * 60)
    pipeline.clear_cache()  # Reset for streaming
    streaming_output, analysis = process_streaming_with_analysis(
        pipeline, audio, rt_config, save_intermediates=True, output_dir=output_dir
    )

    # Compare
    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    analysis = compare_outputs(batch_output, streaming_output, analysis)

    # Count discontinuities
    batch_disc, _, _ = count_discontinuities(batch_output)
    streaming_disc, disc_pos, disc_vals = count_discontinuities(streaming_output)

    logger.info(f"Batch discontinuities: {batch_disc}")
    logger.info(f"Streaming discontinuities: {streaming_disc} ({streaming_disc - batch_disc:+d} vs batch)")

    if streaming_disc > 0 and streaming_disc <= 10:
        for i, (pos, val) in enumerate(zip(disc_pos, disc_vals)):
            time_sec = pos / 48000
            logger.info(f"  #{i+1}: time={time_sec:.3f}s, jump={val:.4f}")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total chunks: {analysis.total_chunks}")
    logger.info(f"Correlation: {analysis.correlation_vs_batch:.4f}")
    logger.info(f"MAE: {analysis.mae_vs_batch:.6f}")
    logger.info(f"RMSE: {analysis.rmse_vs_batch:.6f}")
    logger.info(f"Energy ratio: {analysis.energy_ratio:.4f}")
    logger.info(f"Avg inference: {analysis.avg_inference_ms:.1f}ms")
    logger.info(f"Max inference: {analysis.max_inference_ms:.1f}ms")
    logger.info(f"Total discontinuities in chunks: {analysis.total_discontinuities}")

    # Save outputs
    wavfile.write(
        output_dir / "batch_output.wav",
        48000,
        (batch_output * 32767).astype(np.int16),
    )
    wavfile.write(
        output_dir / "streaming_output.wav",
        48000,
        (streaming_output * 32767).astype(np.int16),
    )
    wavfile.write(
        output_dir / "input.wav",
        48000,
        (audio * 32767).astype(np.int16),
    )

    # Save analysis as JSON
    analysis_dict = asdict(analysis)
    # Convert chunks to simple dicts for JSON serialization
    analysis_dict['chunks'] = [asdict(c) for c in analysis.chunks]
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis_dict, f, indent=2)

    logger.info(f"Results saved to {output_dir}/")

    # Visualize
    if visualize:
        visualize_results(audio, batch_output, streaming_output, analysis, output_dir)

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Realtime Voice Changer Analysis")
    parser.add_argument("--test-file", type=Path, default=Path("sample_data/sustained_voice.wav"),
                       help="Path to test audio file")
    parser.add_argument("--chunk-sec", type=float, default=0.15,
                       help="Chunk size in seconds")
    parser.add_argument("--f0-method", choices=["fcpe", "rmvpe"], default="fcpe",
                       help="F0 extraction method")
    parser.add_argument("--chunking-mode", choices=["wokada", "rvc_webui", "hybrid"], default="wokada",
                       help="Chunking mode")
    parser.add_argument("--context-sec", type=float, default=0.10,
                       help="Context size in seconds")
    parser.add_argument("--crossfade-sec", type=float, default=0.05,
                       help="Crossfade size in seconds")
    parser.add_argument("--no-sola", action="store_true",
                       help="Disable SOLA")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization")
    parser.add_argument("--max-duration", type=float, default=5.0,
                       help="Maximum audio duration in seconds")

    args = parser.parse_args()

    if not args.test_file.exists():
        logger.error(f"Test file not found: {args.test_file}")
        sys.exit(1)

    analysis = run_analysis(
        test_file=args.test_file,
        chunk_sec=args.chunk_sec,
        f0_method=args.f0_method,
        chunking_mode=args.chunking_mode,
        context_sec=args.context_sec,
        crossfade_sec=args.crossfade_sec,
        use_sola=not args.no_sola,
        visualize=args.visualize,
        max_duration_sec=args.max_duration,
    )

    if analysis is None:
        sys.exit(1)

    # Exit with error if correlation is too low
    if analysis.correlation_vs_batch < 0.9:
        logger.error(f"Correlation too low: {analysis.correlation_vs_batch:.4f} < 0.9")
        sys.exit(1)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
