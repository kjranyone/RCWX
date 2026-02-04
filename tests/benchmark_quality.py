"""
Quality Benchmark - バッチ vs リアルタイム処理の品質差分析

各処理ステージでの誤差累積を特定し、根本原因を明らかにする。

Usage:
    uv run python tests/benchmark_quality.py
    uv run python tests/benchmark_quality.py --visualize
    uv run python tests/benchmark_quality.py --test-file sample_data/sustained_voice.wav
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.resample import resample, StatefulResampler
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade, flush_sola_buffer
from rcwx.audio.buffer import ChunkBuffer
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a single comparison."""
    name: str
    correlation: float = 0.0
    rmse: float = 0.0
    max_error: float = 0.0
    snr_db: float = 0.0
    length_diff: int = 0
    phase_shift_samples: int = 0

    def __str__(self):
        return (
            f"{self.name}: corr={self.correlation:.6f}, rmse={self.rmse:.6f}, "
            f"max_err={self.max_error:.4f}, snr={self.snr_db:.1f}dB, "
            f"len_diff={self.length_diff}, phase={self.phase_shift_samples}"
        )


def compute_metrics(batch: np.ndarray, streaming: np.ndarray, name: str) -> QualityMetrics:
    """Compute quality metrics between batch and streaming outputs."""
    metrics = QualityMetrics(name=name)

    if len(batch) == 0 or len(streaming) == 0:
        return metrics

    # Length difference
    metrics.length_diff = len(streaming) - len(batch)

    # Align lengths for comparison
    min_len = min(len(batch), len(streaming))
    b = batch[:min_len].astype(np.float64)
    s = streaming[:min_len].astype(np.float64)

    # Find phase shift using cross-correlation
    if min_len > 100:
        max_lag = min(1000, min_len // 4)
        corr = correlate(s[:max_lag*2], b[:max_lag*2], mode='full')
        lag = np.argmax(corr) - (len(b[:max_lag*2]) - 1)
        metrics.phase_shift_samples = int(lag)

        # Align based on phase shift
        if abs(lag) < max_lag:
            if lag > 0:
                b = batch[lag:min_len].astype(np.float64)
                s = streaming[:min_len-lag].astype(np.float64)
            elif lag < 0:
                b = batch[:min_len+lag].astype(np.float64)
                s = streaming[-lag:min_len].astype(np.float64)

    if len(b) == 0 or len(s) == 0:
        return metrics

    # Correlation
    if np.std(b) > 1e-10 and np.std(s) > 1e-10:
        metrics.correlation = float(np.corrcoef(b, s)[0, 1])

    # RMSE
    diff = b - s
    metrics.rmse = float(np.sqrt(np.mean(diff ** 2)))
    metrics.max_error = float(np.max(np.abs(diff)))

    # SNR
    signal_power = np.mean(b ** 2)
    noise_power = np.mean(diff ** 2)
    if noise_power > 1e-10:
        metrics.snr_db = float(10 * np.log10(signal_power / noise_power))
    else:
        metrics.snr_db = float('inf')

    return metrics


def load_test_audio(path: Path, target_sr: int = 48000) -> np.ndarray:
    """Load and resample audio file."""
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


def analyze_resampler_error(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    chunk_sec: float,
) -> QualityMetrics:
    """Compare batch vs streaming resampling."""
    print("\n=== Resampler Analysis ===")

    # Batch resampling
    batch_result = resample(audio, orig_sr, target_sr)

    # Streaming resampling
    resampler = StatefulResampler(orig_sr, target_sr)
    chunk_samples = int(orig_sr * chunk_sec)
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples // 2:
            break
        resampled = resampler.resample_chunk(chunk)
        chunks.append(resampled)

    streaming_result = np.concatenate(chunks) if chunks else np.array([])

    metrics = compute_metrics(batch_result, streaming_result, "Resampler")
    print(f"  Batch length: {len(batch_result)}")
    print(f"  Streaming length: {len(streaming_result)}")
    print(f"  {metrics}")

    return metrics


def analyze_f0_error(
    pipeline: RVCPipeline,
    audio_16k: np.ndarray,
    chunk_sec: float,
    f0_method: str,
) -> QualityMetrics:
    """Compare batch vs streaming F0 extraction."""
    print("\n=== F0 Extraction Analysis ===")

    # RMVPE requires minimum ~320ms, skip individual chunk analysis
    min_chunk_sec = 0.35 if f0_method == "rmvpe" else 0.10
    if chunk_sec < min_chunk_sec:
        print(f"  Skipping: chunk_sec {chunk_sec}s < minimum {min_chunk_sec}s for {f0_method}")
        return QualityMetrics(name="F0", correlation=1.0)

    # Batch F0
    audio_tensor = torch.from_numpy(audio_16k).float().to(pipeline.device)
    try:
        if f0_method == "rmvpe" and pipeline.rmvpe is not None:
            batch_f0 = pipeline.rmvpe.infer(audio_tensor).cpu().numpy().flatten()
        elif f0_method == "fcpe" and pipeline.fcpe is not None:
            batch_f0 = pipeline.fcpe.infer(audio_tensor).cpu().numpy().flatten()
        else:
            print("  F0 method not available")
            return QualityMetrics(name="F0")
    except Exception as e:
        print(f"  Batch F0 failed: {e}")
        return QualityMetrics(name="F0")

    # Streaming F0 with larger overlap for RMVPE
    chunk_samples = int(16000 * chunk_sec)
    hop_samples = chunk_samples // 2 if f0_method == "rmvpe" else chunk_samples  # 50% overlap for RMVPE
    f0_chunks = []
    errors = 0

    for i in range(0, len(audio_16k) - chunk_samples, hop_samples):
        chunk = audio_16k[i:i + chunk_samples]
        chunk_tensor = torch.from_numpy(chunk).float().to(pipeline.device)
        try:
            if f0_method == "rmvpe" and pipeline.rmvpe is not None:
                f0 = pipeline.rmvpe.infer(chunk_tensor).cpu().numpy().flatten()
            elif f0_method == "fcpe" and pipeline.fcpe is not None:
                f0 = pipeline.fcpe.infer(chunk_tensor).cpu().numpy().flatten()
            f0_chunks.append(f0)
        except Exception:
            errors += 1
            continue

    if errors > 0:
        print(f"  {errors} chunks failed F0 extraction")

    if not f0_chunks:
        print("  No F0 chunks extracted")
        return QualityMetrics(name="F0")

    streaming_f0 = np.concatenate(f0_chunks)

    metrics = compute_metrics(batch_f0, streaming_f0, "F0")
    print(f"  Batch F0 length: {len(batch_f0)}")
    print(f"  Streaming F0 length: {len(streaming_f0)}")
    print(f"  {metrics}")

    # Analyze F0 continuity at chunk boundaries
    print("\n  F0 Boundary Analysis:")
    jumps = 0
    for i in range(1, len(f0_chunks)):
        if len(f0_chunks[i-1]) > 0 and len(f0_chunks[i]) > 0:
            prev_end = f0_chunks[i-1][-1]
            curr_start = f0_chunks[i][0]
            jump = abs(curr_start - prev_end)
            if jump > 50:  # Hz
                jumps += 1
                if jumps <= 5:
                    print(f"    Chunk {i}: F0 jump = {jump:.1f} Hz")
    if jumps > 5:
        print(f"    ... and {jumps - 5} more jumps")

    return metrics


def analyze_hubert_error(
    pipeline: RVCPipeline,
    audio_16k: np.ndarray,
    chunk_sec: float,
) -> QualityMetrics:
    """Compare batch vs streaming HuBERT extraction."""
    print("\n=== HuBERT Feature Analysis ===")

    # Batch HuBERT
    audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(pipeline.device)
    with torch.no_grad():
        batch_features = pipeline.hubert.extract(audio_tensor).cpu().numpy()[0]

    # Streaming HuBERT (without context)
    chunk_samples = int(16000 * chunk_sec)
    feature_chunks = []
    for i in range(0, len(audio_16k), chunk_samples):
        chunk = audio_16k[i:i + chunk_samples]
        if len(chunk) < chunk_samples // 2:
            break
        chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(pipeline.device)
        with torch.no_grad():
            features = pipeline.hubert.extract(chunk_tensor).cpu().numpy()[0]
        feature_chunks.append(features)

    if not feature_chunks:
        return QualityMetrics(name="HuBERT")

    streaming_features = np.concatenate(feature_chunks, axis=0)

    # Compare feature statistics
    print(f"  Batch features: {batch_features.shape}")
    print(f"  Streaming features: {streaming_features.shape}")

    # Flatten for correlation
    min_frames = min(batch_features.shape[0], streaming_features.shape[0])
    b_flat = batch_features[:min_frames].flatten()
    s_flat = streaming_features[:min_frames].flatten()

    metrics = compute_metrics(b_flat, s_flat, "HuBERT")
    print(f"  {metrics}")

    # Analyze boundary continuity
    print("\n  HuBERT Boundary Analysis:")
    for i in range(1, len(feature_chunks)):
        if len(feature_chunks[i-1]) > 0 and len(feature_chunks[i]) > 0:
            prev_end = feature_chunks[i-1][-1]
            curr_start = feature_chunks[i][0]
            cos_sim = np.dot(prev_end, curr_start) / (np.linalg.norm(prev_end) * np.linalg.norm(curr_start) + 1e-8)
            if cos_sim < 0.9:
                print(f"    Chunk {i}: Feature similarity = {cos_sim:.4f}")

    return metrics


def analyze_full_pipeline_error(
    pipeline: RVCPipeline,
    audio: np.ndarray,
    mic_sr: int,
    chunk_sec: float,
    context_sec: float,
    f0_method: str,
) -> QualityMetrics:
    """Compare batch vs streaming full pipeline."""
    print("\n=== Full Pipeline Analysis ===")

    # Batch processing
    audio_16k = resample(audio, mic_sr, 16000)
    pipeline.clear_cache()
    batch_output = pipeline.infer(
        audio_16k,
        input_sr=16000,
        f0_method=f0_method,
        use_feature_cache=False,
        use_parallel_extraction=True,
        allow_short_input=True,
    )
    batch_output = resample(batch_output, pipeline.sample_rate, mic_sr)

    # Streaming processing (simulating realtime)
    pipeline.clear_cache()
    chunk_samples = int(mic_sr * chunk_sec)
    context_samples = int(mic_sr * context_sec)

    # Use ChunkBuffer for realistic chunking
    buffer = ChunkBuffer(
        chunk_samples=chunk_samples,
        crossfade_samples=0,
        context_samples=context_samples,
        lookahead_samples=0,
    )
    buffer.add_input(audio)

    # Setup SOLA with Phase 8 settings (strong crossfade for RVC)
    crossfade_samples = int(mic_sr * 0.05)
    sola_state = SOLAState.create(
        crossfade_samples,
        mic_sr,
        use_advanced_sola=True,
        fallback_threshold=0.8,  # Always use strong crossfade
    )

    input_resampler = StatefulResampler(mic_sr, 16000)
    output_resampler = StatefulResampler(pipeline.sample_rate, mic_sr)

    output_chunks = []
    chunk_idx = 0

    while buffer.has_chunk():
        chunk = buffer.get_chunk()
        if chunk is None:
            break

        # Resample to 16kHz
        chunk_16k = input_resampler.resample_chunk(chunk)

        # Inference
        output = pipeline.infer(
            chunk_16k,
            input_sr=16000,
            f0_method=f0_method,
            use_feature_cache=True,
            use_parallel_extraction=True,
            allow_short_input=True,
        )

        # Resample to output rate
        output = output_resampler.resample_chunk(output)

        # SOLA crossfade
        ctx_samples = context_samples if chunk_idx > 0 else 0
        cf_result = apply_sola_crossfade(
            output, sola_state, wokada_mode=True, context_samples=ctx_samples
        )
        output_chunks.append(cf_result.audio)
        chunk_idx += 1

    # Flush remaining SOLA buffer
    remaining = flush_sola_buffer(sola_state)
    if len(remaining) > 0:
        output_chunks.append(remaining)

    streaming_output = np.concatenate(output_chunks) if output_chunks else np.array([])

    metrics = compute_metrics(batch_output, streaming_output, "Full Pipeline")
    print(f"  Batch output: {len(batch_output)} samples")
    print(f"  Streaming output: {len(streaming_output)} samples ({chunk_idx} chunks)")
    print(f"  {metrics}")

    return metrics, batch_output, streaming_output


def analyze_sola_error(
    audio: np.ndarray,
    sample_rate: int,
    chunk_sec: float,
    context_sec: float,
    crossfade_sec: float,
) -> QualityMetrics:
    """Analyze SOLA-induced error on continuous audio."""
    print("\n=== SOLA Crossfade Analysis ===")

    chunk_samples = int(sample_rate * chunk_sec)
    context_samples = int(sample_rate * context_sec)
    crossfade_samples = int(sample_rate * crossfade_sec)

    # Reference: just concatenate chunks without SOLA
    no_sola_chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples // 2:
            break
        # Trim context from all but first chunk
        if i > 0 and len(chunk) > context_samples:
            chunk = chunk[context_samples:]
        no_sola_chunks.append(chunk)
    no_sola_output = np.concatenate(no_sola_chunks) if no_sola_chunks else np.array([])

    # With SOLA (Phase 8 settings)
    sola_state = SOLAState.create(
        crossfade_samples,
        sample_rate,
        use_advanced_sola=True,
        fallback_threshold=0.8,
    )
    sola_chunks = []
    chunk_idx = 0
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples // 2:
            break
        ctx = context_samples if chunk_idx > 0 else 0
        cf_result = apply_sola_crossfade(chunk, sola_state, wokada_mode=True, context_samples=ctx)
        sola_chunks.append(cf_result.audio)
        chunk_idx += 1

    # Flush remaining SOLA buffer
    remaining = flush_sola_buffer(sola_state)
    if len(remaining) > 0:
        sola_chunks.append(remaining)

    sola_output = np.concatenate(sola_chunks) if sola_chunks else np.array([])

    # Compare SOLA output to original audio
    metrics = compute_metrics(audio[:len(sola_output)], sola_output, "SOLA vs Original")
    print(f"  Original: {len(audio)} samples")
    print(f"  SOLA output: {len(sola_output)} samples")
    print(f"  {metrics}")

    # Analyze SOLA offset distribution
    print("\n  SOLA Offset Analysis:")
    sola_state2 = SOLAState.create(
        crossfade_samples,
        sample_rate,
        use_advanced_sola=True,
        fallback_threshold=0.8,
    )
    offsets = []
    correlations = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples // 2:
            break
        ctx = context_samples if i > 0 else 0
        cf_result = apply_sola_crossfade(chunk, sola_state2, wokada_mode=True, context_samples=ctx)
        offsets.append(cf_result.sola_offset)
        correlations.append(cf_result.correlation)

    if offsets:
        print(f"    Offset range: [{min(offsets)}, {max(offsets)}] samples")
        print(f"    Offset std: {np.std(offsets):.1f} samples")
        print(f"    Mean correlation: {np.mean(correlations):.4f}")
        print(f"    Min correlation: {np.min(correlations):.4f}")

    return metrics


def analyze_context_importance(
    pipeline: RVCPipeline,
    audio_16k: np.ndarray,
    chunk_sec: float,
    f0_method: str,
) -> dict:
    """Analyze how context affects output quality."""
    print("\n=== Context Importance Analysis ===")

    chunk_samples = int(16000 * chunk_sec)
    results = {}

    for ctx_ratio in [0.0, 0.25, 0.5, 0.75]:
        context_samples = int(chunk_samples * ctx_ratio)
        context_sec = context_samples / 16000

        output_chunks = []
        pipeline.clear_cache()

        for i in range(0, len(audio_16k) - chunk_samples, chunk_samples):
            # Get chunk with context
            ctx_start = max(0, i - context_samples)
            chunk_with_ctx = audio_16k[ctx_start:i + chunk_samples]

            output = pipeline.infer(
                chunk_with_ctx,
                input_sr=16000,
                f0_method=f0_method,
                use_feature_cache=True,
                allow_short_input=True,
            )

            # Trim context from output
            if context_samples > 0 and i > 0:
                ctx_output_samples = int(context_sec * pipeline.sample_rate)
                if len(output) > ctx_output_samples:
                    output = output[ctx_output_samples:]

            output_chunks.append(output)

        streaming = np.concatenate(output_chunks) if output_chunks else np.array([])

        # Compare with batch
        pipeline.clear_cache()
        batch = pipeline.infer(audio_16k, input_sr=16000, f0_method=f0_method, use_feature_cache=False)

        min_len = min(len(batch), len(streaming))
        if min_len > 0:
            corr = np.corrcoef(batch[:min_len], streaming[:min_len])[0, 1]
            rmse = np.sqrt(np.mean((batch[:min_len] - streaming[:min_len])**2))
            results[f"ctx_{ctx_ratio:.0%}"] = {"correlation": corr, "rmse": rmse}
            print(f"  Context {ctx_ratio:.0%}: corr={corr:.4f}, rmse={rmse:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Quality Benchmark")
    parser.add_argument("--test-file", type=Path, default=Path("sample_data/sustained_voice.wav"))
    parser.add_argument("--chunk-sec", type=float, default=0.35)
    parser.add_argument("--context-sec", type=float, default=0.10)
    parser.add_argument("--max-duration", type=float, default=3.0)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    # Load config and pipeline
    config = RCWXConfig.load()
    if not config.last_model_path:
        print("ERROR: No model configured. Run GUI first.")
        sys.exit(1)

    print(f"Loading model: {config.last_model_path}")
    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()

    # Detect F0 method
    try:
        import torchfcpe
        f0_method = "fcpe"
    except ImportError:
        f0_method = "rmvpe"
    print(f"F0 method: {f0_method}")

    # Adjust chunk size for RMVPE
    if f0_method == "rmvpe" and args.chunk_sec < 0.32:
        args.chunk_sec = 0.35
        print(f"Adjusted chunk_sec to {args.chunk_sec}s for RMVPE")

    # Load test audio
    if args.test_file.exists():
        audio = load_test_audio(args.test_file, target_sr=48000)
    else:
        print("Generating test audio (440Hz with harmonics)")
        t = np.linspace(0, args.max_duration, int(48000 * args.max_duration), dtype=np.float32)
        audio = 0.4 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)

    # Limit duration
    max_samples = int(48000 * args.max_duration)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    print(f"\nTest audio: {len(audio)} samples ({len(audio)/48000:.2f}s)")
    print(f"Chunk: {args.chunk_sec}s, Context: {args.context_sec}s")
    print("=" * 70)

    # Run analyses
    results = {}

    # 1. Resampler analysis
    results["resampler"] = analyze_resampler_error(audio, 48000, 16000, args.chunk_sec)

    # 2. F0 analysis
    audio_16k = resample(audio, 48000, 16000)
    results["f0"] = analyze_f0_error(pipeline, audio_16k, args.chunk_sec, f0_method)

    # 3. HuBERT analysis
    results["hubert"] = analyze_hubert_error(pipeline, audio_16k, args.chunk_sec)

    # 4. SOLA analysis
    results["sola"] = analyze_sola_error(audio, 48000, args.chunk_sec, args.context_sec, 0.05)

    # 5. Full pipeline analysis
    metrics, batch_out, stream_out = analyze_full_pipeline_error(
        pipeline, audio, 48000, args.chunk_sec, args.context_sec, f0_method
    )
    results["full_pipeline"] = metrics

    # 6. Context importance
    results["context"] = analyze_context_importance(pipeline, audio_16k, args.chunk_sec, f0_method)

    # Summary
    print("\n" + "=" * 70)
    print("QUALITY ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nError by Stage (correlation with batch):")
    for name, m in results.items():
        if isinstance(m, QualityMetrics):
            status = "[OK]" if m.correlation > 0.95 else "[!!]" if m.correlation > 0.8 else "[XX]"
            print(f"  {status} {m.name:<20}: corr={m.correlation:.4f}, rmse={m.rmse:.6f}")

    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)

    # Identify the stage with biggest quality drop
    stages = [(name, m) for name, m in results.items() if isinstance(m, QualityMetrics)]
    stages.sort(key=lambda x: x[1].correlation)

    if stages:
        worst = stages[0]
        print(f"\nPrimary quality degradation: {worst[0]}")
        print(f"  Correlation: {worst[1].correlation:.4f}")
        print(f"  RMSE: {worst[1].rmse:.6f}")

        if "f0" in worst[0].lower():
            print("\n  Root cause: F0 extraction discontinuity at chunk boundaries")
            print("  - F0 models have receptive field requirements")
            print("  - Chunk edges cause abrupt pitch changes")
            print("  Recommendations:")
            print("    1. Increase context to cover F0 model receptive field")
            print("    2. Use overlapping F0 extraction with blending")
            print("    3. Apply F0 smoothing at chunk boundaries")

        elif "hubert" in worst[0].lower():
            print("\n  Root cause: HuBERT feature discontinuity")
            print("  - Transformer attention depends on full context")
            print("  - Chunk boundaries lose global context")
            print("  Recommendations:")
            print("    1. Use feature caching with overlap")
            print("    2. Apply feature blending at boundaries")

        elif "sola" in worst[0].lower():
            print("\n  Root cause: SOLA offset search failure")
            print("  - Low correlation in search window")
            print("  - Incorrect phase alignment")
            print("  Recommendations:")
            print("    1. Increase search window size")
            print("    2. Use frequency-domain correlation")

    # Visualize if requested
    if args.visualize and len(batch_out) > 0 and len(stream_out) > 0:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(14, 8))

            # Waveforms
            t_batch = np.arange(len(batch_out)) / 48000
            t_stream = np.arange(len(stream_out)) / 48000

            axes[0].plot(t_batch, batch_out, 'b-', alpha=0.7, label='Batch')
            axes[0].plot(t_stream, stream_out, 'r-', alpha=0.7, label='Streaming')
            axes[0].set_title('Batch vs Streaming Output')
            axes[0].legend()

            # Difference
            min_len = min(len(batch_out), len(stream_out))
            diff = batch_out[:min_len] - stream_out[:min_len]
            t_diff = np.arange(min_len) / 48000
            axes[1].plot(t_diff, diff, 'purple')
            axes[1].set_title(f'Difference (RMSE={results["full_pipeline"].rmse:.6f})')

            # Spectrogram difference (if long enough)
            if min_len > 2048:
                from scipy.signal import spectrogram
                f, t, Sxx_b = spectrogram(batch_out[:min_len], 48000, nperseg=1024)
                f, t, Sxx_s = spectrogram(stream_out[:min_len], 48000, nperseg=1024)
                Sxx_diff = np.abs(Sxx_b - Sxx_s)
                axes[2].pcolormesh(t, f[:100], 10*np.log10(Sxx_diff[:100]+1e-10), shading='auto')
                axes[2].set_title('Spectrogram Difference (dB)')
                axes[2].set_ylabel('Frequency (Hz)')

            plt.tight_layout()
            plt.savefig("test_output/quality_analysis.png", dpi=150)
            print(f"\nVisualization saved to test_output/quality_analysis.png")
            plt.close()
        except ImportError:
            print("matplotlib not available for visualization")


if __name__ == "__main__":
    main()
