"""Analyze perceptual discontinuities at chunk boundaries.

This script provides objective metrics for auditory artifacts:
1. Sample-level discontinuities (clicks/pops)
2. Envelope discontinuities (sudden volume changes)
3. Spectral discontinuities (timbre changes)
4. Phase discontinuities (waveform misalignment)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.signal as signal
from numpy.typing import NDArray

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.buffer import ChunkBuffer
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade, flush_sola_buffer
from rcwx.audio.resample import StatefulResampler
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs for cleaner output
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def load_audio(path: str, target_sr: int = 48000) -> tuple[NDArray[np.float32], int]:
    """Load audio file and resample to target sample rate."""
    from scipy.io import wavfile
    import wave

    path = str(path)

    # Try scipy first (WAV files)
    if path.lower().endswith('.wav'):
        sr, audio = wavfile.read(path)
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
    else:
        # Try pydub for other formats (mp3, etc.)
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_file(path)
            sr = sound.frame_rate
            samples = np.array(sound.get_array_of_samples())
            if sound.sample_width == 2:
                audio = samples.astype(np.float32) / 32768.0
            else:
                audio = samples.astype(np.float32)
            if sound.channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
        except ImportError:
            raise ValueError(f"Cannot load {path}: install pydub for non-WAV files")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        from rcwx.audio.resample import resample
        audio = resample(audio, sr, target_sr)

    return audio.astype(np.float32), target_sr


def compute_short_time_energy(
    audio: NDArray[np.float32],
    frame_size: int = 480,  # 10ms at 48kHz
    hop_size: int = 240,    # 5ms hop
) -> NDArray[np.float32]:
    """Compute short-time energy (RMS) of audio."""
    num_frames = (len(audio) - frame_size) // hop_size + 1
    energy = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        energy[i] = np.sqrt(np.mean(frame ** 2))

    return energy


def compute_spectral_flux(
    audio: NDArray[np.float32],
    frame_size: int = 2048,
    hop_size: int = 512,
) -> NDArray[np.float32]:
    """Compute spectral flux (change in spectrum between frames)."""
    # Compute STFT
    f, t, Zxx = signal.stft(audio, fs=48000, nperseg=frame_size, noverlap=frame_size - hop_size)
    magnitude = np.abs(Zxx)

    # Compute flux (L2 norm of spectral difference)
    flux = np.zeros(magnitude.shape[1] - 1, dtype=np.float32)
    for i in range(1, magnitude.shape[1]):
        diff = magnitude[:, i] - magnitude[:, i - 1]
        flux[i - 1] = np.sqrt(np.sum(diff ** 2))

    return flux


def detect_clicks(
    audio: NDArray[np.float32],
    threshold_factor: float = 10.0,
) -> list[tuple[int, float]]:
    """Detect click artifacts (sudden large sample-to-sample changes).

    Returns list of (position, magnitude) tuples.
    """
    # Compute sample-to-sample difference
    diff = np.abs(np.diff(audio))

    # Use median absolute deviation for robust threshold
    median_diff = np.median(diff)
    mad = np.median(np.abs(diff - median_diff))
    threshold = median_diff + threshold_factor * mad * 1.4826  # 1.4826 for normal distribution

    # Find clicks
    click_indices = np.where(diff > threshold)[0]
    clicks = [(int(idx), float(diff[idx])) for idx in click_indices]

    return clicks


def analyze_boundary(
    audio: NDArray[np.float32],
    boundary_pos: int,
    window_ms: float = 20.0,
    sample_rate: int = 48000,
) -> dict:
    """Analyze audio quality around a chunk boundary.

    Args:
        audio: Audio signal
        boundary_pos: Sample position of chunk boundary
        window_ms: Analysis window in milliseconds
        sample_rate: Sample rate

    Returns:
        Dictionary with boundary quality metrics
    """
    window_samples = int(sample_rate * window_ms / 1000)

    # Extract regions around boundary
    pre_start = max(0, boundary_pos - window_samples)
    post_end = min(len(audio), boundary_pos + window_samples)

    pre_region = audio[pre_start:boundary_pos]
    post_region = audio[boundary_pos:post_end]

    if len(pre_region) < 10 or len(post_region) < 10:
        return {"error": "insufficient_samples"}

    metrics = {}

    # 1. Sample discontinuity at boundary
    if boundary_pos > 0 and boundary_pos < len(audio):
        sample_jump = abs(audio[boundary_pos] - audio[boundary_pos - 1])
        metrics["sample_jump"] = float(sample_jump)

    # 2. Derivative discontinuity (slope change)
    if boundary_pos > 1 and boundary_pos < len(audio) - 1:
        pre_slope = audio[boundary_pos - 1] - audio[boundary_pos - 2]
        post_slope = audio[boundary_pos + 1] - audio[boundary_pos]
        slope_change = abs(post_slope - pre_slope)
        metrics["slope_change"] = float(slope_change)

    # 3. RMS discontinuity
    pre_rms = np.sqrt(np.mean(pre_region ** 2))
    post_rms = np.sqrt(np.mean(post_region ** 2))
    rms_ratio = max(pre_rms, post_rms) / (min(pre_rms, post_rms) + 1e-8)
    metrics["rms_ratio"] = float(rms_ratio)
    metrics["rms_diff_db"] = float(20 * np.log10(rms_ratio + 1e-8))

    # 4. Zero-crossing rate change
    pre_zc = np.sum(np.abs(np.diff(np.sign(pre_region)))) / (2 * len(pre_region))
    post_zc = np.sum(np.abs(np.diff(np.sign(post_region)))) / (2 * len(post_region))
    metrics["zcr_change"] = float(abs(post_zc - pre_zc))

    # 5. Spectral centroid change
    def spectral_centroid(x):
        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
        return np.sum(spectrum * freqs) / (np.sum(spectrum) + 1e-8)

    pre_centroid = spectral_centroid(pre_region)
    post_centroid = spectral_centroid(post_region)
    metrics["centroid_change_hz"] = float(abs(post_centroid - pre_centroid))

    # 6. Local correlation (how well pre/post regions match)
    min_len = min(len(pre_region), len(post_region))
    if min_len > 10:
        # Correlate end of pre with start of post
        corr = np.corrcoef(pre_region[-min_len:], post_region[:min_len])[0, 1]
        metrics["local_correlation"] = float(corr) if not np.isnan(corr) else 0.0

    return metrics


def process_streaming(
    audio: NDArray[np.float32],
    pipeline: RVCPipeline,
    chunk_sec: float = 0.35,
    context_sec: float = 0.1,
    crossfade_sec: float = 0.05,
    mic_sr: int = 48000,
    f0_method: str = "rmvpe",
) -> tuple[NDArray[np.float32], list[int]]:
    """Process audio through streaming pipeline and return output with boundary positions."""

    chunk_samples = int(mic_sr * chunk_sec)
    context_samples = int(mic_sr * context_sec)
    crossfade_samples = int(mic_sr * crossfade_sec)

    buffer = ChunkBuffer(
        chunk_samples=chunk_samples,
        crossfade_samples=0,
        context_samples=context_samples,
        lookahead_samples=0,
    )
    buffer.add_input(audio)

    sola_state = SOLAState.create(
        crossfade_samples,
        mic_sr,
        use_advanced_sola=True,
        fallback_threshold=0.8,
    )

    input_resampler = StatefulResampler(mic_sr, 16000)
    output_resampler = StatefulResampler(pipeline.sample_rate, mic_sr)

    output_chunks = []
    boundary_positions = []
    current_pos = 0
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

        # Record boundary position (after SOLA)
        if chunk_idx > 0:
            boundary_positions.append(current_pos)

        output_chunks.append(cf_result.audio)
        current_pos += len(cf_result.audio)
        chunk_idx += 1

    # Flush SOLA buffer
    remaining = flush_sola_buffer(sola_state)
    if len(remaining) > 0:
        output_chunks.append(remaining)

    streaming_output = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)

    return streaming_output, boundary_positions


def main():
    parser = argparse.ArgumentParser(description="Analyze perceptual discontinuities")
    parser.add_argument("--test-file", type=str, default="sample_data/sustained_voice.wav")
    parser.add_argument("--model", type=str, default="model/kurumi/kurumi.pth")
    parser.add_argument("--max-duration", type=float, default=3.0)
    parser.add_argument("--chunk-sec", type=float, default=0.35)
    parser.add_argument("--context-sec", type=float, default=0.1)
    parser.add_argument("--f0-method", type=str, default="rmvpe")
    parser.add_argument("--save-audio", action="store_true", help="Save output audio for inspection")
    args = parser.parse_args()

    print("=" * 70)
    print("PERCEPTUAL DISCONTINUITY ANALYSIS")
    print("=" * 70)

    # Load test audio
    audio_path = Path(args.test_file)
    if not audio_path.exists():
        print(f"Error: Test file not found: {audio_path}")
        return

    audio, sr = load_audio(str(audio_path))
    max_samples = int(args.max_duration * sr)
    audio = audio[:max_samples]

    print(f"\nTest audio: {len(audio)} samples ({len(audio)/sr:.2f}s)")
    print(f"Chunk: {args.chunk_sec}s, Context: {args.context_sec}s")

    # Load pipeline
    print("\nLoading RVC pipeline...")
    pipeline = RVCPipeline(
        args.model,
        device="auto",
        dtype="float16",
        use_compile=False,  # Avoid torch.compile issues
    )
    pipeline.load()

    # Process through streaming pipeline
    print("\nProcessing through streaming pipeline...")
    streaming_output, boundary_positions = process_streaming(
        audio,
        pipeline,
        chunk_sec=args.chunk_sec,
        context_sec=args.context_sec,
        f0_method=args.f0_method,
    )

    print(f"Output: {len(streaming_output)} samples, {len(boundary_positions)} boundaries")

    if len(boundary_positions) == 0:
        print("No chunk boundaries to analyze (only 1 chunk processed)")
        return

    # Analyze each boundary
    print("\n" + "=" * 70)
    print("BOUNDARY ANALYSIS")
    print("=" * 70)

    all_metrics = []
    for i, pos in enumerate(boundary_positions):
        if pos >= len(streaming_output):
            continue
        metrics = analyze_boundary(streaming_output, pos, window_ms=20.0)
        all_metrics.append(metrics)

        # Print per-boundary metrics
        print(f"\nBoundary {i+1} (sample {pos}, time {pos/sr:.3f}s):")
        if "error" in metrics:
            print(f"  Error: {metrics['error']}")
            continue

        print(f"  Sample jump:      {metrics['sample_jump']:.6f}")
        print(f"  Slope change:     {metrics['slope_change']:.6f}")
        print(f"  RMS ratio:        {metrics['rms_ratio']:.3f} ({metrics['rms_diff_db']:.1f} dB)")
        print(f"  ZCR change:       {metrics['zcr_change']:.4f}")
        print(f"  Centroid change:  {metrics['centroid_change_hz']:.1f} Hz")
        print(f"  Local correlation:{metrics['local_correlation']:.4f}")

    # Global click detection
    print("\n" + "=" * 70)
    print("CLICK DETECTION (entire output)")
    print("=" * 70)

    clicks = detect_clicks(streaming_output, threshold_factor=8.0)

    # Filter clicks near boundaries
    boundary_clicks = []
    non_boundary_clicks = []
    boundary_tolerance = int(sr * 0.005)  # 5ms

    for click_pos, magnitude in clicks:
        near_boundary = any(abs(click_pos - bp) < boundary_tolerance for bp in boundary_positions)
        if near_boundary:
            boundary_clicks.append((click_pos, magnitude))
        else:
            non_boundary_clicks.append((click_pos, magnitude))

    print(f"\nTotal clicks detected: {len(clicks)}")
    print(f"  At boundaries (±5ms): {len(boundary_clicks)}")
    print(f"  Non-boundary clicks:  {len(non_boundary_clicks)}")

    if boundary_clicks:
        print("\nClicks at boundaries:")
        for pos, mag in boundary_clicks[:10]:  # Show first 10
            time_ms = pos / sr * 1000
            print(f"  {time_ms:.1f}ms: magnitude={mag:.6f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    valid_metrics = [m for m in all_metrics if "error" not in m]
    if valid_metrics:
        print("\nMetric                 Mean        Std         Max")
        print("-" * 55)

        for key in ["sample_jump", "slope_change", "rms_ratio", "zcr_change", "centroid_change_hz", "local_correlation"]:
            values = [m[key] for m in valid_metrics if key in m]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                max_val = np.max(values)
                print(f"{key:22s} {mean:10.4f}  {std:10.4f}  {max_val:10.4f}")

    # Quality assessment
    print("\n" + "=" * 70)
    print("QUALITY ASSESSMENT")
    print("=" * 70)

    issues = []

    # Check for high sample jumps
    if valid_metrics:
        max_jump = max(m.get("sample_jump", 0) for m in valid_metrics)
        if max_jump > 0.1:
            issues.append(f"High sample discontinuity: {max_jump:.4f} (threshold: 0.1)")

    # Check for RMS changes
    if valid_metrics:
        max_rms_db = max(m.get("rms_diff_db", 0) for m in valid_metrics)
        if max_rms_db > 3.0:
            issues.append(f"Large volume change: {max_rms_db:.1f} dB (threshold: 3 dB)")

    # Check for low correlation
    if valid_metrics:
        min_corr = min(m.get("local_correlation", 1) for m in valid_metrics)
        if min_corr < 0.5:
            issues.append(f"Low boundary correlation: {min_corr:.4f} (threshold: 0.5)")

    # Check for boundary clicks
    if len(boundary_clicks) > 0:
        issues.append(f"Clicks at {len(boundary_clicks)} boundaries")

    if issues:
        print("\nISSUES DETECTED:")
        for issue in issues:
            print(f"  [!] {issue}")
    else:
        print("\n[OK] No major perceptual issues detected")

    # Save output if requested
    if args.save_audio:
        from scipy.io import wavfile
        output_path = Path("test_output") / "discontinuity_analysis"
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to int16 for WAV
        audio_int16 = (streaming_output * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(output_path / "streaming_output.wav", sr, audio_int16)
        print(f"\nAudio saved to: {output_path / 'streaming_output.wav'}")

        # Also save boundary markers as text
        with open(output_path / "boundaries.txt", "w") as f:
            for i, pos in enumerate(boundary_positions):
                f.write(f"{i+1},{pos},{pos/sr:.6f}\n")
        print(f"Boundaries saved to: {output_path / 'boundaries.txt'}")


if __name__ == "__main__":
    main()
