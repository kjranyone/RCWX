"""Test HuBERT context expansion: compare chunk-to-chunk timbre consistency.

Compares old context (0.56s = 8960 samples) vs new context (1.0s = 16000 samples)
by measuring inter-chunk feature/waveform stability on seki.wav.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = resample(data, sr, target_sr)
    return data


def run_chunked_streaming(
    pipeline, audio_16k: np.ndarray, chunk_sec: float, overlap_sec: float,
    hubert_context_sec: float, noise_scale: float = 0.0,
) -> tuple[list[np.ndarray], list[float]]:
    """Run chunked streaming and return per-chunk outputs + timings."""
    hop = 320
    chunk_samples = (int(16000 * chunk_sec) // hop) * hop
    overlap_samples = (int(16000 * overlap_sec) // hop) * hop

    pipeline.clear_cache()
    chunks_out = []
    timings = []
    pos = 0
    idx = 0

    while pos < len(audio_16k):
        new_hop = audio_16k[pos: pos + chunk_samples]
        if len(new_hop) < hop:
            break
        aligned = (len(new_hop) // hop) * hop
        new_hop = new_hop[:aligned]
        if len(new_hop) == 0:
            break

        if idx == 0:
            if overlap_samples > 0 and len(new_hop) > overlap_samples:
                reflection = new_hop[:overlap_samples][::-1].copy()
                chunk_16k = np.concatenate([reflection, new_hop])
                ovl = overlap_samples
            else:
                chunk_16k = new_hop
                ovl = 0
        else:
            ovl_start = max(0, pos - overlap_samples)
            overlap_audio = audio_16k[ovl_start:pos]
            aligned_ovl = (len(overlap_audio) // hop) * hop
            if aligned_ovl > 0:
                overlap_audio = overlap_audio[-aligned_ovl:]
                chunk_16k = np.concatenate([overlap_audio, new_hop])
                ovl = len(overlap_audio)
            else:
                chunk_16k = new_hop
                ovl = 0

        t0 = time.perf_counter()
        out = pipeline.infer_streaming(
            audio_16k=chunk_16k,
            overlap_samples=ovl,
            pitch_shift=0,
            f0_method="rmvpe",
            noise_scale=noise_scale,
            voice_gate_mode="off",
            use_parallel_extraction=False,
            hubert_context_sec=hubert_context_sec,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        chunks_out.append(out)
        timings.append(elapsed)
        pos += chunk_samples
        idx += 1

    return chunks_out, timings


def spectral_similarity(a: np.ndarray, b: np.ndarray, sr: int, n_fft: int = 1024) -> float:
    """Compute cosine similarity between magnitude spectra of two chunks."""
    def mag_spectrum(x):
        win = np.hanning(min(len(x), n_fft))
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        x = x[:n_fft] * win
        return np.abs(np.fft.rfft(x))

    sa = mag_spectrum(a)
    sb = mag_spectrum(b)
    dot = np.dot(sa, sb)
    norm = (np.linalg.norm(sa) * np.linalg.norm(sb)) + 1e-10
    return float(dot / norm)


def measure_chunk_consistency(chunks: list[np.ndarray], sr: int) -> dict:
    """Measure inter-chunk waveform and spectral consistency."""
    if len(chunks) < 2:
        return {"n_chunks": len(chunks)}

    # Spectral cosine similarity between consecutive chunks
    spec_sims = []
    for i in range(len(chunks) - 1):
        # Use tail of chunk i and head of chunk i+1
        tail = chunks[i][-sr // 10:]  # last 100ms
        head = chunks[i + 1][:sr // 10:]  # first 100ms
        if len(tail) > 0 and len(head) > 0:
            sim = spectral_similarity(tail, head, sr)
            spec_sims.append(sim)

    # Boundary discontinuity (sample-level jump at concatenation point)
    boundary_jumps = []
    for i in range(len(chunks) - 1):
        if len(chunks[i]) > 0 and len(chunks[i + 1]) > 0:
            jump = abs(float(chunks[i][-1]) - float(chunks[i + 1][0]))
            boundary_jumps.append(jump)

    # RMS variation across chunks
    rms_values = [np.sqrt(np.mean(c ** 2)) for c in chunks if len(c) > 0]
    rms_std = np.std(rms_values) if rms_values else 0.0

    return {
        "n_chunks": len(chunks),
        "spectral_sim_mean": np.mean(spec_sims) if spec_sims else 0.0,
        "spectral_sim_min": np.min(spec_sims) if spec_sims else 0.0,
        "spectral_sim_std": np.std(spec_sims) if spec_sims else 0.0,
        "boundary_jump_mean": np.mean(boundary_jumps) if boundary_jumps else 0.0,
        "boundary_jump_max": np.max(boundary_jumps) if boundary_jumps else 0.0,
        "rms_std": rms_std,
    }


def test_hubert_context_quality():
    from rcwx.config import RCWXConfig
    from rcwx.pipeline.inference import RVCPipeline

    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("No model configured. Run GUI first.")
        return

    # Load pipeline
    pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
    pipeline.load()
    model_sr = pipeline.sample_rate
    logger.info(f"Model: {config.last_model_path}")
    logger.info(f"Model sample rate: {model_sr}")

    # Load seki.wav
    test_file = Path(__file__).parent.parent.parent / "sample_data" / "seki.wav"
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    audio_16k = load_test_audio(str(test_file), 16000)
    # Use 5 seconds for meaningful chunk comparison
    use_sec = min(5.0, len(audio_16k) / 16000)
    audio_16k = audio_16k[:int(use_sec * 16000)]
    logger.info(f"Test audio: {test_file.name}, {len(audio_16k)} samples ({len(audio_16k)/16000:.2f}s)")

    chunk_sec = 0.16
    overlap_sec = 0.16

    # === Run with old context (0.56s) ===
    logger.info("\n" + "=" * 60)
    logger.info("=== OLD context: 0.56s (8960 samples) ===")
    logger.info("=" * 60)
    chunks_old, timings_old = run_chunked_streaming(
        pipeline, audio_16k, chunk_sec, overlap_sec,
        hubert_context_sec=0.56,  # old value
        noise_scale=0.0,
    )
    metrics_old = measure_chunk_consistency(chunks_old, model_sr)
    logger.info(f"Chunks: {metrics_old['n_chunks']}")
    logger.info(f"Spectral similarity: mean={metrics_old['spectral_sim_mean']:.4f}, "
                f"min={metrics_old['spectral_sim_min']:.4f}, std={metrics_old['spectral_sim_std']:.4f}")
    logger.info(f"Boundary jumps: mean={metrics_old['boundary_jump_mean']:.4f}, "
                f"max={metrics_old['boundary_jump_max']:.4f}")
    logger.info(f"RMS std: {metrics_old['rms_std']:.4f}")
    logger.info(f"Inference time: mean={np.mean(timings_old):.1f}ms, "
                f"max={np.max(timings_old):.1f}ms")

    # === Run with new context (1.0s) ===
    logger.info("\n" + "=" * 60)
    logger.info("=== NEW context: 1.0s (16000 samples) ===")
    logger.info("=" * 60)
    chunks_new, timings_new = run_chunked_streaming(
        pipeline, audio_16k, chunk_sec, overlap_sec,
        hubert_context_sec=1.0,  # new value
        noise_scale=0.0,
    )
    metrics_new = measure_chunk_consistency(chunks_new, model_sr)
    logger.info(f"Chunks: {metrics_new['n_chunks']}")
    logger.info(f"Spectral similarity: mean={metrics_new['spectral_sim_mean']:.4f}, "
                f"min={metrics_new['spectral_sim_min']:.4f}, std={metrics_new['spectral_sim_std']:.4f}")
    logger.info(f"Boundary jumps: mean={metrics_new['boundary_jump_mean']:.4f}, "
                f"max={metrics_new['boundary_jump_max']:.4f}")
    logger.info(f"RMS std: {metrics_new['rms_std']:.4f}")
    logger.info(f"Inference time: mean={np.mean(timings_new):.1f}ms, "
                f"max={np.max(timings_new):.1f}ms")

    # === Comparison ===
    logger.info("\n" + "=" * 60)
    logger.info("=== COMPARISON ===")
    logger.info("=" * 60)

    spec_delta = metrics_new["spectral_sim_mean"] - metrics_old["spectral_sim_mean"]
    spec_std_delta = metrics_old["spectral_sim_std"] - metrics_new["spectral_sim_std"]
    rms_delta = metrics_old["rms_std"] - metrics_new["rms_std"]
    time_delta = np.mean(timings_new) - np.mean(timings_old)

    logger.info(f"Spectral sim mean: {metrics_old['spectral_sim_mean']:.4f} -> "
                f"{metrics_new['spectral_sim_mean']:.4f} ({spec_delta:+.4f})")
    logger.info(f"Spectral sim std:  {metrics_old['spectral_sim_std']:.4f} -> "
                f"{metrics_new['spectral_sim_std']:.4f} ({spec_std_delta:+.4f} improvement)")
    logger.info(f"RMS std:           {metrics_old['rms_std']:.4f} -> "
                f"{metrics_new['rms_std']:.4f} ({rms_delta:+.4f} improvement)")
    logger.info(f"Inference time:    {np.mean(timings_old):.1f}ms -> "
                f"{np.mean(timings_new):.1f}ms ({time_delta:+.1f}ms)")

    # Save outputs for listening
    out_dir = Path("test_output/hubert_context_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    if chunks_old:
        concat_old = np.concatenate(chunks_old)
        wavfile.write(str(out_dir / "seki_context_056s.wav"), model_sr, concat_old)
    if chunks_new:
        concat_new = np.concatenate(chunks_new)
        wavfile.write(str(out_dir / "seki_context_100s.wav"), model_sr, concat_new)
    logger.info(f"\nOutputs saved to {out_dir}/ for listening comparison")

    # Assertions
    assert metrics_new["spectral_sim_mean"] >= metrics_old["spectral_sim_mean"] - 0.02, \
        "New context should not significantly degrade spectral consistency"
    assert np.mean(timings_new) < 150, \
        f"Inference too slow: {np.mean(timings_new):.1f}ms > 150ms budget"
    logger.info("\nPASS: All assertions passed")


if __name__ == "__main__":
    test_hubert_context_quality()
