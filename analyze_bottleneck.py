"""Analyze which component causes processing time variance."""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))

from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load pipeline
config = RCWXConfig.load()
pipeline = RVCPipeline(config.last_model_path, device=config.device, use_compile=False)
pipeline.load()

# Load test audio (10 seconds)
test_file = Path("sample_data/seki.wav")
sr, audio_raw = wavfile.read(test_file)
if audio_raw.dtype == np.int16:
    audio_raw = audio_raw.astype(np.float32) / 32768.0
if len(audio_raw.shape) > 1:
    audio_raw = audio_raw[:, 0]

audio = resample(audio_raw, sr, 48000)[:int(48000 * 10)]

# Split into 96ms chunks
chunk_sec = 0.096
chunk_samples = int(48000 * chunk_sec)
num_chunks = len(audio) // chunk_samples

logger.info(f"Analyzing {num_chunks} chunks of {chunk_sec*1000}ms")

# Component timings
timings = {
    "resample_48_16": [],
    "hubert": [],
    "f0": [],
    "synthesizer": [],
    "resample_40_48": [],
    "total": [],
}

chunk_info = []

for i in range(min(num_chunks, 100)):  # First 100 chunks
    chunk_48k = audio[i*chunk_samples:(i+1)*chunk_samples]

    t_total = time.perf_counter()

    # 1. Resample 48k -> 16k
    t0 = time.perf_counter()
    chunk_16k = resample(chunk_48k, 48000, 16000)
    t_resample_in = (time.perf_counter() - t0) * 1000

    # 2. HuBERT
    t0 = time.perf_counter()
    with torch.no_grad():
        audio_tensor = torch.from_numpy(chunk_16k).to(pipeline.device, dtype=pipeline.dtype)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Pad if needed (HuBERT requirement)
        if audio_tensor.shape[1] < 1600:
            padding = 1600 - audio_tensor.shape[1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

        output_layer = 12 if pipeline.synthesizer.version == 2 else 9
        output_dim = 768 if pipeline.synthesizer.version == 2 else 256

        features = pipeline.hubert.extract(audio_tensor, output_layer=output_layer, output_dim=output_dim)
    t_hubert = (time.perf_counter() - t0) * 1000

    # 3. F0 extraction
    t0 = time.perf_counter()
    with torch.no_grad():
        f0 = pipeline.fcpe.infer(audio_tensor, threshold=0.006, f0_min=50.0, f0_max=1100.0)
    t_f0 = (time.perf_counter() - t0) * 1000

    # Count voiced frames
    voiced_frames = torch.sum(f0 > 0).item()
    total_frames = f0.shape[1]
    voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0

    # 4. Synthesizer (simplified - just measure forward pass)
    t0 = time.perf_counter()
    with torch.no_grad():
        # Interpolate features to 100fps
        features_interp = torch.nn.functional.interpolate(
            features.transpose(1, 2),
            scale_factor=2,
            mode='linear',
            align_corners=True
        ).transpose(1, 2)

        # Prepare F0 (quantize to mel scale)
        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        f0_mel = f0_mel.clamp(0, 255).long()

        # Dummy synthesizer call (just to measure time)
        _ = pipeline.synthesizer.model(features_interp, f0_mel.unsqueeze(0))
    t_synth = (time.perf_counter() - t0) * 1000

    # 5. Resample 40k -> 48k (approximate)
    t_resample_out = 0.3  # Typically very fast

    t_total = (time.perf_counter() - t_total) * 1000

    timings["resample_48_16"].append(t_resample_in)
    timings["hubert"].append(t_hubert)
    timings["f0"].append(t_f0)
    timings["synthesizer"].append(t_synth)
    timings["resample_40_48"].append(t_resample_out)
    timings["total"].append(t_total)

    chunk_info.append({
        "chunk": i,
        "voiced_ratio": voiced_ratio,
        "total_time": t_total,
        "hubert": t_hubert,
        "f0": t_f0,
        "synth": t_synth,
    })

    if i % 20 == 0:
        logger.info(f"Chunk {i}: total={t_total:.1f}ms, hubert={t_hubert:.1f}ms, f0={t_f0:.1f}ms, synth={t_synth:.1f}ms, voiced={voiced_ratio:.1%}")

# Analysis
logger.info("\n" + "="*70)
logger.info("COMPONENT TIMING ANALYSIS")
logger.info("="*70)

for comp, times in timings.items():
    if not times:
        continue

    times_np = np.array(times)
    avg = np.mean(times_np)
    median = np.median(times_np)
    p95 = np.percentile(times_np, 95)
    p99 = np.percentile(times_np, 99)
    max_t = np.max(times_np)
    std = np.std(times_np)

    logger.info(f"\n{comp:15s}:")
    logger.info(f"  Avg:    {avg:6.1f}ms")
    logger.info(f"  Median: {median:6.1f}ms")
    logger.info(f"  P95:    {p95:6.1f}ms")
    logger.info(f"  P99:    {p99:6.1f}ms")
    logger.info(f"  Max:    {max_t:6.1f}ms")
    logger.info(f"  Std:    {std:6.1f}ms")
    logger.info(f"  Variance: {(p99/avg - 1)*100:5.1f}%")

# Find slowest chunks
logger.info("\n" + "="*70)
logger.info("SLOWEST CHUNKS ANALYSIS")
logger.info("="*70)

sorted_chunks = sorted(chunk_info, key=lambda x: x["total_time"], reverse=True)

logger.info(f"\n{'Chunk':<8} {'Total':<8} {'HuBERT':<8} {'F0':<8} {'Synth':<8} {'Voiced%':<10}")
logger.info("-" * 70)

for chunk in sorted_chunks[:20]:  # Top 20 slowest
    logger.info(
        f"{chunk['chunk']:<8} {chunk['total_time']:<8.1f} {chunk['hubert']:<8.1f} "
        f"{chunk['f0']:<8.1f} {chunk['synth']:<8.1f} {chunk['voiced_ratio']*100:<10.1f}"
    )

# Correlation analysis
logger.info("\n" + "="*70)
logger.info("CORRELATION ANALYSIS")
logger.info("="*70)

voiced_ratios = np.array([c["voiced_ratio"] for c in chunk_info])
total_times = np.array([c["total_time"] for c in chunk_info])
hubert_times = np.array([c["hubert"] for c in chunk_info])
f0_times = np.array([c["f0"] for c in chunk_info])
synth_times = np.array([c["synth"] for c in chunk_info])

logger.info(f"Correlation: Voiced% vs Total time:  {np.corrcoef(voiced_ratios, total_times)[0,1]:+.3f}")
logger.info(f"Correlation: Voiced% vs HuBERT time: {np.corrcoef(voiced_ratios, hubert_times)[0,1]:+.3f}")
logger.info(f"Correlation: Voiced% vs F0 time:     {np.corrcoef(voiced_ratios, f0_times)[0,1]:+.3f}")
logger.info(f"Correlation: Voiced% vs Synth time:  {np.corrcoef(voiced_ratios, synth_times)[0,1]:+.3f}")

# Recommendations
logger.info("\n" + "="*70)
logger.info("OPTIMIZATION RECOMMENDATIONS")
logger.info("="*70)

# Identify biggest bottleneck
avg_times = {k: np.mean(v) for k, v in timings.items() if k != "total"}
max_component = max(avg_times, key=avg_times.get)
logger.info(f"\n1. PRIMARY BOTTLENECK: {max_component} ({avg_times[max_component]:.1f}ms avg)")

# Identify highest variance
variances = {}
for comp, times in timings.items():
    if comp == "total" or not times:
        continue
    times_np = np.array(times)
    p99 = np.percentile(times_np, 99)
    avg = np.mean(times_np)
    variances[comp] = (p99 / avg - 1) * 100

max_variance_comp = max(variances, key=variances.get)
logger.info(f"\n2. HIGHEST VARIANCE: {max_variance_comp} (+{variances[max_variance_comp]:.0f}% at P99)")

logger.info("\n3. OPTIMIZATION STRATEGIES:")
if max_component == "hubert":
    logger.info("   - Use Intel Extension for PyTorch (IPEX) for XPU optimization")
    logger.info("   - Reduce HuBERT layer depth if possible")
    logger.info("   - Enable IPEX's graph optimization")
elif max_component == "f0":
    logger.info("   - Adjust FCPE threshold for faster processing")
    logger.info("   - Consider hybrid approach (FCPE + cache)")
elif max_component == "synthesizer":
    logger.info("   - Enable IPEX optimization for synthesizer")
    logger.info("   - Check if model can use mixed precision")

logger.info("\n4. VARIANCE REDUCTION:")
logger.info("   - Add processing time cap/timeout")
logger.info("   - Use adaptive prebuffering based on recent P95")
logger.info("   - Implement chunk priority queue")
