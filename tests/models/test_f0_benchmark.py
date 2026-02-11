"""F0 method benchmark: RMVPE vs FCPE vs SwiftF0.

Compares speed, F0 correlation, voicing agreement, and streaming performance.
Uses RMVPE as the reference (ground truth proxy).

Usage:
    uv run python tests/models/test_f0_benchmark.py [--audio sample_data/asano.wav]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rcwx.audio.resample import resample
from rcwx.device import get_device, get_dtype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_audio_16k(path: str) -> np.ndarray:
    """Load audio file and resample to 16kHz mono float32."""
    import scipy.io.wavfile
    sr, data = scipy.io.wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        data = resample(data, sr, 16000)
    return data.astype(np.float32)


def cents_diff(f0_a: np.ndarray, f0_b: np.ndarray) -> np.ndarray:
    """Per-frame absolute difference in cents (only where both voiced)."""
    both_voiced = (f0_a > 0) & (f0_b > 0)
    if not np.any(both_voiced):
        return np.array([])
    a = f0_a[both_voiced]
    b = f0_b[both_voiced]
    return np.abs(1200.0 * np.log2(np.maximum(a, 1e-6) / np.maximum(b, 1e-6)))


def correlation(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """Pearson correlation between two F0 arrays (voiced frames only)."""
    both_voiced = (f0_a > 0) & (f0_b > 0)
    if np.sum(both_voiced) < 10:
        return 0.0
    a = f0_a[both_voiced]
    b = f0_b[both_voiced]
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def voicing_agreement(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """Fraction of frames where voicing decision agrees."""
    v_a = f0_a > 0
    v_b = f0_b > 0
    return float(np.mean(v_a == v_b))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(audio_path: str, n_warmup: int = 2, n_runs: int = 5):
    device = get_device("auto")
    dtype = get_dtype(device, "float16")

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Audio:  {audio_path}")
    print()

    # Load audio
    audio_np = load_audio_16k(audio_path)
    duration = len(audio_np) / 16000
    print(f"Duration: {duration:.2f}s ({len(audio_np)} samples @ 16kHz)")
    print()

    audio_t = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    from rcwx.models.rmvpe import RMVPE
    from rcwx.downloader import get_rmvpe_path

    models_dir = Path.home() / ".cache" / "rcwx" / "models"
    methods: dict[str, object] = {}
    thresholds: dict[str, float] = {}

    # RMVPE (reference)
    rmvpe_path = get_rmvpe_path(models_dir)
    if rmvpe_path.exists():
        rmvpe = RMVPE(str(rmvpe_path), device=device, dtype=dtype)
        methods["RMVPE"] = rmvpe
        thresholds["RMVPE"] = 0.015
        print("[OK] RMVPE loaded")
    else:
        print("[SKIP] RMVPE model not found")

    # FCPE
    try:
        from rcwx.models.fcpe import FCPE, is_fcpe_available
        if is_fcpe_available():
            fcpe = FCPE(device=device, dtype=dtype)
            methods["FCPE"] = fcpe
            thresholds["FCPE"] = 0.006
            print("[OK] FCPE loaded")
        else:
            print("[SKIP] FCPE not available")
    except Exception as e:
        print(f"[SKIP] FCPE: {e}")

    # SwiftF0
    try:
        from rcwx.models.swiftf0 import SwiftF0, is_swiftf0_available
        if is_swiftf0_available():
            swiftf0 = SwiftF0(confidence_threshold=0.5)
            methods["SwiftF0"] = swiftf0
            thresholds["SwiftF0"] = 0.5
            print("[OK] SwiftF0 loaded")
        else:
            print("[SKIP] SwiftF0 not available (pip install swift-f0)")
    except Exception as e:
        print(f"[SKIP] SwiftF0: {e}")

    if len(methods) < 2:
        print("\nNeed at least 2 F0 methods for comparison. Exiting.")
        return

    print(f"\nMethods: {list(methods.keys())}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1) Full-audio inference (batch)
    # -----------------------------------------------------------------------
    print("\n## 1. Batch inference (full audio)")
    print("-" * 70)

    f0_results: dict[str, np.ndarray] = {}
    time_results: dict[str, list[float]] = {}

    for name, model in methods.items():
        times = []
        threshold = thresholds[name]

        for i in range(n_warmup + n_runs):
            if name in ("RMVPE", "FCPE"):
                with torch.autocast(device_type=device, dtype=dtype):
                    t0 = time.perf_counter()
                    f0 = model.infer(audio_t, threshold=threshold)
                    t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                f0 = model.infer(audio_t, threshold=threshold)
                t1 = time.perf_counter()

            if i >= n_warmup:
                times.append(t1 - t0)

        f0_np = f0[0].cpu().float().numpy()
        f0_results[name] = f0_np
        time_results[name] = times

        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        rtf = np.mean(times) / duration
        voiced_pct = 100.0 * np.mean(f0_np > 0)
        print(
            f"  {name:10s}: {avg_ms:7.1f} +/- {std_ms:4.1f} ms  "
            f"(RTF={rtf:.4f})  "
            f"frames={len(f0_np):5d}  voiced={voiced_pct:.1f}%"
        )

    # -----------------------------------------------------------------------
    # 2) Accuracy comparison (RMVPE as reference)
    # -----------------------------------------------------------------------
    ref_name = "RMVPE" if "RMVPE" in f0_results else list(f0_results.keys())[0]
    ref_f0 = f0_results[ref_name]

    print(f"\n## 2. Accuracy vs {ref_name}")
    print("-" * 70)
    print(f"  {'Method':10s}  {'Corr':>8s}  {'Voicing%':>8s}  "
          f"{'Med(cents)':>10s}  {'P75(cents)':>10s}  {'P95(cents)':>10s}  "
          f"{'Gross(>50c)':>11s}")

    for name, f0_np in f0_results.items():
        # Align lengths (interpolate shorter to match ref)
        if len(f0_np) != len(ref_f0):
            f0_t = torch.from_numpy(f0_np).float().unsqueeze(0).unsqueeze(0)
            f0_aligned = torch.nn.functional.interpolate(
                f0_t, size=len(ref_f0), mode="linear", align_corners=False
            ).squeeze().numpy()
            # Re-apply voicing mask via nearest interp
            voiced_t = torch.from_numpy(
                (f0_np > 0).astype(np.float32)
            ).unsqueeze(0).unsqueeze(0)
            voiced_aligned = torch.nn.functional.interpolate(
                voiced_t, size=len(ref_f0), mode="nearest"
            ).squeeze().numpy()
            f0_aligned[voiced_aligned < 0.5] = 0.0
        else:
            f0_aligned = f0_np

        corr = correlation(ref_f0, f0_aligned)
        va = voicing_agreement(ref_f0, f0_aligned)
        cd = cents_diff(ref_f0, f0_aligned)

        if len(cd) > 0:
            med = np.median(cd)
            p75 = np.percentile(cd, 75)
            p95 = np.percentile(cd, 95)
            gross = 100.0 * np.mean(cd > 50)
        else:
            med = p75 = p95 = gross = float("nan")

        marker = " (ref)" if name == ref_name else ""
        print(
            f"  {name:10s}  {corr:8.4f}  {va*100:7.1f}%  "
            f"{med:10.1f}  {p75:10.1f}  {p95:10.1f}  "
            f"{gross:10.1f}%{marker}"
        )

    # -----------------------------------------------------------------------
    # 3) Streaming simulation (chunk-by-chunk)
    # -----------------------------------------------------------------------
    chunk_sec = 0.16  # 160ms chunks (typical realtime setting)
    chunk_samples = int(chunk_sec * 16000)
    # Round to SwiftF0 hop (256) for fair comparison
    chunk_samples = (chunk_samples // 256) * 256
    n_chunks = len(audio_np) // chunk_samples

    print(f"\n## 3. Streaming simulation (chunk={chunk_sec*1000:.0f}ms, {n_chunks} chunks)")
    print("-" * 70)

    for name, model in methods.items():
        threshold = thresholds[name]
        chunk_times = []
        errors = 0

        for c in range(n_chunks):
            start = c * chunk_samples
            end = start + chunk_samples
            chunk_audio = audio_np[start:end]
            chunk_t = torch.from_numpy(chunk_audio).float().unsqueeze(0).to(device)

            try:
                if name in ("RMVPE", "FCPE"):
                    with torch.autocast(device_type=device, dtype=dtype):
                        t0 = time.perf_counter()
                        _ = model.infer(chunk_t, threshold=threshold)
                        t1 = time.perf_counter()
                else:
                    t0 = time.perf_counter()
                    _ = model.infer(chunk_t, threshold=threshold)
                    t1 = time.perf_counter()
                chunk_times.append((t1 - t0) * 1000)
            except Exception:
                errors += 1

        if not chunk_times:
            print(
                f"  {name:10s}: FAILED ({errors}/{n_chunks} chunks errored, "
                f"input too short for this method)"
            )
            continue

        avg = np.mean(chunk_times)
        p50 = np.median(chunk_times)
        p95 = np.percentile(chunk_times, 95)
        p99 = np.percentile(chunk_times, 99)
        budget = chunk_sec * 1000
        over_budget = 100.0 * np.mean(np.array(chunk_times) > budget)

        err_msg = f"  ({errors} errors)" if errors else ""
        print(
            f"  {name:10s}: avg={avg:6.1f}ms  "
            f"p50={p50:5.1f}ms  p95={p95:5.1f}ms  p99={p99:5.1f}ms  "
            f"over_budget={over_budget:.1f}%{err_msg}"
        )

    # -----------------------------------------------------------------------
    # 4) Summary
    # -----------------------------------------------------------------------
    print(f"\n## 4. Summary")
    print("=" * 70)

    if "RMVPE" in time_results and "SwiftF0" in time_results:
        rmvpe_avg = np.mean(time_results["RMVPE"]) * 1000
        swift_avg = np.mean(time_results["SwiftF0"]) * 1000
        print(f"  SwiftF0 is {rmvpe_avg / swift_avg:.1f}x faster than RMVPE (batch)")

    if "FCPE" in time_results and "SwiftF0" in time_results:
        fcpe_avg = np.mean(time_results["FCPE"]) * 1000
        swift_avg = np.mean(time_results["SwiftF0"]) * 1000
        if swift_avg < fcpe_avg:
            print(f"  SwiftF0 is {fcpe_avg / swift_avg:.1f}x faster than FCPE (batch)")
        else:
            print(f"  FCPE is {swift_avg / fcpe_avg:.1f}x faster than SwiftF0 (batch)")

    if "SwiftF0" in f0_results and ref_name in f0_results:
        sf0 = f0_results["SwiftF0"]
        rf0 = f0_results[ref_name]
        if len(sf0) != len(rf0):
            sf0_t = torch.from_numpy(sf0).float().unsqueeze(0).unsqueeze(0)
            sf0 = torch.nn.functional.interpolate(
                sf0_t, size=len(rf0), mode="linear", align_corners=False
            ).squeeze().numpy()
        corr = correlation(rf0, sf0)
        cd = cents_diff(rf0, sf0)
        if len(cd) > 0:
            print(f"  SwiftF0 vs {ref_name}: corr={corr:.4f}, median_error={np.median(cd):.1f} cents")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F0 method benchmark")
    parser.add_argument(
        "--audio", "-a",
        default="sample_data/asano.wav",
        help="Audio file for benchmark",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    run_benchmark(args.audio, n_warmup=args.warmup, n_runs=args.runs)
