"""Compare active regions of input vs output."""
from scipy.io import wavfile
import numpy as np
from rcwx.audio.resample import resample

# Load original
sr_orig, audio_orig = wavfile.read("debug_audio/01_input_raw.wav")
if audio_orig.dtype == np.int16:
    audio_orig = audio_orig.astype(np.float32) / 32768.0
if audio_orig.ndim > 1:
    audio_orig = audio_orig[:, 0]

# Use first 1.5s
max_samples = int(sr_orig * 1.5)
audio_orig = audio_orig[:max_samples]

# Resample to 48kHz
if sr_orig != 48000:
    audio_orig = resample(audio_orig, sr_orig, 48000)

print("Original audio:")
print(f"  Length: {len(audio_orig)/48000:.2f}s")

# Analyze 30-70% range
start_idx = int(len(audio_orig) * 0.3)
end_idx = int(len(audio_orig) * 0.7)
orig_region = audio_orig[start_idx:end_idx]
orig_rms = np.sqrt(np.mean(orig_region**2))

print(f"  30-70% range: {start_idx/48000:.2f}s ~ {end_idx/48000:.2f}s")
print(f"  RMS: {orig_rms:.6f}")
print(f"  Min: {orig_region.min():+.6f}")
print(f"  Max: {orig_region.max():+.6f}")

# Load processed
sr_proc, audio_proc = wavfile.read("test_output/realtime_volume_output.wav")
audio_proc_f = audio_proc.astype(np.float32) / 32768.0

print(f"\nProcessed audio:")
print(f"  Length: {len(audio_proc)/sr_proc:.2f}s")

# Find active region
nonzero_idx = np.where(audio_proc != 0)[0]
if len(nonzero_idx) > 0:
    active_start = nonzero_idx[0]
    active_end = nonzero_idx[-1]
    print(f"  Active region: {active_start/sr_proc:.2f}s ~ {active_end/sr_proc:.2f}s")

    # Get 30-70% of active region
    active_len = active_end - active_start
    proc_start = active_start + int(active_len * 0.3)
    proc_end = active_start + int(active_len * 0.7)

    proc_region = audio_proc_f[proc_start:proc_end]
    proc_rms = np.sqrt(np.mean(proc_region**2))

    print(f"  30-70% of active: {proc_start/sr_proc:.2f}s ~ {proc_end/sr_proc:.2f}s")
    print(f"  RMS: {proc_rms:.6f}")
    print(f"  Min: {proc_region.min():+.6f}")
    print(f"  Max: {proc_region.max():+.6f}")

    # Compare
    print(f"\n" + "="*80)
    print("COMPARISON (30-70% ranges)")
    print("="*80)

    rms_ratio = proc_rms / orig_rms if orig_rms > 0 else 0
    rms_db = 20 * np.log10(rms_ratio) if rms_ratio > 0 else -np.inf

    print(f"\nRMS: {proc_rms:.6f} vs {orig_rms:.6f}")
    print(f"Ratio: {rms_ratio:.1%} ({rms_db:+.1f}dB)")

    if rms_ratio >= 0.95:
        verdict = "SUCCESS - Volume preserved (>=95%)"
    elif rms_ratio >= 0.85:
        verdict = "OK - Minor reduction (85-95%)"
    elif rms_ratio >= 0.70:
        verdict = "WARNING - Noticeable reduction (70-85%)"
    else:
        verdict = "ISSUE - Significant loss (<70%)"

    print(f"\nVERDICT: {verdict}")
    print("="*80)
