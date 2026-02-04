"""Analyze chunk boundary consistency in output."""
from scipy.io import wavfile
import numpy as np

sr, audio = wavfile.read("test_output/realtime_volume_output.wav")
audio_f = audio.astype(np.float32) / 32768.0

# Find active region
nonzero_idx = np.where(audio != 0)[0]
if len(nonzero_idx) == 0:
    print("No active audio found!")
    exit(1)

active_start = nonzero_idx[0]
active_end = nonzero_idx[-1]
active_audio = audio_f[active_start:active_end+1]

print("="*80)
print("CHUNK BOUNDARY CONSISTENCY ANALYSIS")
print("="*80)
print(f"\nActive region: {active_start/sr:.2f}s ~ {active_end/sr:.2f}s")
print(f"Length: {len(active_audio)/sr:.3f}s")

# Analyze chunks (chunk_sec = 0.15s = 7200 samples @ 48kHz)
chunk_samples = int(sr * 0.15)
num_chunks = len(active_audio) // chunk_samples

print(f"\nAnalyzing {num_chunks} chunks (150ms each):")

energies = []
for i in range(num_chunks):
    start = i * chunk_samples
    end = start + chunk_samples
    chunk = active_audio[start:end]

    rms = np.sqrt(np.mean(chunk**2))
    energies.append(rms)

    print(f"  Chunk {i}: RMS={rms:.6f}")

if len(energies) > 1:
    print(f"\n" + "="*80)
    print("BOUNDARY ENERGY CHANGES")
    print("="*80)

    significant_drops = 0
    for i in range(1, len(energies)):
        prev_rms = energies[i-1]
        curr_rms = energies[i]
        change_pct = (curr_rms - prev_rms) / prev_rms * 100 if prev_rms > 0 else 0

        marker = "!!" if abs(change_pct) > 5.0 else "OK"
        print(f"  {marker} Boundary {i}: {change_pct:+6.1f}% ({prev_rms:.6f} -> {curr_rms:.6f})")

        if abs(change_pct) > 5.0:
            significant_drops += 1

    print(f"\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    mean_rms = np.mean(energies)
    std_rms = np.std(energies)
    cv = (std_rms / mean_rms * 100) if mean_rms > 0 else 0

    print(f"\nMean RMS: {mean_rms:.6f}")
    print(f"Std RMS:  {std_rms:.6f}")
    print(f"CV:       {cv:.2f}%")
    print(f"Significant changes (>5%): {significant_drops}/{len(energies)-1}")

    print(f"\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if significant_drops == 0:
        print("[SUCCESS] No significant energy drops at chunk boundaries!")
        print("Chunk-to-chunk consistency is excellent.")
    elif significant_drops <= (len(energies)-1) * 0.2:
        print("[OK] Minor energy variations at some boundaries.")
        print(f"{significant_drops}/{len(energies)-1} boundaries affected.")
    else:
        print("[ISSUE] Significant energy variations at chunk boundaries.")
        print(f"{significant_drops}/{len(energies)-1} boundaries affected.")

    print("="*80)
