"""Analyze chunk energy in original audio."""
from scipy.io import wavfile
import numpy as np
from rcwx.audio.resample import resample

sr, audio = wavfile.read("debug_audio/01_input_raw.wav")
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
if audio.ndim > 1:
    audio = audio[:, 0]

# Use first 1.5s
max_samples = int(sr * 1.5)
audio = audio[:max_samples]

# Resample to 48kHz
if sr != 48000:
    audio = resample(audio, sr, 48000)
    sr = 48000

print("="*80)
print("ORIGINAL AUDIO CHUNK ANALYSIS")
print("="*80)
print(f"Length: {len(audio)/sr:.2f}s")

# Analyze chunks (150ms)
chunk_samples = int(sr * 0.15)
num_chunks = len(audio) // chunk_samples

print(f"\nAnalyzing {num_chunks} chunks:")

energies = []
for i in range(num_chunks):
    start = i * chunk_samples
    end = start + chunk_samples
    chunk = audio[start:end]

    rms = np.sqrt(np.mean(chunk**2))
    energies.append(rms)

    print(f"  Chunk {i}: RMS={rms:.6f}")

if len(energies) > 1:
    print(f"\nBoundary changes:")
    for i in range(1, len(energies)):
        prev_rms = energies[i-1]
        curr_rms = energies[i]
        change_pct = (curr_rms - prev_rms) / prev_rms * 100 if prev_rms > 0 else 0

        marker = "!!" if abs(change_pct) > 5.0 else "OK"
        print(f"  {marker} Boundary {i}: {change_pct:+6.1f}%")

    print(f"\nMean RMS: {np.mean(energies):.6f}")
    print(f"Std RMS:  {np.std(energies):.6f}")
    print(f"CV:       {np.std(energies)/np.mean(energies)*100:.2f}%")
    print("="*80)
