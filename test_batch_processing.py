"""Test batch processing (no chunking) for discontinuities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline

print("="*80)
print("BATCH PROCESSING TEST (No Chunking)")
print("="*80)

# Load audio
sr, audio = wavfile.read("debug_audio/01_input_raw.wav")
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
if audio.ndim > 1:
    audio = audio[:, 0]

# Use first 1.5s
max_samples = int(sr * 1.5)
audio = audio[:max_samples]

print(f"\nInput: {len(audio)/sr:.2f}s @ {sr}Hz")

# Load model
print("Loading model...")
pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
pipeline.load()

# Batch inference (single pass, no chunking)
print("Processing (batch mode - no chunking)...")
output = pipeline.infer(
    audio,
    input_sr=sr,
    f0_method="fcpe",
    pitch_shift=0,
    index_rate=0.0,
)

output_sr = 40000  # RVC output rate
print(f"Output: {len(output)/output_sr:.2f}s @ {output_sr}Hz")

# Save
output_path = "test_output/batch_output.wav"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
output_int16 = (output * 32767).astype(np.int16)
wavfile.write(output_path, output_sr, output_int16)
print(f"Saved: {output_path}")

# Analyze discontinuities
diff = np.abs(np.diff(output))
disc_010 = len(np.where(diff > 0.10)[0])
disc_015 = len(np.where(diff > 0.15)[0])

print(f"\n" + "="*80)
print("DISCONTINUITY ANALYSIS")
print("="*80)
print(f"Discontinuities (>0.10): {disc_010}")
print(f"Discontinuities (>0.15): {disc_015}")

if disc_010 == 0:
    print("\n[SUCCESS] No discontinuities in batch processing!")
    print("Issue is specific to realtime chunking.")
else:
    print(f"\n[ISSUE] Discontinuities exist even in batch processing.")
    print("Issue is in RVC inference itself, not chunking.")

print("="*80)
