"""Test both channels to compare audio quality."""
import sounddevice as sd
import numpy as np
from scipy.io import wavfile

# Settings
device_name = "Analog 1/2 (E2x2 OTG)"
duration = 3.0
sample_rate = 48000

# Find device
devices = sd.query_devices()
device_idx = None
for i, d in enumerate(devices):
    if device_name in d['name'] and d['max_input_channels'] >= 2:
        device_idx = i
        break

if device_idx is None:
    print(f"Device '{device_name}' not found")
    exit(1)

print(f"Recording from: {devices[device_idx]['name']}")
print(f"Sample rate: {sample_rate}Hz, Duration: {duration}s")
print("Recording in 3... 2... 1...")

# Record stereo
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=2,
    dtype=np.float32,
    device=device_idx,
)
sd.wait()

# Extract channels
left = audio[:, 0]
right = audio[:, 1]
average = np.mean(audio, axis=1)

# Save
wavfile.write("test_left.wav", sample_rate, left)
wavfile.write("test_right.wav", sample_rate, right)
wavfile.write("test_average.wav", sample_rate, average)

# Analysis
print("\n=== Channel Analysis ===")
print(f"Left   - Min: {left.min():.4f}, Max: {left.max():.4f}, RMS: {np.sqrt(np.mean(left**2)):.4f}")
print(f"Right  - Min: {right.min():.4f}, Max: {right.max():.4f}, RMS: {np.sqrt(np.mean(right**2)):.4f}")
print(f"Average- Min: {average.min():.4f}, Max: {average.max():.4f}, RMS: {np.sqrt(np.mean(average**2)):.4f}")
print("\nFiles saved:")
print("  - test_left.wav")
print("  - test_right.wav")
print("  - test_average.wav")
print("\nListen to each file to determine which channel has better quality.")
