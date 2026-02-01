"""Compare output sample rates between batch and streaming."""

import numpy as np
from scipy.io import wavfile

# Load both outputs
sr, batch = wavfile.read("test_output/debug_batch.wav")
sr, streaming = wavfile.read("test_output/debug_streaming.wav")

# Convert to float
if batch.dtype == np.int16:
    batch = batch.astype(np.float32) / 32768.0
if streaming.dtype == np.int16:
    streaming = streaming.astype(np.float32) / 32768.0

if len(batch.shape) > 1:
    batch = batch[:, 0]
if len(streaming.shape) > 1:
    streaming = streaming[:, 0]

print(f"Batch: {len(batch)} samples @ 48kHz = {len(batch) / sr:.2f}s")
print(f"Streaming: {len(streaming)} samples @ 48kHz = {len(streaming) / sr:.2f}s")

# Check if they're phase-shifted
min_len = min(len(batch), len(streaming))
batch_trim = batch[:min_len]
streaming_trim = streaming[:min_len]

correlation = np.corrcoef(batch_trim, streaming_trim)[0, 1]
print(f"Correlation: {correlation:.6f}")

# Check if one is phase-inverted
batch_shifted = batch_trim * -1
correlation_inverted = np.corrcoef(batch_shifted, streaming_trim)[0, 1]
print(f"Correlation (batch inverted vs streaming): {correlation_inverted:.6f}")

if abs(correlation_inverted) > abs(correlation):
    print("WARNING: Phase inversion detected!")
