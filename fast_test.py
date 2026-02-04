"""Ultra-fast test - just count discontinuities in existing output."""
import numpy as np
from scipy.io import wavfile

print("Checking existing output file...")
sr, audio = wavfile.read('test_output/test_sola_cache_no_smoothing.wav')
audio = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
if audio.ndim > 1:
    audio = audio[:, 0]

diff = np.abs(np.diff(audio))
count = int(np.sum(diff > 0.10))

print(f"Result: {count} discontinuities (>0.10)")
print(f"Previous: 115 discontinuities")

if count < 115:
    print(f"SUCCESS! Reduced by {115 - count} ({(115-count)/115*100:.1f}%)")
elif count == 115:
    print("No change - fix did not work")
else:
    print(f"WORSE! Increased by {count - 115}")
