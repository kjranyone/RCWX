import numpy as np
from scipy.io import wavfile

sr, audio = wavfile.read('test_output/test_sola_cache_no_smoothing.wav')
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
else:
    audio = audio.astype(np.float32)
if audio.ndim > 1:
    audio = audio[:, 0]

diff = np.abs(np.diff(audio))
disc_count = np.sum(diff > 0.10)

print(f"Discontinuities (>0.10): {disc_count}")
