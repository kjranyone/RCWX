"""Analyze output file."""
from scipy.io import wavfile
import numpy as np

sr, audio = wavfile.read('test_output/realtime_volume_output.wav')
audio_f = audio.astype(np.float32)/32768

nonzero_idx = np.where(audio != 0)[0]
print(f"Total samples: {len(audio)}")
print(f"Non-zero samples: {len(nonzero_idx)} ({len(nonzero_idx)/len(audio)*100:.1f}%)")

if len(nonzero_idx) > 0:
    print(f"\nFirst nonzero at: {nonzero_idx[0]} ({nonzero_idx[0]/sr:.2f}s)")
    print(f"Last nonzero at: {nonzero_idx[-1]} ({nonzero_idx[-1]/sr:.2f}s)")

    region = audio_f[nonzero_idx[0]:nonzero_idx[-1]+1]
    rms = np.sqrt(np.mean(region**2))
    print(f"\nActive region:")
    print(f"  Length: {len(region)/sr:.2f}s")
    print(f"  RMS: {rms:.6f}")
    print(f"  Min: {region.min():+.6f}")
    print(f"  Max: {region.max():+.6f}")
