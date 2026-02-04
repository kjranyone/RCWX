"""Detailed discontinuity analysis of output."""
from scipy.io import wavfile
import numpy as np

sr, audio = wavfile.read("test_output/realtime_volume_output.wav")
audio_f = audio.astype(np.float32) / 32768.0

# Find active region
nonzero_idx = np.where(audio != 0)[0]
if len(nonzero_idx) == 0:
    print("No active audio!")
    exit(1)

active_start = nonzero_idx[0]
active_end = nonzero_idx[-1]
active_audio = audio_f[active_start:active_end+1]

print("="*80)
print("DETAILED DISCONTINUITY ANALYSIS")
print("="*80)
print(f"\nActive region: {active_start/sr:.3f}s - {active_end/sr:.3f}s ({len(active_audio)/sr:.3f}s)")

# Calculate first-order differences (derivative)
diff = np.abs(np.diff(active_audio))

# Find discontinuities (large jumps)
thresholds = [0.05, 0.10, 0.15, 0.20]

for threshold in thresholds:
    discontinuities = np.where(diff > threshold)[0]
    print(f"\nDiscontinuities (>{threshold:.2f}): {len(discontinuities)}")

    if len(discontinuities) > 0 and threshold == 0.10:
        print(f"\nFirst 20 discontinuities (>{threshold:.2f}):")
        for i, idx in enumerate(discontinuities[:20]):
            time = (active_start + idx) / sr
            jump = diff[idx]
            # Check if it's at chunk boundary (150ms = 7200 samples)
            chunk_pos = idx % 7200
            is_boundary = chunk_pos < 100 or chunk_pos > 7100
            boundary_marker = " [BOUNDARY]" if is_boundary else ""
            print(f"  {i+1}. {time:.4f}s: jump={jump:.4f}, pos_in_chunk={chunk_pos}{boundary_marker}")

# Analyze around chunk boundaries (150ms = 7200 samples)
chunk_samples = 7200
num_chunks = len(active_audio) // chunk_samples

print(f"\n" + "="*80)
print(f"CHUNK BOUNDARY DISCONTINUITIES")
print("="*80)
print(f"\nChunks: {num_chunks}, Chunk size: {chunk_samples} samples (150ms)")

boundary_issues = []
for i in range(1, num_chunks):
    boundary_idx = i * chunk_samples

    # Check ±50 samples around boundary
    window = 50
    start_idx = max(0, boundary_idx - window)
    end_idx = min(len(active_audio) - 1, boundary_idx + window)

    region = active_audio[start_idx:end_idx]
    region_diff = np.abs(np.diff(region))

    max_jump = np.max(region_diff)
    max_jump_idx = np.argmax(region_diff) + start_idx - boundary_idx

    if max_jump > 0.05:
        boundary_issues.append({
            'boundary': i,
            'time': (active_start + boundary_idx) / sr,
            'max_jump': max_jump,
            'offset_from_boundary': max_jump_idx
        })

print(f"\nBoundaries with discontinuities (>0.05): {len(boundary_issues)}")
for issue in boundary_issues:
    print(f"  Boundary {issue['boundary']} @ {issue['time']:.4f}s: "
          f"jump={issue['max_jump']:.4f}, offset={issue['offset_from_boundary']:+d}")

# Check for zero-crossings (clicks)
zero_crossings = 0
for i in range(1, len(active_audio)):
    if active_audio[i-1] * active_audio[i] < 0:  # Sign change
        if abs(active_audio[i] - active_audio[i-1]) > 0.1:
            zero_crossings += 1

print(f"\n" + "="*80)
print(f"SUMMARY")
print("="*80)
print(f"Large discontinuities (>0.10): {len(np.where(diff > 0.10)[0])}")
print(f"Chunk boundaries with issues: {len(boundary_issues)}/{num_chunks-1}")
print(f"Severe zero-crossings: {zero_crossings}")

if len(boundary_issues) > 0:
    print(f"\n⚠ ISSUE: {len(boundary_issues)} chunk boundaries have discontinuities!")
    print("This causes audible clicks/pops.")
else:
    print(f"\n✓ No significant discontinuities at chunk boundaries")

print("="*80)
