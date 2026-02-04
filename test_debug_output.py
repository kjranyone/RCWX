"""Debug realtime output to find where audio is lost."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig

print("="*80)
print("DEBUG OUTPUT TEST")
print("="*80)

# Load audio
sr, audio = wavfile.read("debug_audio/01_input_raw.wav")
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
if audio.ndim > 1:
    audio = audio[:, 0]

# Use first 0.5s only for quick test
max_samples = int(sr * 0.5)
audio = audio[:max_samples]

print(f"\nInput: {len(audio)/sr:.2f}s @ {sr}Hz")
print(f"Input RMS: {np.sqrt(np.mean(audio**2)):.6f}")
print(f"Input range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")

# Load model
print("\nLoading model...")
pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
pipeline.load()

# Setup realtime with minimal config (no SOLA, no cache)
print("\nSetting up RealtimeVoiceChanger...")
config = RealtimeConfig(
    mic_sample_rate=48000 if sr != 48000 else sr,
    input_sample_rate=16000,
    output_sample_rate=40000,  # RVC output rate
    chunk_sec=0.15,
    f0_method="fcpe",
    chunking_mode="wokada",
    context_sec=0.10,
    crossfade_sec=0.05,
    use_sola=False,  # Disable SOLA to simplify
    index_rate=0.0,
    voice_gate_mode="off",  # Disable voice gate
    use_feature_cache=False,  # Disable cache
    prebuffer_chunks=0,  # No pre-buffering
    buffer_margin=1.0,
)

# Resample to 48kHz if needed
if sr != 48000:
    from rcwx.audio.resample import resample
    audio = resample(audio, sr, 48000)
    sr = 48000

changer = RealtimeVoiceChanger(pipeline, config=config)
changer._recalculate_buffers()
changer._running = True
changer._output_started = True  # Skip pre-buffering

# Process audio
print("\nProcessing audio...")
input_block = int(48000 * 0.02)
pos = 0
chunks_processed = 0

while pos < len(audio):
    block = audio[pos:pos + input_block]
    if len(block) < input_block:
        block = np.pad(block, (0, input_block - len(block)))
    pos += input_block

    changer.process_input_chunk(block)

    if changer.process_next_chunk():
        chunks_processed += 1
        print(f"  Chunk {chunks_processed} processed")

        # Check output queue
        print(f"    Output queue size: {changer._output_queue.qsize()}")

        # Check if there's audio in the queue
        try:
            output_chunk = changer._output_queue.get_nowait()
            print(f"    Got output chunk: {len(output_chunk)} samples")
            print(f"    Output RMS: {np.sqrt(np.mean(output_chunk**2)):.6f}")
            print(f"    Output range: [{np.min(output_chunk):.6f}, {np.max(output_chunk):.6f}]")
            # Put it back
            changer._output_queue.put_nowait(output_chunk)
        except:
            print("    No output in queue")

print("\nFinal processing...")
while changer.process_next_chunk():
    chunks_processed += 1
    print(f"  Final chunk {chunks_processed}")

print(f"\nTotal chunks processed: {chunks_processed}")
print(f"Final output queue size: {changer._output_queue.qsize()}")

# Collect output
print("\nCollecting output...")
output_chunks = []
iterations = 0
max_iterations = 100

while iterations < max_iterations:
    chunk = changer.get_output_chunk(int(40000 * 0.02))
    if len(chunk) == 0:
        break

    chunk_rms = np.sqrt(np.mean(chunk**2))
    if chunk_rms > 0:
        print(f"  Chunk {iterations}: {len(chunk)} samples, RMS={chunk_rms:.6f}")

    output_chunks.append(chunk)
    iterations += 1

output = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)
print(f"\nFinal output: {len(output)} samples ({len(output)/40000:.2f}s)")

if len(output) > 0:
    output_rms = np.sqrt(np.mean(output**2))
    print(f"Output RMS: {output_rms:.6f}")
    print(f"Output range: [{np.min(output):.6f}, {np.max(output):.6f}]")
else:
    print("NO OUTPUT GENERATED!")

print("="*80)
