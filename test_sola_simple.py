"""Simple SOLA test to debug the issue."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.io import wavfile
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig

print("="*80)
print("SIMPLE SOLA TEST")
print("="*80)

# Load audio
sr, audio = wavfile.read("debug_audio/01_input_raw.wav")
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
if audio.ndim > 1:
    audio = audio[:, 0]

# Use first 0.5s for quick test
max_samples = int(sr * 0.5)
audio = audio[:max_samples]
print(f"\nInput: {len(audio)/sr:.2f}s @ {sr}Hz, RMS={np.sqrt(np.mean(audio**2)):.6f}")

# Resample to 48kHz if needed
if sr != 48000:
    from rcwx.audio.resample import resample
    audio = resample(audio, sr, 48000)
    sr = 48000

# Load model
print("Loading model...")
pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)
pipeline.load()

# Setup with SOLA enabled
config = RealtimeConfig(
    mic_sample_rate=48000,
    input_sample_rate=16000,
    output_sample_rate=48000,
    chunk_sec=0.15,
    f0_method="fcpe",
    chunking_mode="wokada",
    context_sec=0.10,
    crossfade_sec=0.05,
    use_sola=True,
    index_rate=0.0,
    voice_gate_mode="off",
    use_feature_cache=False,
    prebuffer_chunks=0,
    buffer_margin=1.0,
)

changer = RealtimeVoiceChanger(pipeline, config=config)
changer._recalculate_buffers()
changer._running = True
changer._output_started = True

print(f"\nSOLA state: buffer_frame={changer._sola_state.sola_buffer_frame}")
print(f"Context samples: {int(48000 * 0.10)}")
print(f"Crossfade samples: {int(48000 * 0.05)}")

# Process audio
print("\nProcessing...")
input_block = int(48000 * 0.02)
pos = 0

while pos < len(audio):
    block = audio[pos:pos + input_block]
    if len(block) < input_block:
        block = np.pad(block, (0, input_block - len(block)))
    pos += input_block
    changer.process_input_chunk(block)

    if changer.process_next_chunk():
        # Check output
        try:
            chunk = changer._output_queue.get_nowait()
            rms = np.sqrt(np.mean(chunk**2))
            print(f"  Output chunk: {len(chunk)} samples, RMS={rms:.6f}")
            if rms == 0:
                print("    WARNING: Zero RMS!")
            changer._output_queue.put_nowait(chunk)
        except:
            pass

print("\nFinal processing...")
while changer.process_next_chunk():
    pass

# Collect all output
print("\nCollecting...")
output_chunks = []
while True:
    try:
        chunk = changer._output_queue.get_nowait()
        output_chunks.append(chunk)
    except:
        break

output = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)
print(f"\nOutput: {len(output)} samples, RMS={np.sqrt(np.mean(output**2)) if len(output) > 0 else 0:.6f}")

print("="*80)
