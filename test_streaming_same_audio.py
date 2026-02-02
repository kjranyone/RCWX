"""Test streaming vs batch with same 2-second audio."""

import sys
import numpy as np
from scipy.io import wavfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rcwx.audio.resample import resample
from rcwx.device import get_device
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig

# Load pipeline
pipeline = RVCPipeline(
    model_path="sample_data/hogaraka/hogarakav2.pth",
    device=get_device("xpu"),
    dtype="float16",
    use_compile=False,
)
pipeline.load()

# Load test audio (use 2 seconds)
sr, audio = wavfile.read("sample_data/seki.wav")
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
if len(audio.shape) > 1:
    audio = audio[:, 0]
if sr != 48000:
    audio = resample(audio, sr, 48000)

audio_test = audio[: int(2 * 48000)]

print(f"Input: {len(audio_test)} samples @ 48kHz = {len(audio_test) / 48000:.2f}s")

# Batch inference
audio_16k = resample(audio_test, 48000, 16000)
batch_output = pipeline.infer(
    audio_16k,
    input_sr=16000,
    pitch_shift=0,
    f0_method="fcpe",
    index_rate=0.0,
    voice_gate_mode="off",
    use_feature_cache=False,
)

# Resample to 48kHz
batch_48k = resample(batch_output, 40000, 48000)

print(f"Batch output: {len(batch_48k)} samples @ 48kHz = {len(batch_48k) / 48000:.2f}s")

# Streaming processing (w-okada mode)
config_wokada = RealtimeConfig(
    chunking_mode="wokada",
    chunk_sec=0.35,
    mic_sample_rate=48000,
    input_sample_rate=16000,
    output_sample_rate=48000,
    f0_method="fcpe",
    use_f0=True,
    use_feature_cache=False,  # Disable for testing
    use_sola=True,
    context_sec=0.05,
    crossfade_sec=0.05,
    prebuffer_chunks=0,
    max_queue_size=1000,
)

changer_wokada = RealtimeVoiceChanger(pipeline=pipeline, config=config_wokada)

# Process input
block_size = int(48000 * 0.35 / 4)
outputs = []

for i in range(0, len(audio_test), block_size):
    block = audio_test[i : i + block_size]
    changer_wokada.process_input_chunk(block)
    while changer_wokada.process_next_chunk():
        pass

# Flush and get all output
changer_wokada.flush_final_sola_buffer()
try:
    while True:
        audio_chunk = changer_wokada._output_queue.get_nowait()
        outputs.append(audio_chunk)
except:
    pass

streaming_output = np.concatenate(outputs)

print(
    f"Streaming output: {len(streaming_output)} samples @ 48kHz = {len(streaming_output) / 48000:.2f}s"
)

# Compare
min_len = min(len(batch_48k), len(streaming_output))
batch_trim = batch_48k[:min_len]
stream_trim = streaming_output[:min_len]

correlation = np.corrcoef(batch_trim, stream_trim)[0, 1]
mae = np.mean(np.abs(batch_trim - stream_trim))
rmse = np.sqrt(np.mean((batch_trim - stream_trim) ** 2))
energy_ratio = np.sum(stream_trim**2) / np.sum(batch_48k**2)

print(f"\nComparison:")
print(f"  Batch length: {len(batch_48k)} samples")
print(f"  Streaming length: {len(streaming_output)} samples")
print(f"  Correlation: {correlation:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  Energy ratio: {energy_ratio:.4f}x")

# Check result
if abs(correlation) > 0.9:
    print(f"\n  ✓ PASS: Correlation > 0.9")
elif abs(correlation) > 0.7:
    print(f"\n  ⚠ WARNING: Correlation between 0.7 and 0.9")
else:
    print(f"\n  ✗ FAIL: Correlation < 0.7")
