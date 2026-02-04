"""Realtime volume test with detailed logging."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging to file
log_file = Path("test_output/realtime_test.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
log = open(log_file, "w", encoding="utf-8", buffering=1)

def log_msg(msg):
    timestamp = time.strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    log.write(full_msg + "\n")
    log.flush()

log_msg("="*80)
log_msg("REALTIME VOLUME TEST START")
log_msg("="*80)

try:
    log_msg("Importing numpy...")
    import numpy as np

    log_msg("Importing scipy...")
    from scipy.io import wavfile

    log_msg("Importing RCWX modules...")
    from rcwx.pipeline.inference import RVCPipeline
    from rcwx.pipeline.realtime import RealtimeVoiceChanger, RealtimeConfig
    from rcwx.audio.resample import resample

    log_msg("Loading audio file...")
    sr, audio = wavfile.read("debug_audio/01_input_raw.wav")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Use first 1.5s only
    max_samples = int(sr * 1.5)
    audio = audio[:max_samples]

    log_msg(f"Audio loaded: {len(audio)/sr:.2f}s @ {sr}Hz")

    # Resample to 48kHz
    if sr != 48000:
        log_msg("Resampling to 48kHz...")
        audio = resample(audio, sr, 48000)
        sr = 48000

    # Analyze original
    start_idx = int(len(audio) * 0.3)
    end_idx = int(len(audio) * 0.7)
    orig_region = audio[start_idx:end_idx]
    orig_rms = float(np.sqrt(np.mean(orig_region**2)))
    orig_min = float(np.min(orig_region))
    orig_max = float(np.max(orig_region))

    log_msg(f"Original (30-70%): RMS={orig_rms:.6f}, Min={orig_min:+.6f}, Max={orig_max:+.6f}")

    # Load model
    log_msg("Creating RVCPipeline...")
    pipeline = RVCPipeline("sample_data/hogaraka/hogarakav2.pth", use_compile=False)

    log_msg("Loading model (this may take 1-2 minutes)...")
    load_start = time.time()
    pipeline.load()
    load_time = time.time() - load_start
    log_msg(f"Model loaded in {load_time:.1f}s")

    # Setup realtime
    log_msg("Setting up RealtimeVoiceChanger...")
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
        voice_gate_mode="off",  # Disable for testing
        use_feature_cache=False,  # Disable for testing
        prebuffer_chunks=1,
        buffer_margin=0.3,
    )

    changer = RealtimeVoiceChanger(pipeline, config=config)
    pipeline.clear_cache()
    changer._recalculate_buffers()
    changer._running = True
    changer._output_started = True  # Skip pre-buffering

    expected_chunks = int(len(audio) / 48000 / 0.15) + 10
    changer.output_buffer.set_max_latency(expected_chunks * int(48000 * 0.15) * 3)

    log_msg("Processing audio...")
    input_block = int(48000 * 0.02)
    pos = 0
    chunks_done = 0

    while pos < len(audio):
        block = audio[pos:pos + input_block]
        if len(block) < input_block:
            block = np.pad(block, (0, input_block - len(block)))
        pos += input_block
        changer.process_input_chunk(block)
        if changer.process_next_chunk():
            chunks_done += 1
            if chunks_done % 3 == 0:
                log_msg(f"  Chunk {chunks_done} done ({pos/len(audio)*100:.0f}%)")

    log_msg("Final processing...")
    while changer.process_next_chunk():
        chunks_done += 1
        log_msg(f"  Final chunk {chunks_done}")

    changer.flush_final_sola_buffer()

    log_msg(f"Output queue size: {changer._output_queue.qsize()}")
    log_msg(f"Output buffer available: {changer.output_buffer.available}")

    # Check first chunk from queue
    if changer._output_queue.qsize() > 0:
        first_chunk = changer._output_queue.get_nowait()
        chunk_rms = float(np.sqrt(np.mean(first_chunk**2)))
        log_msg(f"First queue chunk: {len(first_chunk)} samples, RMS={chunk_rms:.6f}")
        changer._output_queue.put_nowait(first_chunk)

    log_msg("Collecting output...")
    output_chunks = []
    # Collect directly from queue
    while True:
        try:
            chunk = changer._output_queue.get_nowait()
            output_chunks.append(chunk)
        except:
            break
    log_msg(f"Collected {len(output_chunks)} chunks from queue")

    output = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)
    log_msg(f"Output: {len(output)} samples ({len(output)/48000:.2f}s)")

    # Analyze processed
    if len(output) > 0:
        start_idx = int(len(output) * 0.3)
        end_idx = int(len(output) * 0.7)
        if end_idx > len(output):
            end_idx = len(output)
        if start_idx < end_idx:
            proc_region = output[start_idx:end_idx]
            proc_rms = float(np.sqrt(np.mean(proc_region**2)))
            proc_min = float(np.min(proc_region))
            proc_max = float(np.max(proc_region))

            log_msg(f"Processed (30-70%): RMS={proc_rms:.6f}, Min={proc_min:+.6f}, Max={proc_max:+.6f}")

            rms_ratio = proc_rms / orig_rms if orig_rms > 0 else 0
            rms_db = 20 * np.log10(rms_ratio) if rms_ratio > 0 else -np.inf

            log_msg("")
            log_msg("="*80)
            log_msg("COMPARISON")
            log_msg("="*80)
            log_msg(f"RMS: {proc_rms:.6f} vs {orig_rms:.6f} = {rms_ratio:.1%} ({rms_db:+.1f}dB)")
            log_msg(f"Min: {proc_min:+.6f} vs {orig_min:+.6f}")
            log_msg(f"Max: {proc_max:+.6f} vs {orig_max:+.6f}")

            if rms_ratio >= 0.85:
                verdict = "SUCCESS - Volume preserved"
            elif rms_ratio >= 0.70:
                verdict = "WARNING - Noticeable reduction"
            else:
                verdict = "ISSUE - Significant loss"

            log_msg(f"\nVERDICT: {verdict}")
            log_msg("="*80)

            # Save
            output_path = "test_output/realtime_volume_output.wav"
            output_int16 = (output * 32767).astype(np.int16)
            wavfile.write(output_path, 48000, output_int16)
            log_msg(f"Saved: {output_path}")
        else:
            log_msg("ERROR: Output too short for analysis")
    else:
        log_msg("ERROR: No output generated")

except Exception as e:
    log_msg(f"ERROR: {e}")
    import traceback
    log_msg(traceback.format_exc())
finally:
    log.close()
    print(f"\nLog saved to: {log_file}")
