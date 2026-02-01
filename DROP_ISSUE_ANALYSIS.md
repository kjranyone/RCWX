# Buffer Drop Issue Analysis

## Executive Summary

**Root Cause**: The output buffer is accumulating far more samples than expected due to a mismatch between:
1. The audio callback blocksize (variable, OS-controlled)
2. The inference chunk processing rate (fixed, 100ms chunks)
3. The output buffer max latency calculation (incorrectly configured)

**Impact**: 1,000-7,000 samples dropped every ~14 seconds during realtime inference.

---

## Problem Details

### 1. The Input/Output Flow Mismatch

From logs (`rcwx_20260202_072954.log`):

```
[INFER] Chunk #1580: in=8000, out=18235, infer=141ms, latency=1782ms, buf=60638
[OUTPUT] frames=5512, chunks_added=1, output_buffer=66150, dropped=1834
```

**Analysis**:

#### Input Side (@ 16kHz processing rate)
- **Expected**: 1,600 samples per chunk (100ms * 16,000 Hz)
- **Actual**: 8,000 samples per chunk
- **Ratio**: 5x larger than expected!
- **Actual duration**: 500ms per chunk (not 100ms)

#### Output Side (@ 48kHz output rate)
- **Model output**: 17,640-18,515 samples @ 40kHz (varies due to RVC synthesizer)
- **After resampling to 48kHz**: ~21,168-22,218 samples
- **Expected per 100ms chunk**: 4,800 samples @ 48kHz
- **Actual per chunk**: ~21,000 samples (4.4x larger)

#### Buffer Settings
- **Max latency**: 6,240 samples (130ms)
- **Actual buffer**: 60,000-66,000 samples (1,250-1,375ms)
- **Overflow**: ~60,000 samples = **10x the limit**

### 2. Root Cause: Audio Callback Blocksize

From `realtime.py:990-1011`:

```python
output_chunk_sec = self.config.chunk_sec / 4  # 0.10 / 4 = 0.025s = 25ms
output_blocksize = int(self.config.output_sample_rate * output_chunk_sec)
# = 48000 * 0.025 = 1200 samples

self.audio_input = AudioInput(
    blocksize=int(self.config.mic_sample_rate * output_chunk_sec),
    # = 48000 * 0.025 = 1200 samples
)
```

**The Problem**:
- Code requests blocksize = 1,200 samples (25ms)
- OS/driver may not support this exact blocksize
- Actual blocksize could be 5,512 samples (from logs) = **115ms**, not 25ms
- This is 4.6x larger than requested!

**Why this happens**:
1. WASAPI/ASIO drivers have preferred blocksizes (power of 2, or specific multiples)
2. `stream_base.py` tries fallback blocksizes: [scaled, original, 2048, 4096, 1024, 512, 8192]
3. Eventually settles on a size the driver accepts
4. No validation that the actual blocksize matches the requested one

### 3. The Accumulation Problem

**Normal flow (if blocksize matched)**:
```
Input callback (25ms):  1,200 samples @ 48kHz
                       ↓
Resample to 16kHz:       400 samples
                       ↓
Process 5 callbacks:   2,000 samples (> 1,600 = 1 chunk threshold)
                       ↓
Inference (100ms):    ~5,000 samples @ 48kHz output
                       ↓
Output buffer adds:    5,000 samples
Output callback (25ms): removes 1,200 samples
Net gain:             +3,800 samples per chunk
```

**Actual flow (with blocksize=5,512)**:
```
Input callback (115ms): 5,512 samples @ 48kHz
                       ↓
Resample to 16kHz:     1,837 samples
                       ↓
Process 4-5 callbacks: 7,348-9,185 samples (triggers inference)
                       ↓
Inference (500ms):    ~21,000 samples @ 48kHz output
                       ↓
Output buffer adds:    21,000 samples
Output callback (115ms): removes 5,512 samples
Net gain:             +15,488 samples per chunk
```

**Result**: Buffer grows from 0 → 66,150 in ~4 chunks, triggers drop.

---

## Why StatefulResampler is Not the Culprit

Tests confirm `StatefulResampler` produces correct output lengths:
- 4,800 @ 48kHz → 1,600 @ 16kHz ✓
- 20,000 @ 40kHz → 24,000 @ 48kHz ✓
- Consistent across 10 chunks ✓

The resampler is working as designed.

---

## Evidence from Logs

### Inference Pattern
```
in=8000    # 500ms @ 16kHz (not 100ms!)
out=18235  # ~456ms @ 40kHz
```

### Buffer Growth
```
buf=44102 → buf=60638 → buf=66150 (overflow) → dropped=1834
```

### Drop Pattern
Every 10-20 chunks (~5-10 seconds), drops 1,000-7,000 samples.

---

## Solutions

### Solution 1: Validate Actual Blocksize (Recommended)

After `AudioInput.start()`, check if actual blocksize matches requested:

```python
# In realtime.py:start()
requested_blocksize = int(self.config.mic_sample_rate * output_chunk_sec)
actual_blocksize = self.audio_input.blocksize

if actual_blocksize != requested_blocksize:
    logger.warning(
        f"Blocksize mismatch: requested={requested_blocksize}, "
        f"actual={actual_blocksize} ({actual_blocksize/self.config.mic_sample_rate*1000:.1f}ms)"
    )
    # Recalculate max_latency based on actual blocksize
    actual_chunk_sec = actual_blocksize / self.config.mic_sample_rate
    # ... adjust buffer settings
```

### Solution 2: Adjust Max Latency Dynamically

Current calculation:
```python
max_latency_sec = self.config.chunk_sec * (prebuffer_chunks + buffer_margin)
# = 0.10 * (1 + 0.3) = 0.13s = 6,240 samples
```

Should be based on **actual processing rate**:
```python
# Account for actual inference output size
avg_output_per_chunk = 21000  # empirical from logs
avg_output_sec = avg_output_per_chunk / self.config.output_sample_rate
# = 21000 / 48000 = 0.4375s

max_latency_sec = avg_output_sec * (prebuffer_chunks + buffer_margin)
# = 0.4375 * (1 + 0.3) = 0.57s = 27,360 samples
```

### Solution 3: Use Fixed Chunk Accumulation

Instead of relying on callback blocksize, accumulate audio until exactly `chunk_sec` worth:

```python
# In _on_audio_input()
self.input_buffer.add_input(audio)

# Only trigger inference when we have EXACTLY chunk_sec worth
target_samples = int(self.config.mic_sample_rate * self.config.chunk_sec)
while self.input_buffer.buffered_samples >= target_samples:
    chunk = self.input_buffer.get_chunk()
    # ... process
```

This decouples inference from OS callback timing.

### Solution 4: Adaptive Buffer Sizing

Monitor actual output rate and adjust max_latency in real-time:

```python
# Track actual output per chunk
self.output_sizes_history.append(len(output))
if len(self.output_sizes_history) > 10:
    avg_output = np.mean(self.output_sizes_history[-10:])
    new_max_latency = int(avg_output * (prebuffer_chunks + buffer_margin) * 1.5)
    self.output_buffer.set_max_latency(new_max_latency)
```

---

## Recommended Fix

**Implement Solution 1 + Solution 3**:

1. **Validate blocksize** after stream start, log warnings if mismatched
2. **Decouple inference from callbacks** by accumulating to fixed chunk size
3. **Adjust max_latency** based on actual output measurements (first 10 chunks)

This ensures:
- Predictable chunk processing (100ms chunks regardless of OS blocksize)
- No buffer overflow (latency limit matches actual output rate)
- Robust across different audio drivers (WASAPI, ASIO, MME)

---

## Files to Modify

1. **`rcwx/pipeline/realtime.py`**:
   - Add blocksize validation after `start()`
   - Implement fixed chunk accumulation in `_on_audio_input()`
   - Dynamic max_latency adjustment based on first N chunks

2. **`rcwx/audio/buffer.py`** (optional):
   - Add `OutputBuffer.adjust_max_latency()` method for adaptive sizing

---

## Testing Plan

1. **Blocksize test**: Log requested vs actual blocksize on different systems
2. **Chunk accumulation test**: Verify inference triggers at exactly chunk_sec intervals
3. **Buffer stability test**: Monitor output buffer size over 5 minutes, should stabilize
4. **Drop count test**: Confirm drops = 0 after fix

---

## Additional Notes

### Why the input is 8000 samples @ 16kHz

From the data:
- Blocksize: 5,512 samples @ 48kHz
- After resample to 16kHz: 5512 * 16000/48000 = 1,837 samples
- After 4-5 callbacks: 7,348-9,185 samples buffered
- ChunkBuffer triggers when >= required (1,600 + context)
- Processes all available: ~8,000 samples

This is correct behavior **given the large blocksize**.

### Why output buffer grows

- Input rate: 5,512 samples / 115ms @ 48kHz = **47,930 Hz effective** (close to 48kHz)
- Processing produces: ~21,000 samples per 500ms chunk
- Output rate: 5,512 samples / 115ms @ 48kHz = **47,930 Hz effective**
- Net accumulation: +15,488 samples per chunk (21,000 - 5,512)
- After 4 chunks: 61,952 samples (exceeds 6,240 limit by 10x)

The fundamental issue: **inference produces 500ms of audio but consumes 500ms of input, while output only drains 115ms per callback**.

---

## Conclusion

The drops are **not caused by StatefulResampler**, but by:

1. **OS audio driver returning larger blocksizes than requested** (5,512 instead of 1,200)
2. **Inference processing larger chunks** (500ms instead of 100ms)
3. **Output buffer max_latency too small** (6,240 samples vs 60,000 actual)

The fix is to decouple inference from OS callback timing and adjust buffer limits dynamically.
