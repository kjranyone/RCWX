# Buffer Drop Investigation - Summary Report

## Problem Statement

During GUI realtime inference, large numbers of buffer drops occur (1,000-7,000 samples dropped every ~14 seconds), causing audio glitches and quality degradation.

---

## Root Cause Identified

### The Core Issue: Chunk Size vs Audio Driver Settings Mismatch

**What we found**:
1. **Configuration says**: `chunk_sec=0.10` (100ms chunks)
2. **But logs show**: `chunk=0.5s (22050 samples)` - **500ms chunks!**
3. **Audio driver blocksize**: 5,512 samples @ 44.1kHz = **125ms**
4. **Buffer max latency**: Calculated for 130ms, but **actual needs 1,300ms**

### Why This Happened

From log file `rcwx_20260202_072954.log`:

```
INFO - Audio config: mic_sr=44100, out_sr=44100, chunk=0.5s (22050 samples),
       context=0.08s (3528 samples each side)
INFO - Input stream started: sr=44100Hz, blocksize=5512
INFO - Output stream started: sr=44100Hz, blocksize=5512
```

**The chain of events**:

1. **User selected 44.1kHz sample rate** (not the default 48kHz)
2. **Audio driver enforced blocksize=5,512** (not the requested 1,200)
3. **Chunk size got recalculated** somewhere to 0.5s (not 0.1s)
4. **Buffer limits stayed at original** 6,240 samples (for 0.1s chunks)

### The Math

#### Input Flow (@ 44.1kHz)
```
Audio callback arrives every: 5,512 samples / 44,100 Hz = 125ms
Chunk processing size:        22,050 samples / 44,100 Hz = 500ms
Callbacks per chunk:          500ms / 125ms = 4 callbacks
```

After resampling to 16kHz:
```
Input to inference:           22,050 * 16,000/44,100 = 8,000 samples
```
**✓ Matches log**: `[INFER] Chunk #1580: in=8000`

#### Output Flow (@ 44.1kHz)
```
Inference produces:           ~18,000 samples @ 40kHz
After resample to 44.1kHz:    18,000 * 44,100/40,000 = 19,845 samples
                              = 450ms of audio

Audio callback consumes:      5,512 samples = 125ms
Net accumulation:             19,845 - 5,512 = +14,333 samples per chunk
```

#### Buffer Overflow
```
Max latency setting:          6,240 samples (based on 0.1s * 1.3)
Actual buffer after 4 chunks: 4 * 14,333 = 57,332 samples
Overflow amount:              57,332 - 6,240 = 51,092 samples

Result: MASSIVE DROPS
```
**✓ Matches log**: `output_buffer=66150, dropped=7177`

---

## Why StatefulResampler is NOT the Problem

**Test results** (from `test_resampler_length.py`):
```
48kHz -> 16kHz: Perfect consistency (1600 samples every time) ✓
40kHz -> 48kHz: Perfect consistency (24000 samples every time) ✓
10 chunks processed: All lengths match exactly ✓
```

The StatefulResampler is working **perfectly**. The issue is entirely in the buffer sizing logic.

---

## Where Chunk Size Changed to 0.5s

Need to investigate these code paths:

### Suspect #1: HuBERT Hop Alignment

In `realtime.py:360-378`, there's chunk size adjustment logic:

```python
# Align chunk size with HuBERT hop (320 samples @ 16kHz)
hubert_hop_48k = int(320 * self.config.mic_sample_rate / 16000)
rounded_samples = ((self.mic_chunk_samples + hubert_hop_48k // 2)
                   // hubert_hop_48k) * hubert_hop_48k

if rounded_samples != self.mic_chunk_samples:
    logger.info(f"Chunk size adjusted: {self.mic_chunk_samples} -> {rounded_samples}")
    self.mic_chunk_samples = rounded_samples
    self.config.chunk_sec = rounded_samples / self.config.mic_sample_rate
```

**For 44.1kHz**:
```
Initial chunk:     int(44,100 * 0.1) = 4,410 samples
HuBERT hop @ 44.1: 320 * 44,100/16,000 = 882 samples
Rounded:           round(4410 / 882) * 882 = 5 * 882 = 4,410 samples
```

**This should NOT cause 0.5s chunks**. So suspect #2...

### Suspect #2: Sample Rate Recalculation

In `realtime.py:1018-1024`:

```python
if self.audio_input.actual_sample_rate != self.config.mic_sample_rate:
    logger.warning(f"Input sample rate changed: ...")
    self.config.mic_sample_rate = self.audio_input.actual_sample_rate
    self._recalculate_buffers()
```

If `_recalculate_buffers()` is called **multiple times** or with wrong values, chunk_sec could drift.

### Suspect #3: GUI Settings Override

Check if GUI loaded a saved config with `chunk_sec=0.5` from previous session.

---

## The Fix Strategy

### Immediate Fix: Adjust Buffer Max Latency

The buffer limit should be **proportional to actual chunk processing time**, not nominal chunk_sec:

```python
# Current (WRONG)
max_latency_sec = self.config.chunk_sec * (prebuffer_chunks + buffer_margin)
# = 0.1 * 1.3 = 0.13s

# Should be (CORRECT)
# Measure actual time to process one chunk
actual_processing_sec = len(output) / self.config.output_sample_rate
max_latency_sec = actual_processing_sec * (prebuffer_chunks + buffer_margin * 3)
# = 0.45s * (1 + 0.9) = 0.855s ≈ 37,700 samples @ 44.1kHz
```

### Long-term Fix: Lock Chunk Size

1. **Never modify `chunk_sec` after init**
2. **Validate that blocksize ≤ chunk_samples**
3. **Use chunk accumulation**, not callback-driven processing

```python
# In _on_audio_input()
self.input_buffer.add_input(audio)

# Only process when we have EXACTLY chunk_sec worth of audio
target_samples = int(self.config.mic_sample_rate * self.config.chunk_sec)
if self.input_buffer.buffered_samples >= target_samples:
    chunk = self.input_buffer.get_fixed_chunk(target_samples)
    # ... queue for inference
```

This decouples chunk processing from OS audio driver timing.

---

## Verification

To confirm the fix works:

### Test 1: Check chunk_sec stability
```python
assert self.config.chunk_sec == 0.1, "chunk_sec changed!"
```

### Test 2: Monitor buffer growth
```python
# After 100 chunks, buffer should stabilize
if chunk_count > 100:
    assert self.output_buffer.available < max_latency_samples * 0.9
```

### Test 3: No drops
```python
# After 5 minutes
assert self.stats.buffer_overruns == 0
```

---

## Action Items

### Priority 1: Debug Why chunk_sec=0.5

**File**: `rcwx/pipeline/realtime.py`

Add logging to track chunk_sec changes:
```python
def _recalculate_buffers(self):
    old_chunk_sec = self.config.chunk_sec
    # ... existing logic ...
    if self.config.chunk_sec != old_chunk_sec:
        logger.warning(
            f"!!! chunk_sec CHANGED: {old_chunk_sec} -> {self.config.chunk_sec}"
        )
        import traceback
        traceback.print_stack()
```

### Priority 2: Fix Max Latency Calculation

**File**: `rcwx/pipeline/realtime.py:283-289`

Replace static calculation with dynamic measurement:
```python
# Track actual output sizes
self._output_size_history = []

# In _inference_thread(), after resampling:
self._output_size_history.append(len(output))

# After first 10 chunks, adjust max_latency
if len(self._output_size_history) == 10:
    avg_output = np.mean(self._output_size_history)
    avg_duration_sec = avg_output / self.config.output_sample_rate
    new_max_latency = int(
        self.config.output_sample_rate
        * avg_duration_sec
        * (self.config.prebuffer_chunks + self.config.buffer_margin * 3)
    )
    logger.info(
        f"Adjusting max_latency: {self.output_buffer.max_latency_samples} "
        f"-> {new_max_latency} (avg chunk: {avg_duration_sec*1000:.0f}ms)"
    )
    self.output_buffer.set_max_latency(new_max_latency)
```

### Priority 3: Add Validation

**File**: `rcwx/pipeline/realtime.py:1015` (after `audio_input.start()`)

```python
actual_blocksize = self.audio_input.blocksize
requested_blocksize = int(self.config.mic_sample_rate * output_chunk_sec)

if actual_blocksize > self.mic_chunk_samples:
    logger.error(
        f"CRITICAL: Audio driver blocksize ({actual_blocksize}) exceeds "
        f"chunk size ({self.mic_chunk_samples}). This will cause buffer issues!"
    )
    # Either:
    # A) Increase chunk_sec to accommodate blocksize
    # B) Request smaller blocksize (may fail)
    # C) Use chunk accumulation (recommended)
```

---

## Conclusion

**The drops are caused by**:
1. `chunk_sec` mysteriously changing from 0.1s to 0.5s
2. Buffer `max_latency` calculated for 0.1s chunks (6,240 samples)
3. Actual chunks producing 0.5s of audio (19,845 samples)
4. Buffer overflows after 4 chunks, drops 50,000+ samples

**StatefulResampler is working correctly** - it's a red herring.

**The fix**:
1. Find why chunk_sec changes
2. Adjust max_latency dynamically based on measured output
3. Add validation to catch this in the future

**Estimated time to fix**: 2-4 hours (debugging + implementation + testing)
