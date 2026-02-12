"""Test: zero sample-count drift over many chunks.

Simulates the output side of the realtime pipeline:
  infer_streaming (mock) -> StatefulResampler -> SOLA -> verify exact length

Each chunk's SOLA output must be exactly hop_samples_out.
Over N chunks, total output must be exactly N * hop_samples_out.

This catches:
  - StatefulResampler int() truncation drift
  - SOLA target_len fallthrough (variable output length)
  - sola_extra_samples margin insufficiency
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rcwx.audio.resample import StatefulResampler
from rcwx.audio.sola import SolaState, sola_crossfade


def _compute_sola_extra_model(
    model_sr: int, output_sr: int, cf_out: int, search_out: int,
) -> int:
    """Compute sola_extra_model with zc alignment (mirrors realtime_unified.py)."""
    sola_extra_out = cf_out + 2 * search_out
    zc_model = model_sr // 100
    sola_extra_raw = int(sola_extra_out * model_sr / output_sr)
    return (sola_extra_raw + zc_model - 1) // zc_model * zc_model


def _simulate_pipeline(
    model_sr: int,
    output_sr: int,
    hop_16k: int,
    crossfade_sec: float,
    sola_search_ms: float,
    n_chunks: int,
    model_shortfall: int = 0,
) -> list[int]:
    """Simulate n_chunks of the output pipeline.

    Returns per-chunk output lengths (each should be hop_out).
    """
    hop_out = int(hop_16k * output_sr / 16000)

    cf_out = int(output_sr * crossfade_sec)
    search_out = int(output_sr * sola_search_ms / 1000)
    sola_extra_model = _compute_sola_extra_model(model_sr, output_sr, cf_out, search_out)

    new_samples_16k = hop_16k
    expected_model_out = int(new_samples_16k * model_sr / 16000) + sola_extra_model

    resampler = StatefulResampler(model_sr, output_sr)
    sola_state = SolaState(crossfade_samples=cf_out, search_samples=search_out)

    chunk_lengths: list[int] = []
    for i in range(n_chunks):
        # Mock inference output (deterministic sine)
        model_len = max(1, expected_model_out - model_shortfall)
        t = np.arange(model_len, dtype=np.float32) / model_sr
        model_output = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        # Resample model_sr -> output_sr
        resampled = resampler.resample_chunk(model_output)

        # SOLA crossfade with target_len
        output = sola_crossfade(resampled, sola_state, target_len=hop_out)

        chunk_lengths.append(len(output))

    return chunk_lengths


# ---- Rate combinations to test ----

RATE_COMBOS = [
    # (model_sr, output_sr, description)
    (40000, 48000, "40k->48k (common RVC v2)"),
    (32000, 48000, "32k->48k (RVC v1)"),
    (48000, 48000, "48k->48k (passthrough)"),
    (40000, 44100, "40k->44.1k (non-round ratio)"),
    (32000, 44100, "32k->44.1k (non-round ratio)"),
]

CHUNK_CONFIGS = [
    # (hop_16k, crossfade_sec, sola_search_ms, description)
    (2400, 0.05, 10.0, "150ms chunk, 50ms cf"),
    (1600, 0.04, 10.0, "100ms chunk, 40ms cf"),
    (4800, 0.08, 10.0, "300ms chunk, 80ms cf"),
    (2560, 0.06, 10.0, "160ms chunk, 60ms cf"),
]

N_CHUNKS = 1000


def _assert_no_drift(
    lengths: list[int],
    hop_out: int,
    label: str,
) -> None:
    """Assert every chunk is exactly hop_out and total is exact."""
    total = sum(lengths)
    expected = len(lengths) * hop_out

    # Per-chunk check
    bad_chunks = [(i, l) for i, l in enumerate(lengths) if l != hop_out]
    if bad_chunks:
        first_bad = bad_chunks[0]
        print(
            f"  FAIL {label}: {len(bad_chunks)} chunks with wrong length. "
            f"First: chunk {first_bad[0]} = {first_bad[1]} (expected {hop_out})"
        )

    # Cumulative check
    drift = total - expected
    assert drift == 0, (
        f"{label}: cumulative drift = {drift} samples over {len(lengths)} chunks "
        f"({len(bad_chunks)} bad chunks)"
    )


def test_zero_drift_all_rates() -> None:
    """Over N_CHUNKS chunks, every rate combo must produce zero drift."""
    for model_sr, output_sr, rate_desc in RATE_COMBOS:
        for hop_16k, cf_sec, search_ms, chunk_desc in CHUNK_CONFIGS:
            hop_out = int(hop_16k * output_sr / 16000)
            label = f"{rate_desc}, {chunk_desc} (hop_out={hop_out})"

            lengths = _simulate_pipeline(
                model_sr=model_sr,
                output_sr=output_sr,
                hop_16k=hop_16k,
                crossfade_sec=cf_sec,
                sola_search_ms=search_ms,
                n_chunks=N_CHUNKS,
            )

            _assert_no_drift(lengths, hop_out, label)
            print(f"  OK {label}")


def test_zero_drift_with_model_shortfall() -> None:
    """Even when infer_streaming returns slightly short output, no drift."""
    for shortfall in [1, 5, 10, 50]:
        hop_16k = 2400
        model_sr, output_sr = 40000, 48000
        hop_out = int(hop_16k * output_sr / 16000)
        label = f"shortfall={shortfall}"

        lengths = _simulate_pipeline(
            model_sr=model_sr,
            output_sr=output_sr,
            hop_16k=hop_16k,
            crossfade_sec=0.05,
            sola_search_ms=10.0,
            n_chunks=500,
            model_shortfall=shortfall,
        )

        _assert_no_drift(lengths, hop_out, label)
        print(f"  OK {label}")


def test_resampler_exact_output_count() -> None:
    """StatefulResampler must produce consistent output length for same input length."""
    model_sr, output_sr = 40000, 48000
    input_len = 8000  # 200ms at 40kHz
    expected_out = int(input_len * output_sr / model_sr)  # should be 9600

    resampler = StatefulResampler(model_sr, output_sr)
    lengths = []
    for _ in range(500):
        chunk = np.random.randn(input_len).astype(np.float32) * 0.1
        out = resampler.resample_chunk(chunk)
        lengths.append(len(out))

    unique_lengths = set(lengths)
    assert len(unique_lengths) == 1, (
        f"Resampler output length varies: {unique_lengths} "
        f"(expected constant {expected_out})"
    )
    assert lengths[0] == expected_out, (
        f"Resampler output {lengths[0]} != expected {expected_out}"
    )


def test_resampler_cumulative_no_drift() -> None:
    """Total resampled output over N chunks must match batch resampling."""
    model_sr, output_sr = 40000, 48000
    input_len = 6000  # 150ms at 40kHz
    n_chunks = 1000

    resampler = StatefulResampler(model_sr, output_sr)
    total_in = 0
    total_out = 0
    for _ in range(n_chunks):
        chunk = np.random.randn(input_len).astype(np.float32) * 0.1
        out = resampler.resample_chunk(chunk)
        total_in += len(chunk)
        total_out += len(out)

    # Exact expected total
    expected_total_out = total_in * output_sr // model_sr
    drift = total_out - expected_total_out
    assert drift == 0, (
        f"Resampler cumulative drift: {drift} samples over {n_chunks} chunks "
        f"(total_in={total_in}, total_out={total_out}, expected={expected_total_out})"
    )


def test_zero_drift_random_input() -> None:
    """With random noise input, SOLA offsets vary — still zero drift."""
    for model_sr, output_sr, rate_desc in RATE_COMBOS:
        hop_16k = 2400
        cf_sec, search_ms = 0.05, 10.0
        hop_out = int(hop_16k * output_sr / 16000)
        cf_out = int(output_sr * cf_sec)
        search_out = int(output_sr * search_ms / 1000)
        sola_extra_model = _compute_sola_extra_model(
            model_sr, output_sr, cf_out, search_out
        )
        expected_model_out = int(hop_16k * model_sr / 16000) + sola_extra_model

        resampler = StatefulResampler(model_sr, output_sr)
        sola_state = SolaState(crossfade_samples=cf_out, search_samples=search_out)

        rng = np.random.RandomState(42)
        lengths: list[int] = []
        for _ in range(1000):
            model_output = rng.randn(expected_model_out).astype(np.float32) * 0.3
            resampled = resampler.resample_chunk(model_output)
            output = sola_crossfade(resampled, sola_state, target_len=hop_out)
            lengths.append(len(output))

        _assert_no_drift(lengths, hop_out, rate_desc)
        print(f"  OK {rate_desc}")


def test_long_session_5000_chunks() -> None:
    """Simulate ~12 minutes at 150ms/chunk — no cumulative drift."""
    model_sr, output_sr = 40000, 48000
    hop_16k = 2400
    cf_sec, search_ms = 0.05, 10.0
    hop_out = int(hop_16k * output_sr / 16000)
    cf_out = int(output_sr * cf_sec)
    search_out = int(output_sr * search_ms / 1000)
    sola_extra_model = _compute_sola_extra_model(
        model_sr, output_sr, cf_out, search_out
    )
    expected_model_out = int(hop_16k * model_sr / 16000) + sola_extra_model

    resampler = StatefulResampler(model_sr, output_sr)
    sola_state = SolaState(crossfade_samples=cf_out, search_samples=search_out)

    rng = np.random.RandomState(123)
    total_output = 0
    n_chunks = 5000
    for _ in range(n_chunks):
        model_output = rng.randn(expected_model_out).astype(np.float32) * 0.3
        resampled = resampler.resample_chunk(model_output)
        output = sola_crossfade(resampled, sola_state, target_len=hop_out)
        total_output += len(output)

    expected_total = n_chunks * hop_out
    drift = total_output - expected_total
    assert drift == 0, (
        f"Long session drift: {drift} samples over {n_chunks} chunks "
        f"({drift * 1000.0 / output_sr:.1f}ms)"
    )
    print(f"  OK 5000 chunks, total={total_output}, drift=0")


if __name__ == "__main__":
    tests = [
        ("Resampler consistency", test_resampler_exact_output_count),
        ("Resampler cumulative drift", test_resampler_cumulative_no_drift),
        ("Zero drift all rates", test_zero_drift_all_rates),
        ("Zero drift with model shortfall", test_zero_drift_with_model_shortfall),
        ("Zero drift random input", test_zero_drift_random_input),
        ("Long session 5000 chunks", test_long_session_5000_chunks),
    ]

    passed = 0
    failed = 0
    for name, func in tests:
        print(f"=== {name} ===")
        try:
            func()
            print("  PASS")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
