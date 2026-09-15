"""Tests for the input-side noise gate (guitar-NS style downward expander).

Guarantees:
- loud (above-threshold) speech passes essentially unchanged
- steady sub-threshold noise is attenuated by ~range_db
- hysteresis + hold keep the gate open through short dips
- release is a smooth ramp (no click-scale discontinuity)
- the gate is wired into the realtime preprocess stage
"""

from __future__ import annotations

import numpy as np

import rcwx.pipeline.realtime_unified as ru
from rcwx.audio.noise_gate import RANGE_DB, NoiseGate
from rcwx.pipeline.realtime_config import RealtimeConfig

SR = 16000


def _sine(freq: float, seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _auto_gate(sensitivity: str = "mid") -> NoiseGate:
    gate = NoiseGate(SR, threshold_db=-40.0)
    gate.auto_threshold = True
    gate.set_sensitivity(sensitivity)
    return gate


def test_loud_signal_passes_through() -> None:
    gate = NoiseGate(SR, threshold_db=-40.0)
    speech = _sine(220.0, 1.0, 0.3)  # ~-13 dBFS, well above threshold
    out = gate.process(speech)
    # Skip the attack ramp (~10ms to 99%); the rest must be unity gain.
    skip = int(SR * 0.03)
    assert np.allclose(out[skip:], speech[skip:], atol=2e-3)


def test_quiet_noise_attenuated() -> None:
    gate = NoiseGate(SR, threshold_db=-40.0)
    noise = (np.random.RandomState(0).randn(SR * 2) * 0.001).astype(np.float32)
    out = gate.process(noise)
    # After envelope release + hold + gain release (~0.6s worst case), the
    # output must sit at the attenuation floor.
    settled = slice(int(SR * 1.5), None)
    rms_in = float(np.sqrt(np.mean(noise[settled] ** 2)))
    rms_out = float(np.sqrt(np.mean(out[settled] ** 2)))
    assert rms_out < rms_in * (10 ** (-RANGE_DB / 20)) * 1.5 + 1e-7


def test_hold_keeps_gate_open_through_short_dip() -> None:
    gate = NoiseGate(SR, threshold_db=-40.0)
    loud = _sine(220.0, 0.2, 0.3)
    # 30ms dip (< 60ms hold) then loud again
    dip = np.zeros(int(SR * 0.03), dtype=np.float32)
    speech = np.concatenate([loud, dip, loud])
    out = gate.process(speech)
    # During the dip the gate must still be open (hold) -> near-silence out
    # is just the input's own silence, not extra attenuation... verify by
    # checking the signal right after the dip passes at ~unity immediately.
    after = slice(2 * len(loud), 2 * len(loud) + int(SR * 0.01))
    assert np.allclose(out[after], speech[after], atol=5e-3)


def test_release_is_smooth() -> None:
    gate = NoiseGate(SR, threshold_db=-40.0)
    loud = _sine(220.0, 0.3, 0.3)
    quiet = (np.random.RandomState(1).randn(SR * 2) * 0.001).astype(np.float32)
    out = gate.process(np.concatenate([loud, quiet]))
    # Across the whole close transition the one-pole gain ramp keeps the
    # output slope close to the input's own slope (a hard gate would chop
    # the signal with full-scale steps).
    quiet_out = out[len(loud) :]
    step = float(np.max(np.abs(np.diff(quiet_out))))
    raw_step = float(np.max(np.abs(np.diff(quiet))))
    assert step < raw_step + 0.05


def test_threshold_property_updates_levels() -> None:
    gate = NoiseGate(SR, threshold_db=-30.0)
    gate.threshold_db = -50.0
    assert gate.threshold_db == -50.0
    # Signal at -45 dBFS must now open the (re-armed) gate.
    gate.reset()
    sig = _sine(220.0, 0.3, 10 ** (-45 / 20))
    out = gate.process(sig)
    skip = int(SR * 0.01)
    assert float(np.mean(np.abs(out[skip:]))) > 0.5 * float(
        np.mean(np.abs(sig[skip:]))
    )


def test_auto_threshold_gates_noise_not_tone() -> None:
    gate = _auto_gate()
    rng = np.random.RandomState(3)
    noise = (rng.randn(SR) * 0.001).astype(np.float32)
    tone = _sine(220.0, 1.0 / 3, 0.3)
    signal = np.concatenate([noise, tone, noise])
    out = gate.process(signal)
    t0, t1 = SR, SR + len(tone)
    assert _rms(out[t0:t1]) > 0.9 * _rms(signal[t0:t1])
    tail = slice(len(signal) - SR // 2, None)
    assert _rms(out[tail]) < 0.3 * _rms(signal[tail])


def test_auto_loud_opening_does_not_lock_open() -> None:
    # A stream that starts loud must still gate the quiet tail (the floor
    # is a sliding-window minimum, not "min so far").
    gate = _auto_gate()
    rng = np.random.RandomState(4)
    loud = _sine(220.0, 1.0, 0.3)
    quiet = (rng.randn(SR) * 0.001).astype(np.float32)
    out = gate.process(np.concatenate([loud, quiet]))
    head = slice(0, SR // 2)
    tail = slice(len(out) - SR // 2, None)
    assert _rms(out[head]) > 0.9 * _rms(loud[head])
    assert _rms(out[tail]) < 0.3 * _rms(quiet[len(quiet) - SR // 2 :])


def test_auto_sustained_loud_tone_stays_open() -> None:
    # A tone longer than the 1s floor window contaminates the floor with
    # its own level; the -30dBFS open guard must keep it passing.
    gate = _auto_gate()
    tone = _sine(220.0, 2.5, 0.3)
    out = gate.process(tone)
    settled = slice(SR, None)
    assert _rms(out[settled]) > 0.9 * _rms(tone[settled])


def test_auto_sensitivity_margins() -> None:
    # Tone ~+7dB over the noise floor: "high" (+3dB margin) opens,
    # "low" (+10dB margin) stays closed.
    rng = np.random.RandomState(5)
    noise = (rng.randn(2 * SR) * 0.001).astype(np.float32)
    tone = _sine(220.0, 1.0, 0.001 * float(np.sqrt(5.0)))
    ratios = {}
    for sens in ("low", "high"):
        gate = _auto_gate(sens)
        out = gate.process(np.concatenate([noise, tone]))
        ratios[sens] = _rms(out[2 * SR :]) / _rms(tone)
    assert ratios["high"] > 3.0 * ratios["low"]


class _FakePipeline:
    _loaded = True
    device = "cpu"
    sample_rate = 16000
    synthesizer = None
    accelerator_index = None
    stage_times: dict = {}


def test_preprocess_hop_applies_gate() -> None:
    cfg = RealtimeConfig(
        mic_sample_rate=16000,
        output_sample_rate=16000,
        chunk_sec=0.04,
        latency_mode="normal",
        denoise_enabled=False,
        noise_gate_enabled=True,
        noise_gate_threshold_db=-40.0,
        use_sola=False,
    )
    changer = ru.RealtimeVoiceChangerUnified(_FakePipeline(), cfg)
    loud = _sine(220.0, 0.3, 0.3)
    quiet = (np.random.RandomState(2).randn(SR) * 0.001).astype(np.float32)
    out_loud, _ = changer._preprocess_hop(loud)
    out_quiet, _ = changer._preprocess_hop(quiet)
    # Loud: unity after attack; quiet (processed after the loud hop, so
    # after the gate's full close sequence): heavily attenuated.
    assert float(np.mean(np.abs(out_loud[int(SR * 0.02) :]))) > 0.9 * float(
        np.mean(np.abs(loud[int(SR * 0.02) :]))
    )
    tail = slice(int(SR * 0.8), None)
    assert float(np.mean(np.abs(out_quiet[tail]))) < 0.1 * float(
        np.mean(np.abs(quiet[tail]))
    )


if __name__ == "__main__":
    test_loud_signal_passes_through()
    test_quiet_noise_attenuated()
    test_hold_keeps_gate_open_through_short_dip()
    test_release_is_smooth()
    test_threshold_property_updates_levels()
    test_auto_threshold_gates_noise_not_tone()
    test_auto_loud_opening_does_not_lock_open()
    test_auto_sustained_loud_tone_stays_open()
    test_auto_sensitivity_margins()
    test_preprocess_hop_applies_gate()
    print("OK: all noise gate tests passed")
