"""
Post-processing for voice conversion output:
- High-frequency emphasis (treble boost) for clarity/brightness
- EMA-smoothed RMS normalizer (slow AGC) for volume stabilization
- Lookahead limiter for peak safety
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from dataclasses import dataclass


@dataclass
class PostprocessConfig:
    enabled: bool = True
    treble_boost_db: float = 4.0
    treble_cutoff_hz: float = 2800.0
    limiter_threshold_db: float = -1.0
    limiter_release_ms: float = 80.0
    limiter_lookahead_ms: float = 5.0
    # RMS Normalizer (EMA-smoothed AGC)
    normalizer_enabled: bool = True
    normalizer_target_rms: float = 0.1       # -20dBFS target
    normalizer_ema_alpha: float = 0.15       # ~1s time constant
    normalizer_max_gain_db: float = 12.0     # max 4x boost
    normalizer_min_gain_db: float = -12.0    # max 4x cut


class TrebleBoost:
    def __init__(self, sample_rate: int, config: PostprocessConfig):
        self.sample_rate = sample_rate
        self.config = config
        self._design_filter()

    def _design_filter(self) -> None:
        if self.config.treble_boost_db <= 0:
            self.b, self.a = np.array([1.0]), np.array([1.0])
            self.zi = np.array([0.0])
            return

        nyquist = self.sample_rate / 2
        cutoff = min(self.config.treble_cutoff_hz, nyquist * 0.49)
        gain_db = self.config.treble_boost_db
        gain_linear = 10 ** (gain_db / 20)

        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * cutoff / self.sample_rate
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.9 - 1) + 2)

        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        self.b = np.array([b0, b1, b2], dtype=np.float32) / a0
        self.a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
        self.zi = signal.lfilter_zi(self.b, self.a).astype(np.float32)

    def reset(self) -> None:
        self.zi = signal.lfilter_zi(self.b, self.a).astype(np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if self.config.treble_boost_db <= 0 or len(audio) == 0:
            return audio

        filtered, self.zi = signal.lfilter(self.b, self.a, audio, zi=self.zi * audio[0])
        return filtered.astype(np.float32)


class RmsNormalizer:
    """EMA-smoothed RMS normalizer (slow AGC).

    Measures per-chunk RMS, tracks it with an exponential moving average,
    and applies a smoothly-ramped gain to converge toward a target RMS.
    Silent chunks are passed through without updating the EMA.
    """

    MIN_RMS = 0.005  # below this, treat as silence

    def __init__(self, sample_rate: int, config: PostprocessConfig):
        self.sample_rate = sample_rate
        self.config = config
        self._target_rms = config.normalizer_target_rms
        self._alpha = config.normalizer_ema_alpha
        self._max_gain = 10 ** (config.normalizer_max_gain_db / 20)
        self._min_gain = 10 ** (config.normalizer_min_gain_db / 20)
        self._ema_rms: float = 0.0
        self._prev_gain: float = 1.0
        # 5ms cosine ramp length
        self._ramp_samples = max(1, int(sample_rate * 0.005))

    def reset(self) -> None:
        self._ema_rms = 0.0
        self._prev_gain = 1.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.config.normalizer_enabled or len(audio) == 0:
            return audio

        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Silent chunk: pass through, don't update EMA
        if rms < self.MIN_RMS:
            return audio

        # Update EMA
        if self._ema_rms <= 0.0:
            self._ema_rms = rms  # initialise on first voiced chunk
        else:
            self._ema_rms += self._alpha * (rms - self._ema_rms)

        # Compute gain
        if self._ema_rms > self.MIN_RMS:
            gain = self._target_rms / self._ema_rms
        else:
            gain = 1.0
        gain = float(np.clip(gain, self._min_gain, self._max_gain))

        # Apply with cosine ramp from prev_gain -> gain
        ramp_len = min(self._ramp_samples, len(audio))
        if abs(gain - self._prev_gain) > 1e-6 and ramp_len > 1:
            # cosine interpolation: 0 -> pi maps prev_gain -> gain
            t = np.arange(ramp_len, dtype=np.float32) / ramp_len
            ramp = self._prev_gain + (gain - self._prev_gain) * 0.5 * (1.0 - np.cos(np.pi * t))
            output = audio.copy()
            output[:ramp_len] *= ramp
            output[ramp_len:] *= gain
        else:
            output = audio * gain

        self._prev_gain = gain
        return output.astype(np.float32)


class LookaheadLimiter:
    def __init__(self, sample_rate: int, config: PostprocessConfig):
        self.sample_rate = sample_rate
        self.config = config
        self._envelope_state = 1.0
        self._delay_buffer: np.ndarray | None = None
        self._delay_idx = 0

        lookahead_samples = int(config.limiter_lookahead_ms * sample_rate / 1000)
        self._lookahead_samples = max(1, min(lookahead_samples, 256))
        self._threshold = 10 ** (config.limiter_threshold_db / 20)
        release_samples = config.limiter_release_ms * sample_rate / 1000
        self._release_coeff = np.exp(-1.0 / max(release_samples, 1.0))

    def reset(self) -> None:
        self._envelope_state = 1.0
        self._delay_buffer = None
        self._delay_idx = 0

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        lookahead = self._lookahead_samples
        threshold = self._threshold

        if self._delay_buffer is None:
            self._delay_buffer = np.zeros(lookahead, dtype=np.float32)
            self._delay_idx = 0

        output = np.zeros_like(audio)
        delay_buf = self._delay_buffer
        delay_idx = self._delay_idx
        envelope = self._envelope_state

        for i in range(len(audio)):
            in_sample = audio[i]

            out_sample = delay_buf[delay_idx]
            delay_buf[delay_idx] = in_sample
            delay_idx = (delay_idx + 1) % lookahead

            peak = abs(in_sample)
            if peak > envelope:
                envelope = peak
            else:
                envelope = self._release_coeff * envelope + (1 - self._release_coeff) * peak

            if envelope > threshold:
                gain = threshold / envelope
            else:
                gain = 1.0

            output[i] = out_sample * gain

        self._delay_buffer = delay_buf
        self._delay_idx = delay_idx
        self._envelope_state = envelope

        return output


class Postprocessor:
    def __init__(self, sample_rate: int, config: PostprocessConfig | None = None):
        self.sample_rate = sample_rate
        self.config = config or PostprocessConfig()
        self._treble = TrebleBoost(sample_rate, self.config)
        self._normalizer = RmsNormalizer(sample_rate, self.config)
        self._limiter = LookaheadLimiter(sample_rate, self.config)

    def reset(self) -> None:
        self._treble.reset()
        self._normalizer.reset()
        self._limiter.reset()

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.config.enabled or len(audio) == 0:
            return audio

        output = audio.astype(np.float32)

        if self.config.treble_boost_db > 0:
            output = self._treble.process(output)

        output = self._normalizer.process(output)

        output = self._limiter.process(output)

        return output
