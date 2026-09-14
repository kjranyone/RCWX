"""Input-side noise gate (guitar noise-suppressor style).

A stateful downward expander placed after denoise and before inference.
Residual noise that survives denoising (noisy offices, keyboards, fans) is
attenuated below the configured threshold so HuBERT / F0 never see it —
preventing noise from being mis-converted as speech.

Classic noise-suppressor topology:

* peak/RMS envelope follower (fast attack, slow release)
* hysteresis state machine: opens at ``threshold_db``, closes at
  ``threshold_db - hysteresis_db`` only after ``hold_ms`` below the close
  level (no chatter on signals riding the threshold)
* smoothed gain: fast attack to unity, slow release down to a fixed
  ``range_db`` attenuation floor (not hard silence, so downstream
  resamplers and SOLA never see discontinuities)
"""

from __future__ import annotations

import numpy as np

# Fixed defaults; only the threshold is user-facing (GUI keeps it simple).
ENV_ATTACK_MS = 0.5
ENV_RELEASE_MS = 60.0
GAIN_ATTACK_MS = 2.0
GAIN_RELEASE_MS = 80.0
HOLD_MS = 60.0
HYSTERESIS_DB = 3.0
RANGE_DB = 40.0


def _one_pole_coeff(time_ms: float, sample_rate: int) -> float:
    """One-pole smoothing coefficient for the given time constant."""
    if time_ms <= 0:
        return 0.0
    return float(np.exp(-1.0 / (time_ms / 1000.0 * sample_rate)))


class NoiseGate:
    """Streaming noise gate for fixed- or variable-size 16kHz hops."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold_db: float = -40.0,
        hysteresis_db: float = HYSTERESIS_DB,
        hold_ms: float = HOLD_MS,
        range_db: float = RANGE_DB,
    ) -> None:
        self.sample_rate = sample_rate
        self.hysteresis_db = float(hysteresis_db)
        self.hold_ms = float(hold_ms)
        self.range_db = float(range_db)
        self._env_attack = _one_pole_coeff(ENV_ATTACK_MS, sample_rate)
        self._env_release = _one_pole_coeff(ENV_RELEASE_MS, sample_rate)
        self._gain_attack = _one_pole_coeff(GAIN_ATTACK_MS, sample_rate)
        self._gain_release = _one_pole_coeff(GAIN_RELEASE_MS, sample_rate)
        self._hold_samples = max(1, int(sample_rate * self.hold_ms / 1000))
        self._floor_gain = 10 ** (-self.range_db / 20)
        self.threshold_db = threshold_db  # property setter computes levels
        self.reset()

    @property
    def threshold_db(self) -> float:
        return self._threshold_db

    @threshold_db.setter
    def threshold_db(self, value: float) -> None:
        self._threshold_db = float(value)
        self._open_level = 10 ** (self._threshold_db / 10)  # power (x^2)
        self._close_level = 10 ** ((self._threshold_db - self.hysteresis_db) / 10)

    def reset(self) -> None:
        """Reset envelope / gate / gain state (fresh stream segment)."""
        self._env_power = 0.0
        self._open = False
        self._hold_count = 0
        self._gain = self._floor_gain

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Gate ``audio`` (mono float32); returns the same length."""
        audio = audio.astype(np.float32, copy=False)
        n = len(audio)
        if n == 0:
            return audio

        env_atk = self._env_attack
        env_rel = self._env_release
        g_atk = self._gain_attack
        g_rel = self._gain_release
        open_level = self._open_level
        close_level = self._close_level
        hold_limit = self._hold_samples
        floor = self._floor_gain

        env = self._env_power
        is_open = self._open
        hold = self._hold_count
        gain = self._gain

        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            x = audio[i]
            p = x * x
            if p > env:
                env = env_atk * env + (1.0 - env_atk) * p
            else:
                env = env_rel * env + (1.0 - env_rel) * p

            if not is_open and env >= open_level:
                is_open = True
                hold = 0
            elif is_open:
                if env < close_level:
                    hold += 1
                    if hold >= hold_limit:
                        is_open = False
                        hold = 0
                else:
                    hold = 0

            if is_open:
                target = 1.0
                gain = g_atk * gain + (1.0 - g_atk) * target
            else:
                target = floor
                gain = g_rel * gain + (1.0 - g_rel) * target
            out[i] = x * gain

        self._env_power = env
        self._open = is_open
        self._hold_count = hold
        self._gain = gain
        return out
