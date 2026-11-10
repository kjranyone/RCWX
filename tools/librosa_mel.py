"""librosa 未導入環境向け slaney mel 基底 (LibrosaMelNormals と同一式)。

librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, htk=False, norm='slaney')
の結果に一致する実装。Rust 側 crates/rcwx-train/src/modules/stft_mel.rs の
slaney_mel と突き合わせる基準としても使う。
"""
from __future__ import annotations

import numpy as np


def _hz_to_mel(f):
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - 0.0) / f_sp
    logstep = np.log(6.4) / 27.0
    f = np.asarray(f, dtype=float)
    mel = np.where(f < min_log_hz, f / f_sp, min_log_mel + np.log(f / min_log_hz + 1e-15) / logstep)
    return mel


def _mel_to_hz(m):
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    m = np.asarray(m, dtype=float)
    return np.where(m < min_log_mel, m * f_sp, min_log_hz * np.exp(logstep * (m - min_log_mel)))


def slaney_mel(n_mels, bins, sr, n_fft, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    fmin_mel = _hz_to_mel(fmin)
    fmax_mel = _hz_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    freqs = _mel_to_hz(mels)
    fft_freqs = np.arange(bins) * sr / n_fft
    fdiff = freqs[1:] - freqs[:-1]  # ramps diff
    ramps = freqs[np.newaxis, :] - fft_freqs[:, np.newaxis]  # (bins, n_mels+2)
    lower = -ramps[:, :-2] / fdiff[:-1][np.newaxis, :]
    upper = ramps[:, 2:] / fdiff[1:][np.newaxis, :]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (freqs[2:] - freqs[:-2])
    weights *= enorm[np.newaxis, :]
    return weights.T  # (n_mels, bins)
