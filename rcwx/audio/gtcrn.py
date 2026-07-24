"""GTCRN streaming denoiser (MIT license, CPU ONNX Runtime).

GTCRN (ICASSP 2024) is an ultra-lightweight speech-enhancement model
(23.7K params, 33 MMACs/s) that runs comfortably in real time on CPU, so it
never competes with HuBERT / Synthesizer for the GPU.  It is 16kHz native,
matching the pipeline's denoise stage directly.

Streaming contract (official ``stream/`` example in the GTCRN repository):
  STFT: n_fft=512, hop=256, sqrt-Hann analysis AND synthesis windows.
  ONNX inputs : mix (1, 257, 1, 2)  — one STFT frame as [real, imag]
                conv_cache (2, 1, 16, 16, 33)
                tra_cache  (2, 3, 1, 1, 16)
                inter_cache (2, 1, 33, 16)
  ONNX outputs: enh (1, 257, 1, 2) + the three updated caches.

Two operating modes:

* ``zero_latency=True`` (default, used by the realtime path): the samples
  newer than the last complete STFT hop are emitted SPECULATIVELY — the
  unknown future needed by their frames is fabricated by time-reversal
  (reflection) of the most recent input, the model states are snapshotted,
  the speculative frames are run, and the states are rolled back.  When the
  real samples arrive on the next call, the authoritative stream is
  recomputed from the snapshot (rollback-netcode style).  Added latency is
  ZERO; the cost is ~2 extra model frames per call and a small
  approximation confined to the freshest <=16ms of each call.  With an
  identity model the reconstruction is exact (see tests).

* ``zero_latency=False``: plain streaming with a one-hop overlap-add delay
  plus one hop of FIFO priming — 512 samples total (32ms @ 16kHz).  Higher
  fidelity tail; suited to offline / file conversion.

Reference: https://github.com/Xiaobin-Rong/gtcrn
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from rcwx.downloader import download_gtcrn, get_gtcrn_path

logger = logging.getLogger(__name__)

_ORT_AVAILABLE = False
try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    ort = None

N_FFT = 512
HOP = 256
# Periodic sqrt-Hann: analysis * synthesis sums to exactly 1 at 50% overlap.
_WINDOW = np.sin(np.pi * np.arange(N_FFT) / N_FFT).astype(np.float64)

_DEFAULT_MODELS_DIR = Path.home() / ".cache" / "rcwx" / "models"


def is_gtcrn_available() -> bool:
    """True when ONNX Runtime is importable (model file downloads on demand)."""
    return _ORT_AVAILABLE


def _fabricate_future(recent: np.ndarray, n: int) -> np.ndarray:
    """Continue ``recent`` by time-reversal (reflection) to length ``n``.

    Reflection preserves short-term spectral statistics, which is all the
    mask estimation needs for the speculative tail frames.
    """
    if len(recent) == 0:
        return np.zeros(n, dtype=np.float32)
    r = recent[::-1]
    reps = -(-n // len(r))
    return np.tile(r, reps)[:n].astype(np.float32)


class GTCRNDenoiser:
    """Stateful streaming GTCRN wrapper with arbitrary-hop adaptation."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        zero_latency: bool = True,
    ) -> None:
        if not _ORT_AVAILABLE:
            raise RuntimeError("onnxruntime is not available")
        self._model_path = (
            Path(model_path) if model_path else get_gtcrn_path(_DEFAULT_MODELS_DIR)
        )
        self._zero_latency = zero_latency
        self._session = None
        self.reset()

    def _load(self) -> None:
        if self._session is not None:
            return
        if not self._model_path.exists():
            logger.info("GTCRN model not found, downloading...")
            download_gtcrn(self._model_path.parent.parent)
        opts = ort.SessionOptions()
        # A few single frames per call on the audio path — no fan-out needed.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self._model_path), opts, providers=["CPUExecutionProvider"]
        )
        logger.info(
            f"GTCRN denoiser loaded (CPU, zero_latency={self._zero_latency}): "
            f"{self._model_path}"
        )

    def reset(self) -> None:
        """Reset all streaming state (caches, FIFOs, overlap-add)."""
        self._conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
        self._tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
        self._inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)
        self._carry = np.zeros(HOP, dtype=np.float32)  # previous hop (window tail)
        self._ola = np.zeros(N_FFT, dtype=np.float32)
        self._inbuf = np.zeros(0, dtype=np.float32)
        if self._zero_latency:
            self._outbuf = np.zeros(0, dtype=np.float32)
            self._recent = np.zeros(0, dtype=np.float32)  # last real samples
            self._head = 0  # absolute input position received
            self._emitted = 0  # absolute stream position already returned
            # Finalized (non-speculative) stream.  The first real emission
            # (after frame 1) covers content positions [-HOP, 0).
            self._finalized = np.zeros(0, dtype=np.float32)
            self._fin_start = -HOP  # absolute position of _finalized[0]
        else:
            # Prime one hop of zeros so process() can always return
            # len(input) samples (worst-case FIFO shortfall is HOP-1).
            self._outbuf = np.zeros(HOP, dtype=np.float32)
        # Dry path delayed by the total latency, for strength < 1 blending.
        self._dry_fifo = np.zeros(0 if self._zero_latency else 2 * HOP, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core frame processing
    # ------------------------------------------------------------------

    def _process_hop(self, hop: np.ndarray) -> np.ndarray:
        """Enhance one 256-sample hop; returns 256 finalized output samples.

        The returned samples are final for the window region ONE hop behind
        the newest consumed sample (overlap-add).
        """
        frame = np.concatenate([self._carry, hop]).astype(np.float64)
        self._carry = hop.copy()

        spec = np.fft.rfft(frame * _WINDOW)  # (257,)
        mix = np.empty((1, N_FFT // 2 + 1, 1, 2), dtype=np.float32)
        mix[0, :, 0, 0] = spec.real
        mix[0, :, 0, 1] = spec.imag

        enh, self._conv_cache, self._tra_cache, self._inter_cache = self._session.run(
            None,
            {
                "mix": mix,
                "conv_cache": self._conv_cache,
                "tra_cache": self._tra_cache,
                "inter_cache": self._inter_cache,
            },
        )

        spec_enh = enh[0, :, 0, 0].astype(np.float64) + 1j * enh[0, :, 0, 1]
        y = np.fft.irfft(spec_enh, n=N_FFT) * _WINDOW

        # Overlap-add; the first half of the current frame's span is final
        # once this frame lands (sqrt-Hann pairs satisfy COLA at 50% overlap).
        self._ola += y.astype(np.float32)
        out = self._ola[:HOP].copy()
        self._ola[:-HOP] = self._ola[HOP:]
        self._ola[-HOP:] = 0.0
        return out

    def _snapshot(self):
        return (
            self._conv_cache.copy(),
            self._tra_cache.copy(),
            self._inter_cache.copy(),
            self._carry.copy(),
            self._ola.copy(),
        )

    def _restore(self, snap) -> None:
        (
            self._conv_cache,
            self._tra_cache,
            self._inter_cache,
            self._carry,
            self._ola,
        ) = snap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Denoise ``audio`` (16kHz mono float32); returns the same length.

        ``strength`` < 1.0 crossfades toward the (latency-matched) dry
        signal; values above 1.0 are clamped — GTCRN has no natural
        over-suppression second stage.
        """
        if sample_rate != 16000:
            raise ValueError(f"GTCRN is 16kHz-only, got {sample_rate}")
        self._load()
        strength = max(0.5, min(2.0, float(strength)))
        audio = audio.astype(np.float32, copy=False)

        if self._zero_latency:
            wet = self._process_zero_latency(audio)
            dry = audio  # zero latency: dry is already aligned
        else:
            wet = self._process_buffered(audio)
            self._dry_fifo = np.concatenate([self._dry_fifo, audio])
            dry = self._dry_fifo[: len(audio)]
            self._dry_fifo = self._dry_fifo[len(audio) :]

        if strength < 1.0:
            return (dry * (1.0 - strength) + wet * strength).astype(np.float32)
        return wet.astype(np.float32, copy=False)

    def _process_buffered(self, audio: np.ndarray) -> np.ndarray:
        """Plain streaming path: +512 samples (32ms) total latency."""
        n = len(audio)
        self._inbuf = np.concatenate([self._inbuf, audio])
        outs = []
        while len(self._inbuf) >= HOP:
            outs.append(self._process_hop(self._inbuf[:HOP]))
            self._inbuf = self._inbuf[HOP:]
        if outs:
            self._outbuf = np.concatenate([self._outbuf, *outs])

        if len(self._outbuf) < n:
            # Cannot happen with the one-hop priming invariant; guard anyway.
            self._outbuf = np.concatenate(
                [np.zeros(n - len(self._outbuf), dtype=np.float32), self._outbuf]
            )
        out = self._outbuf[:n]
        self._outbuf = self._outbuf[n:]
        return out

    def _process_zero_latency(self, audio: np.ndarray) -> np.ndarray:
        """Speculative-edge path: zero added latency.

        Position bookkeeping (absolute sample positions):
          head = total input received; P = last complete hop boundary.
          Real stateful frames finalize the stream through P - HOP.
          Speculative frame A completes hop [P, P+HOP) with a fabricated
          future and finalizes [P-HOP, P); speculative frame B (fully
          fabricated) finalizes [P, P+HOP), of which the first head-P
          samples are used.  Model state is rolled back afterwards, so the
          next call's real processing replays history authoritatively.
        """
        n = len(audio)
        self._head += n
        self._inbuf = np.concatenate([self._inbuf, audio])
        self._recent = np.concatenate([self._recent, audio])[-N_FFT:]

        # Real (authoritative) processing of complete hops.
        while len(self._inbuf) >= HOP:
            out = self._process_hop(self._inbuf[:HOP])
            self._inbuf = self._inbuf[HOP:]
            self._finalized = np.concatenate([self._finalized, out])
        # Trim finalized samples that were already emitted.
        drop = min(self._emitted - self._fin_start, len(self._finalized))
        if drop > 0:
            self._finalized = self._finalized[drop:]
            self._fin_start += drop

        P = self._head - len(self._inbuf)  # multiple of HOP
        t = self._head - P  # 0..HOP-1 samples of not-yet-framed input

        # Speculative completion of [P-HOP, head).
        snap = self._snapshot()
        fab = _fabricate_future(self._recent, 2 * HOP)
        seg_a = np.concatenate([self._inbuf, fab])[:HOP]
        out_a = self._process_hop(seg_a)  # finalizes [P-HOP, P)
        out_b = (
            self._process_hop(fab[:HOP]) if t > 0 else None
        )  # finalizes [P, P+HOP)
        self._restore(snap)

        # Assemble [emitted, head).
        pieces = []
        pos = self._emitted
        fin_end = self._fin_start + len(self._finalized)  # == P - HOP
        if pos < fin_end:
            pieces.append(self._finalized[pos - self._fin_start :])
            pos = fin_end
        if pos < P:
            pieces.append(out_a[pos - (P - HOP) :])
            pos = P
        if pos < self._head:
            # pos may already sit inside [P, P+HOP) when the previous call
            # ended mid-hop (re-speculation of the same region).
            pieces.append(out_b[pos - P : self._head - P])
            pos = self._head
        out = (
            np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        )
        self._emitted = self._head
        if len(out) != n:  # defensive: never break the length contract
            logger.warning(f"GTCRN ZL length mismatch: {len(out)} != {n}")
            out = np.pad(out[:n], (0, max(0, n - len(out))))
        return out


_cached_denoiser: Optional[GTCRNDenoiser] = None
_cache_lock = threading.Lock()


def get_cached_gtcrn() -> GTCRNDenoiser:
    """Return the shared zero-latency GTCRN instance (state lives across calls)."""
    global _cached_denoiser
    with _cache_lock:
        if _cached_denoiser is None:
            _cached_denoiser = GTCRNDenoiser(zero_latency=True)
        return _cached_denoiser
