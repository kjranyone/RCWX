"""SwiftF0 F0 extraction model wrapper.

SwiftF0 is a lightweight ONNX-based pitch estimator (95.8K params, MIT license)
that runs on CPU via ONNX Runtime. Ideal for parallel extraction with HuBERT
on XPU since it doesn't compete for GPU resources.

Reference: https://github.com/lars76/swift-f0
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SWIFTF0_AVAILABLE = False
try:
    from swift_f0 import SwiftF0 as _SwiftF0Detector
    _SWIFTF0_AVAILABLE = True
except ImportError:
    pass


def is_swiftf0_available() -> bool:
    """Check if SwiftF0 is available."""
    return _SWIFTF0_AVAILABLE


def _iter_voiced_segments(mask: np.ndarray):
    """Yield (start, end) index pairs for contiguous voiced regions."""
    i = 0
    n = int(mask.shape[0])
    while i < n:
        while i < n and not mask[i]:
            i += 1
        start = i
        while i < n and mask[i]:
            i += 1
        end = i
        if start < end:
            yield start, end


def _reflect_frame(audio: np.ndarray, center: int, frame_size: int) -> np.ndarray:
    """Extract frame with reflection padding."""
    n = int(audio.shape[0])
    if frame_size <= 0:
        return np.zeros(0, dtype=np.float32)
    if n <= 0:
        return np.zeros(frame_size, dtype=np.float32)
    if n == 1:
        return np.full(frame_size, float(audio[0]), dtype=np.float32)

    start = center - frame_size // 2
    idx = np.arange(start, start + frame_size, dtype=np.int64)
    period = 2 * n - 2
    idx_mod = np.mod(idx, period)
    idx_reflect = np.where(idx_mod < n, idx_mod, period - idx_mod)
    frame = audio[idx_reflect.astype(np.int64)].astype(np.float32, copy=False)
    frame = frame - float(np.mean(frame))
    return frame


def _normalized_periodicity(frame: np.ndarray, lag: int) -> float:
    """Return normalized auto-correlation score at a given lag."""
    if lag <= 0 or lag >= frame.shape[0] - 1:
        return -1.0
    a = frame[:-lag]
    b = frame[lag:]
    energy = float(np.dot(a, a) * np.dot(b, b))
    if energy < 1e-16:
        return -1.0
    return float(np.dot(a, b) / np.sqrt(energy))


def _reflected_frames(
    audio: np.ndarray,
    frame_count: int,
    frame_size: int,
    hop_length: int,
) -> np.ndarray:
    """Extract all centered analysis frames with one reflection-pad operation."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if frame_count <= 0 or frame_size <= 0:
        return np.empty((0, max(0, frame_size)), dtype=np.float32)
    if audio.size <= 0:
        return np.zeros((frame_count, frame_size), dtype=np.float32)
    if audio.size == 1:
        return np.zeros((frame_count, frame_size), dtype=np.float32)

    half_frame = frame_size // 2
    padded = np.pad(audio, (half_frame, half_frame), mode="reflect")
    centers = ((np.arange(frame_count, dtype=np.float64) + 0.5) * hop_length).astype(
        np.int64
    )
    offsets = np.arange(frame_size, dtype=np.int64)
    frames = padded[centers[:, None] + offsets[None, :]]
    return frames - np.mean(frames, axis=1, keepdims=True)


def _resolve_octave_from_waveform(
    f0: np.ndarray,
    voiced: np.ndarray,
    audio: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    f0_min: float,
    f0_max: float,
    confidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resolve harmonic octave confusions using waveform periodicity + Viterbi.

    SwiftF0 can latch onto higher harmonics on difficult phonation. We treat
    the harmonic multiplier as a hidden state and solve the best state sequence
    with dynamic programming:
    - Emission: normalized auto-correlation at candidate period lag
    - Transition: continuity cost on log-F0 and octave-state jumps
    - Prior: confidence-aware preference to keep the detector's raw state
    """
    if f0.size == 0:
        return f0.astype(np.float32, copy=True)

    resolved = f0.astype(np.float32, copy=True)
    voiced_mask = voiced.astype(np.bool_, copy=False)
    resolved[~voiced_mask] = 0.0
    if not np.any(voiced_mask):
        return resolved

    audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
    if confidence is None or len(confidence) != len(resolved):
        conf = np.full(len(resolved), 0.5, dtype=np.float32)
    else:
        conf = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)

    # 64ms context captures enough cycles down to ~50Hz.
    frame_size = int(round(sample_rate * 0.064))
    frame_size = max(256, frame_size)
    if frame_size % 2 != 0:
        frame_size += 1

    analysis_frames = _reflected_frames(
        audio_np,
        frame_count=len(resolved),
        frame_size=frame_size,
        hop_length=hop_length,
    )

    multipliers = (0.25, 0.5, 1.0, 2.0)
    prior_weight = 0.08
    trans_weight = 0.28
    state_weight = 0.06

    for seg_start, seg_end in _iter_voiced_segments(voiced_mask):
        states: list[list[tuple[float, float, float]]] = []

        for idx in range(seg_start, seg_end):
            raw_f0 = float(max(resolved[idx], 0.0))
            if raw_f0 <= 0.0:
                states.append([(0.0, 1.0, -1.0)])
                continue

            frame = analysis_frames[idx]
            conf_i = float(conf[idx])

            cand: list[tuple[float, float, float]] = []
            for mult in multipliers:
                cand_f0 = raw_f0 * mult
                if cand_f0 < f0_min or cand_f0 > f0_max:
                    continue
                lag = int(round(sample_rate / max(cand_f0, 1e-6)))
                periodicity = _normalized_periodicity(frame, lag)
                prior_penalty = prior_weight * conf_i * abs(np.log2(mult))
                emission = periodicity - prior_penalty
                cand.append((cand_f0, mult, emission))

            if not cand:
                cand = [(raw_f0, 1.0, -1.0)]
            states.append(cand)

        seg_len = len(states)
        if seg_len == 0:
            continue

        dp_prev = np.asarray([s[2] for s in states[0]], dtype=np.float32)
        back_ptrs: list[np.ndarray] = [
            np.full(len(states[0]), -1, dtype=np.int32)
        ]

        for t in range(1, seg_len):
            prev_states = states[t - 1]
            cur_states = states[t]
            dp_cur = np.full(len(cur_states), -1e9, dtype=np.float32)
            back_cur = np.full(len(cur_states), -1, dtype=np.int32)

            for j, (f_j, m_j, e_j) in enumerate(cur_states):
                best_score = -1e9
                best_k = -1
                for k, (f_k, m_k, _) in enumerate(prev_states):
                    if f_j <= 0.0 or f_k <= 0.0:
                        trans_cost = 0.0
                    else:
                        log_step = abs(np.log2(max(f_j, 1e-6) / max(f_k, 1e-6)))
                        state_step = abs(np.log2(max(m_j, 1e-6) / max(m_k, 1e-6)))
                        trans_cost = -(trans_weight * log_step + state_weight * state_step)

                    score = float(dp_prev[k]) + float(e_j) + trans_cost
                    if score > best_score:
                        best_score = score
                        best_k = k

                dp_cur[j] = best_score
                back_cur[j] = best_k

            dp_prev = dp_cur
            back_ptrs.append(back_cur)

        state_idx = int(np.argmax(dp_prev))
        for t in range(seg_len - 1, -1, -1):
            f_sel, _, _ = states[t][state_idx]
            resolved[seg_start + t] = float(f_sel)
            if t > 0:
                prev_idx = int(back_ptrs[t][state_idx])
                state_idx = prev_idx if prev_idx >= 0 else 0

    resolved[~voiced_mask] = 0.0
    return resolved


class SwiftF0:
    """SwiftF0 F0 extraction model.

    A very fast CPU-based alternative using ONNX Runtime.
    Internal hop is 256 samples (16ms @ 16kHz); output is resampled
    to match RCWX's standard 160-sample hop (10ms @ 16kHz, 100fps).
    """

    INTERNAL_HOP = 256  # SwiftF0's native hop length at 16kHz
    TARGET_HOP = 160    # RCWX standard F0 hop (100fps at 16kHz)
    MAX_ORT_THREADS = 4

    def __init__(
        self,
        hop_length: int = 160,
        confidence_threshold: float = 0.35,
    ):
        if not _SWIFTF0_AVAILABLE:
            raise ImportError(
                "swift-f0 is not installed. Install with: pip install swift-f0"
            )

        self.hop_length = hop_length
        self.sample_rate = 16000
        self.confidence_threshold = confidence_threshold

        logger.info(
            f"Loading SwiftF0 (ONNX/CPU, confidence_threshold={confidence_threshold})"
        )
        self.detector = _SwiftF0Detector(confidence_threshold=confidence_threshold)
        self.ort_intra_op_threads = self._configure_onnx_runtime()
        logger.info(
            "SwiftF0 model loaded (ONNX intra-op threads=%d)",
            self.ort_intra_op_threads,
        )

    def _configure_onnx_runtime(self) -> int:
        """Replace swift-f0's single-thread session with a bounded CPU session."""
        threads = min(self.MAX_ORT_THREADS, max(1, os.cpu_count() or 1))
        try:
            import onnxruntime as ort

            detector_module = __import__(
                type(self.detector).__module__, fromlist=[type(self.detector).__name__]
            )
            model_path = Path(detector_module.__file__).with_name("model.onnx")
            if not model_path.is_file():
                raise FileNotFoundError(f"SwiftF0 ONNX model not found: {model_path}")

            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = threads
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.detector.pitch_session = session
            self.detector.pitch_input_name = session.get_inputs()[0].name
        except Exception as exc:
            logger.warning(
                "Could not configure SwiftF0 ONNX session; using dependency defaults: %s",
                exc,
            )
            session = getattr(self.detector, "pitch_session", None)
            if session is not None:
                try:
                    return max(1, int(session.get_session_options().intra_op_num_threads))
                except Exception:
                    pass
            return 1
        return threads

    @torch.no_grad()
    def infer(
        self,
        audio: torch.Tensor,
        threshold: float = 0.35,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
    ) -> torch.Tensor:
        """Extract F0 from audio.

        Args:
            audio: Audio tensor [B, T] or [T] at 16kHz
            threshold: Voicing confidence threshold
            f0_min: Minimum F0 frequency in Hz
            f0_max: Maximum F0 frequency in Hz

        Returns:
            F0 tensor [B, T_frames] where unvoiced frames are 0
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        batch_size = audio.shape[0]
        device = audio.device
        n_samples = audio.shape[1]

        # Expected output frames at RCWX's 160-hop rate
        n_target_frames = n_samples // self.TARGET_HOP

        results = []
        for b in range(batch_size):
            audio_np = audio[b].detach().cpu().float().numpy()

            # Silence detection (same as FCPE)
            if np.max(np.abs(audio_np)) < 1e-6:
                results.append(np.zeros(n_target_frames, dtype=np.float32))
                continue

            # Run SwiftF0 detection
            try:
                result = self.detector.detect_from_array(
                    audio_np, sample_rate=self.sample_rate
                )
            except Exception as e:
                logger.warning(f"SwiftF0 inference failed: {e}")
                results.append(np.zeros(n_target_frames, dtype=np.float32))
                continue

            # Apply voicing mask + frequency range filter
            f0 = result.pitch_hz.copy()
            confidence = (
                np.asarray(result.confidence, dtype=np.float32)
                if hasattr(result, "confidence")
                else None
            )
            voiced = result.voicing & (f0 >= f0_min) & (f0 <= f0_max)

            # Re-apply threshold if different from constructor
            if abs(threshold - self.confidence_threshold) > 1e-6:
                voiced = voiced & (result.confidence >= threshold)

            f0[~voiced] = 0.0

            # Resample from 256-hop frames to 160-hop frames
            n_swift_frames = len(f0)
            if n_swift_frames == 0:
                results.append(np.zeros(n_target_frames, dtype=np.float32))
                continue

            if n_swift_frames == n_target_frames:
                f0_resampled = f0
                voiced_resampled = voiced.astype(np.bool_, copy=False)
                if confidence is not None and len(confidence) == n_swift_frames:
                    conf_resampled = confidence.astype(np.float32, copy=False)
                else:
                    conf_resampled = np.full(n_target_frames, 0.5, dtype=np.float32)
            else:
                # Use linear interpolation via torch for consistency
                f0_t = torch.from_numpy(f0).float().unsqueeze(0).unsqueeze(0)
                f0_resampled_t = torch.nn.functional.interpolate(
                    f0_t, size=n_target_frames, mode="linear", align_corners=False
                )
                f0_resampled = f0_resampled_t.squeeze().numpy()

                if confidence is not None and len(confidence) == n_swift_frames:
                    conf_t = torch.from_numpy(confidence).float().unsqueeze(0).unsqueeze(0)
                    conf_resampled = torch.nn.functional.interpolate(
                        conf_t, size=n_target_frames, mode="linear", align_corners=False
                    ).squeeze().numpy()
                    conf_resampled = np.clip(conf_resampled, 0.0, 1.0).astype(
                        np.float32, copy=False
                    )
                else:
                    conf_resampled = np.full(n_target_frames, 0.5, dtype=np.float32)

                # Zero out frames that were unvoiced in the original
                # (interpolation can smear voiced/unvoiced boundaries)
                if n_target_frames > 0 and n_swift_frames > 0:
                    voiced_t = torch.from_numpy(
                        voiced.astype(np.float32)
                    ).unsqueeze(0).unsqueeze(0)
                    voiced_resampled = torch.nn.functional.interpolate(
                        voiced_t, size=n_target_frames, mode="nearest"
                    ).squeeze().numpy()
                    voiced_resampled = voiced_resampled >= 0.5
                    f0_resampled[~voiced_resampled] = 0.0
                else:
                    voiced_resampled = np.zeros(n_target_frames, dtype=np.bool_)

            # Resolve harmonic tracking errors using waveform periodicity.
            f0_resampled = _resolve_octave_from_waveform(
                f0_resampled,
                voiced_resampled,
                audio_np,
                sample_rate=self.sample_rate,
                hop_length=self.TARGET_HOP,
                f0_min=f0_min,
                f0_max=f0_max,
                confidence=conf_resampled,
            )

            results.append(f0_resampled.astype(np.float32))

        f0_batch = np.stack(results, axis=0)
        return torch.from_numpy(f0_batch).to(device=device)
