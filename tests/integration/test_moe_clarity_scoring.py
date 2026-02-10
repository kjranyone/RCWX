"""Objective scoring proof for moe voice clarity.

This test builds a reproducible scoring pipeline and verifies that
moe-boosted settings can improve articulation/intonation clarity over
baseline on a speech sample.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torchaudio.utils import _download_asset

from rcwx.audio.resample import resample
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    return audio.astype(np.float32)


def _moving_average_replicate(x: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x_pad, kernel, mode="valid")


def _short_gap_ratio(voiced: np.ndarray, max_gap_frames: int = 5) -> float:
    n = len(voiced)
    i = 0
    short_gap = 0
    voiced_frames = max(1, int(voiced.sum()))
    while i < n:
        if voiced[i]:
            i += 1
            continue
        start = i
        while i < n and not voiced[i]:
            i += 1
        end = i
        gap = end - start
        if (
            gap <= max_gap_frames
            and start > 0
            and end < n
            and voiced[start - 1]
            and voiced[end]
        ):
            short_gap += gap
    return float(short_gap / voiced_frames)


def _score_clarity_from_f0(f0: np.ndarray) -> dict[str, float]:
    voiced = f0 > 0
    if voiced.sum() < 16:
        return {
            "accent": 0.0,
            "continuity": 0.0,
            "floor": 0.0,
            "total": 0.0,
            "contrast_st": 0.0,
            "short_gap_ratio": 1.0,
            "median_f0_hz": 0.0,
        }

    voiced_f0 = np.clip(f0[voiced], 1e-5, 1400.0)
    log2_f0 = np.log2(voiced_f0)
    trend = _moving_average_replicate(log2_f0, window=11)
    dev_st = 12.0 * (log2_f0 - trend)

    contrast_st = float(np.percentile(dev_st, 90) - np.percentile(dev_st, 10))
    accent = 100.0 * np.clip(contrast_st / 5.5, 0.0, 1.0)

    short_gap = _short_gap_ratio(voiced, max_gap_frames=5)
    continuity = 100.0 * np.clip(1.0 - 4.0 * short_gap, 0.0, 1.0)

    p10 = float(np.percentile(voiced_f0, 10))
    med = float(np.median(voiced_f0))
    floor_ratio = p10 / (med + 1e-6)
    floor = 100.0 * np.clip((floor_ratio - 0.28) / 0.55, 0.0, 1.0)

    total = 0.50 * accent + 0.30 * continuity + 0.20 * floor
    return {
        "accent": float(accent),
        "continuity": float(continuity),
        "floor": float(floor),
        "total": float(total),
        "contrast_st": contrast_st,
        "short_gap_ratio": short_gap,
        "median_f0_hz": med,
    }


def _extract_f0_rmvpe(pipeline: RVCPipeline, audio_model_sr: np.ndarray) -> np.ndarray:
    audio_16k = resample(audio_model_sr, pipeline.sample_rate, 16000).astype(np.float32)
    with torch.no_grad():
        t = torch.from_numpy(audio_16k).unsqueeze(0).to(pipeline.device)
        f0 = pipeline.rmvpe.infer(t).squeeze(0).detach().cpu().numpy()
    return f0


def test_moe_clarity_scoring_proof():
    """Prove moe settings improve objective clarity score over baseline."""
    config = RCWXConfig.load()
    model_path = config.last_model_path or "model/kurumi/kurumi.pth"
    model_path = Path(model_path)
    if not model_path.exists():
        logger.info(f"SKIP: model not found: {model_path}")
        return

    sample_path = _download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav")
    sr, wav = wavfile.read(sample_path)
    audio = _to_float32(wav)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    pipeline = RVCPipeline(str(model_path), device=config.device, use_compile=False)
    pipeline.load()
    if pipeline.rmvpe is None:
        logger.info("SKIP: RMVPE is required for objective F0 scoring")
        return

    common = {
        "input_sr": sr,
        "pitch_shift": 8,
        "f0_method": "rmvpe",
        "index_rate": 0.0,
        "pre_hubert_pitch_ratio": 0.15,
        "noise_scale": 0.0,
    }

    candidates = [0.0, 0.30, 0.45, 0.60, 0.75]
    scores: dict[float, dict[str, float]] = {}
    for moe in candidates:
        pipeline.clear_cache()
        output = pipeline.infer(audio, moe_boost=moe, **common)
        f0 = _extract_f0_rmvpe(pipeline, output)
        scores[moe] = _score_clarity_from_f0(f0)
        logger.info(
            "moe=%.2f total=%.2f accent=%.2f continuity=%.2f floor=%.2f contrast=%.2fst sgr=%.4f med=%.2fHz",
            moe,
            scores[moe]["total"],
            scores[moe]["accent"],
            scores[moe]["continuity"],
            scores[moe]["floor"],
            scores[moe]["contrast_st"],
            scores[moe]["short_gap_ratio"],
            scores[moe]["median_f0_hz"],
        )

    baseline = scores[0.0]["total"]
    best_moe = max((m for m in candidates if m > 0.0), key=lambda m: scores[m]["total"])
    best = scores[best_moe]["total"]
    improvement = best - baseline

    report = {
        "baseline_moe": 0.0,
        "best_moe": float(best_moe),
        "baseline_total": float(baseline),
        "best_total": float(best),
        "improvement": float(improvement),
        "scores": {str(k): v for k, v in scores.items()},
    }
    report_path = Path("tests") / "integration" / "moe_clarity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "Best moe=%.2f improves clarity score by %.2f points (%.2f -> %.2f)",
        best_moe,
        improvement,
        baseline,
        best,
    )

    assert improvement >= 2.0, (
        f"Expected moe clarity improvement >= 2.0, got {improvement:.2f} "
        f"(baseline={baseline:.2f}, best={best:.2f}, best_moe={best_moe:.2f})"
    )


if __name__ == "__main__":
    test_moe_clarity_scoring_proof()
