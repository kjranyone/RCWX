"""
Step-by-Step Processing Analysis

各処理ステップを個別に実行し、中間結果を保存・比較する。
問題箇所を特定するためのデバッグツール。

Usage:
    uv run python tests/test_step_by_step.py
    uv run python tests/test_step_by_step.py --chunk-idx 0 1 2  # 特定のチャンクを解析
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from rcwx.audio.resample import resample, StatefulResampler
from rcwx.audio.crossfade import SOLAState, apply_sola_crossfade
from rcwx.config import RCWXConfig
from rcwx.pipeline.inference import RVCPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class StepOutput:
    """Output from a processing step."""
    name: str
    audio: np.ndarray
    sample_rate: int
    metadata: dict


def load_test_audio(path: Path, target_sr: int = 48000) -> np.ndarray:
    """Load audio file."""
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def split_into_chunks(
    audio: np.ndarray,
    chunk_sec: float,
    context_sec: float,
    sample_rate: int,
) -> list[tuple[np.ndarray, int, int]]:
    """
    Split audio into chunks with context using ChunkBuffer.

    Returns:
        List of (chunk, main_start, main_end) tuples
    """
    from rcwx.audio.buffer import ChunkBuffer

    chunk_samples = int(sample_rate * chunk_sec)
    context_samples = int(sample_rate * context_sec)

    # Use ChunkBuffer for consistent behavior with realtime processing
    buffer = ChunkBuffer(
        chunk_samples=chunk_samples,
        crossfade_samples=0,
        context_samples=context_samples,
        lookahead_samples=0,
    )

    buffer.add_input(audio)

    chunks = []
    chunk_idx = 0

    while buffer.has_chunk():
        chunk = buffer.get_chunk()
        if chunk is None:
            break

        if chunk_idx == 0:
            # First chunk: [reflection | main]
            main_start_in_chunk = context_samples
            main_end_in_chunk = len(chunk)
        else:
            # Subsequent chunks: [context | main]
            main_start_in_chunk = context_samples
            main_end_in_chunk = len(chunk)

        chunks.append((chunk, main_start_in_chunk, main_end_in_chunk))
        chunk_idx += 1

    return chunks


class StepByStepProcessor:
    """Process audio step by step with detailed logging."""

    def __init__(
        self,
        pipeline: RVCPipeline,
        mic_sample_rate: int = 48000,
        output_sample_rate: int = 48000,
        chunk_sec: float = 0.15,
        context_sec: float = 0.10,
        crossfade_sec: float = 0.05,
        f0_method: str = "fcpe",
    ):
        self.pipeline = pipeline
        self.mic_sample_rate = mic_sample_rate
        self.output_sample_rate = output_sample_rate
        self.chunk_sec = chunk_sec
        self.context_sec = context_sec
        self.crossfade_sec = crossfade_sec
        self.f0_method = f0_method

        # Stateful resamplers
        self.input_resampler = StatefulResampler(mic_sample_rate, 16000)
        self.output_resampler = StatefulResampler(pipeline.sample_rate, output_sample_rate)

        # SOLA state
        crossfade_samples = int(output_sample_rate * crossfade_sec)
        self.sola_state = SOLAState.create(crossfade_samples, output_sample_rate)

        # Step outputs storage
        self.step_outputs: list[list[StepOutput]] = []

    def process_chunk(
        self,
        chunk: np.ndarray,
        chunk_idx: int,
        main_start: int,
        main_end: int,
        save_steps: bool = True,
    ) -> tuple[np.ndarray, list[StepOutput]]:
        """
        Process a single chunk with step-by-step logging.

        Returns:
            Tuple of (output_audio, step_outputs)
        """
        steps = []

        # Step 0: Input chunk
        if save_steps:
            steps.append(StepOutput(
                name="0_input",
                audio=chunk.copy(),
                sample_rate=self.mic_sample_rate,
                metadata={"chunk_idx": chunk_idx, "main_start": main_start, "main_end": main_end},
            ))

        logger.info(f"Chunk {chunk_idx}: input len={len(chunk)}, main=[{main_start}:{main_end}]")

        # Step 1: Resample to 16kHz
        chunk_16k = self.input_resampler.resample_chunk(chunk)
        if save_steps:
            steps.append(StepOutput(
                name="1_resample_16k",
                audio=chunk_16k.copy(),
                sample_rate=16000,
                metadata={"original_len": len(chunk), "resampled_len": len(chunk_16k)},
            ))

        logger.info(f"  After resample: len={len(chunk_16k)}")

        # Step 2: RVC Inference
        output_model = self.pipeline.infer(
            chunk_16k,
            input_sr=16000,
            pitch_shift=0,
            f0_method=self.f0_method,
            index_rate=0.0,
            voice_gate_mode="off",
            use_feature_cache=True,
            use_parallel_extraction=True,
            allow_short_input=True,
        )
        if save_steps:
            steps.append(StepOutput(
                name="2_infer",
                audio=output_model.copy(),
                sample_rate=self.pipeline.sample_rate,
                metadata={"model_output_len": len(output_model)},
            ))

        logger.info(f"  After infer: len={len(output_model)}")

        # Step 3: Resample to output rate
        output_resampled = self.output_resampler.resample_chunk(output_model)
        if save_steps:
            steps.append(StepOutput(
                name="3_resample_output",
                audio=output_resampled.copy(),
                sample_rate=self.output_sample_rate,
                metadata={"resampled_len": len(output_resampled)},
            ))

        logger.info(f"  After output resample: len={len(output_resampled)}")

        # Step 4: SOLA crossfade
        context_samples_output = int(self.output_sample_rate * self.context_sec)
        pre_sola_len = len(output_resampled)

        cf_result = apply_sola_crossfade(
            output_resampled,
            self.sola_state,
            wokada_mode=True,
            context_samples=context_samples_output,
        )
        output_final = cf_result.audio

        if save_steps:
            steps.append(StepOutput(
                name="4_sola",
                audio=output_final.copy(),
                sample_rate=self.output_sample_rate,
                metadata={
                    "pre_sola_len": pre_sola_len,
                    "post_sola_len": len(output_final),
                    "sola_offset": cf_result.sola_offset,
                    "correlation": cf_result.correlation,
                },
            ))

        logger.info(f"  After SOLA: len={len(output_final)}, offset={cf_result.sola_offset}, corr={cf_result.correlation:.4f}")

        # Check for discontinuities
        if len(output_final) > 1:
            diff = np.abs(np.diff(output_final))
            max_diff = np.max(diff)
            jumps = np.sum(diff > 0.2)
            if jumps > 0:
                logger.warning(f"  DISCONTINUITIES: {jumps} jumps > 0.2, max_diff={max_diff:.4f}")
            else:
                logger.info(f"  No discontinuities (max_diff={max_diff:.4f})")

        return output_final, steps

    def process_audio(
        self,
        audio: np.ndarray,
        save_steps: bool = True,
        chunk_indices: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, list[list[StepOutput]]]:
        """
        Process entire audio with step-by-step analysis.

        Args:
            audio: Input audio at mic_sample_rate
            save_steps: Whether to save intermediate steps
            chunk_indices: If provided, only process these chunks

        Returns:
            Tuple of (output_audio, all_step_outputs)
        """
        # Clear pipeline cache
        self.pipeline.clear_cache()

        # Reset resamplers
        self.input_resampler = StatefulResampler(self.mic_sample_rate, 16000)
        self.output_resampler = StatefulResampler(self.pipeline.sample_rate, self.output_sample_rate)

        # Reset SOLA state
        crossfade_samples = int(self.output_sample_rate * self.crossfade_sec)
        self.sola_state = SOLAState.create(crossfade_samples, self.output_sample_rate)

        # Split into chunks
        chunks = split_into_chunks(
            audio, self.chunk_sec, self.context_sec, self.mic_sample_rate
        )
        logger.info(f"Split audio into {len(chunks)} chunks")

        outputs = []
        all_steps = []

        for idx, (chunk, main_start, main_end) in enumerate(chunks):
            if chunk_indices is not None and idx not in chunk_indices:
                # Skip but still process to maintain state
                output, _ = self.process_chunk(chunk, idx, main_start, main_end, save_steps=False)
                outputs.append(output)
                continue

            output, steps = self.process_chunk(chunk, idx, main_start, main_end, save_steps=save_steps)
            outputs.append(output)
            all_steps.append(steps)

        # Concatenate outputs
        final_output = np.concatenate(outputs)

        return final_output, all_steps


def save_step_outputs(
    all_steps: list[list[StepOutput]],
    output_dir: Path,
):
    """Save step outputs to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_idx, steps in enumerate(all_steps):
        chunk_dir = output_dir / f"chunk_{chunk_idx:03d}"
        chunk_dir.mkdir(exist_ok=True)

        for step in steps:
            if len(step.audio) > 0:
                wavfile.write(
                    chunk_dir / f"{step.name}.wav",
                    step.sample_rate,
                    (step.audio * 32767).astype(np.int16),
                )

            # Save metadata
            with open(chunk_dir / f"{step.name}_meta.txt", "w") as f:
                for k, v in step.metadata.items():
                    f.write(f"{k}: {v}\n")


def analyze_boundary_artifacts(
    all_steps: list[list[StepOutput]],
    output_sample_rate: int,
) -> dict:
    """Analyze artifacts at chunk boundaries."""
    results = {
        "total_chunks": len(all_steps),
        "boundary_issues": [],
    }

    for i in range(1, len(all_steps)):
        prev_steps = all_steps[i - 1]
        curr_steps = all_steps[i]

        # Get final output of each chunk
        prev_final = next((s for s in reversed(prev_steps) if s.name.startswith("4_")), None)
        curr_final = next((s for s in reversed(curr_steps) if s.name.startswith("4_")), None)

        if prev_final is None or curr_final is None:
            continue

        # Check transition
        if len(prev_final.audio) > 0 and len(curr_final.audio) > 0:
            # Jump at boundary
            boundary_jump = abs(prev_final.audio[-1] - curr_final.audio[0])

            if boundary_jump > 0.1:
                results["boundary_issues"].append({
                    "chunk_idx": i,
                    "jump": float(boundary_jump),
                    "prev_tail": float(prev_final.audio[-1]),
                    "curr_head": float(curr_final.audio[0]),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Step-by-Step Processing Analysis")
    parser.add_argument("--test-file", type=Path, default=Path("sample_data/seki.wav"))
    parser.add_argument("--chunk-sec", type=float, default=0.15)
    parser.add_argument("--context-sec", type=float, default=0.10)
    parser.add_argument("--crossfade-sec", type=float, default=0.05)
    parser.add_argument("--f0-method", choices=["fcpe", "rmvpe"], default="fcpe")
    parser.add_argument("--chunk-idx", type=int, nargs="*", default=None,
                       help="Only save steps for these chunk indices")
    parser.add_argument("--max-duration", type=float, default=3.0)

    args = parser.parse_args()

    output_dir = Path("test_output/step_by_step")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and pipeline
    config = RCWXConfig.load()
    if not config.last_model_path:
        logger.error("No model configured. Run GUI first to select a model.")
        sys.exit(1)

    logger.info(f"Loading model: {config.last_model_path}")
    pipeline = RVCPipeline(
        config.last_model_path,
        device=config.device,
        use_compile=False,
    )
    pipeline.load()

    # Load test audio
    logger.info(f"Loading test audio: {args.test_file}")
    audio = load_test_audio(args.test_file, target_sr=48000)

    # Limit duration
    max_samples = int(48000 * args.max_duration)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        logger.info(f"Trimmed audio to {args.max_duration}s")

    # Create processor
    processor = StepByStepProcessor(
        pipeline,
        chunk_sec=args.chunk_sec,
        context_sec=args.context_sec,
        crossfade_sec=args.crossfade_sec,
        f0_method=args.f0_method,
    )

    # Process
    logger.info("=" * 60)
    logger.info("Starting step-by-step processing")
    logger.info("=" * 60)

    output, all_steps = processor.process_audio(
        audio,
        save_steps=True,
        chunk_indices=args.chunk_idx,
    )

    # Save results
    save_step_outputs(all_steps, output_dir)

    wavfile.write(
        output_dir / "final_output.wav",
        48000,
        (output * 32767).astype(np.int16),
    )
    wavfile.write(
        output_dir / "input.wav",
        48000,
        (audio * 32767).astype(np.int16),
    )

    # Analyze boundaries
    boundary_results = analyze_boundary_artifacts(all_steps, 48000)
    logger.info("=" * 60)
    logger.info("BOUNDARY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total chunks with saved steps: {boundary_results['total_chunks']}")
    logger.info(f"Boundary issues detected: {len(boundary_results['boundary_issues'])}")

    for issue in boundary_results["boundary_issues"]:
        logger.warning(
            f"  Chunk {issue['chunk_idx']}: jump={issue['jump']:.4f} "
            f"(prev_tail={issue['prev_tail']:.4f}, curr_head={issue['curr_head']:.4f})"
        )

    logger.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
