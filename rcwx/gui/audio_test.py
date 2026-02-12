"""Audio test manager for recording and playback testing."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from rcwx.audio.resample import resample

if TYPE_CHECKING:
    from rcwx.gui.app import RCWXApp

logger = logging.getLogger(__name__)


class AudioTestManager:
    """
    Manages the audio test functionality (3-second record -> convert -> playback).
    """

    def __init__(self, app: RCWXApp):
        """
        Initialize audio test manager.

        Args:
            app: Reference to main application
        """
        self.app = app
        self._test_running = False  # Flag to prevent concurrent tests
        self._conversion_done = threading.Event()
        self._conversion_result: dict = {}
        # Intermediate state for async conversion
        self._pending_audio: np.ndarray | None = None
        self._pending_out_sr: int = 0
        self._pending_debug_dir: Path | None = None

    def run_test(self) -> None:
        """
        Run audio test: record -> convert -> playback.

        NOTE: sounddevice/PortAudio on Windows requires audio streams to be
        created from the main thread. Running sd.rec() or sd.play() in a
        separate thread causes "Invalid sample rate" or WDM-KS errors.
        """
        # Prevent concurrent tests
        if self._test_running:
            logger.warning("Audio test already running, ignoring duplicate call")
            return

        if self.app._is_running:
            self.app.test_status.configure(text="変換中は使用できません", text_color="orange")
            return

        # Set flag before disabling button to prevent race condition
        self._test_running = True

        # Disable button during test
        self.app.test_btn.configure(state="disabled")
        self.app.test_status.configure(text="🔴 録音中...", text_color="#ff6666")
        self.app.update_idletasks()

        try:
            duration = 3.0  # seconds

            # Debug output directory
            debug_dir = Path("debug_audio")
            debug_dir.mkdir(exist_ok=True)

            # Get device settings (auto-detected sample rates)
            input_device = self.app.audio_settings.input_device
            output_device = self.app.audio_settings.output_device
            mic_sr = self.app.audio_settings.input_sample_rate
            out_sr = self.app.audio_settings.output_sample_rate
            process_sr = 16000  # HuBERT/RMVPE expect 16kHz

            # Try to record with device's native rate, fallback to common rates if needed
            # Must run on main thread for Windows compatibility
            common_rates = [mic_sr, 48000, 44100, 16000]
            audio_raw = None
            actual_mic_sr = mic_sr

            for try_sr in common_rates:
                try:
                    # Use device's actual channel count
                    input_channels = self.app.audio_settings.input_channels
                    logger.info(f"Recording: device={input_device}, sr={try_sr}, channels={input_channels}, duration={duration}s")
                    audio_raw = sd.rec(
                        int(duration * try_sr),
                        samplerate=try_sr,
                        channels=input_channels,
                        dtype=np.float32,
                        device=input_device,
                    )
                    sd.wait()

                    # Convert to mono based on channel selection
                    if audio_raw.ndim > 1 and audio_raw.shape[1] > 1:
                        # Stereo input - apply channel selection
                        channel_selection = self.app.audio_settings.get_channel_selection()
                        if channel_selection == "left":
                            audio_raw = audio_raw[:, 0]
                        elif channel_selection == "right":
                            audio_raw = audio_raw[:, 1]
                        else:  # "average"
                            audio_raw = np.mean(audio_raw, axis=1)
                    else:
                        # Mono input
                        audio_raw = audio_raw.flatten()

                    actual_mic_sr = try_sr
                    if try_sr != mic_sr:
                        logger.warning(f"Fallback to {try_sr}Hz (device may not support {mic_sr}Hz)")
                    break
                except Exception as e:
                    if try_sr == common_rates[-1]:  # Last attempt
                        raise
                    logger.warning(f"Failed to record at {try_sr}Hz: {e}, trying next rate...")
                    continue

            if audio_raw is None:
                raise RuntimeError("Failed to record audio with any sample rate")

            logger.info(
                f"Recorded: shape={audio_raw.shape}, min={audio_raw.min():.4f}, max={audio_raw.max():.4f}"
            )

            # Apply input gain
            input_gain_db = self.app.audio_settings.input_gain_db
            if input_gain_db != 0.0:
                gain_linear = 10 ** (input_gain_db / 20)
                audio_raw = audio_raw * gain_linear
                logger.info(f"Applied gain: {input_gain_db:+.0f} dB, max={audio_raw.max():.4f}")

            # Save raw input (with gain applied)
            wavfile.write(debug_dir / "01_input_raw.wav", actual_mic_sr, audio_raw)
            logger.info(f"Saved: debug_audio/01_input_raw.wav ({actual_mic_sr}Hz)")

            # Resample to 16kHz for processing
            audio = audio_raw
            if actual_mic_sr != process_sr:
                audio = resample(audio, actual_mic_sr, process_sr)
                logger.info(
                    f"Resampled to 16kHz: shape={audio.shape}, min={audio.min():.4f}, max={audio.max():.4f}"
                )

            # Save resampled input
            wavfile.write(debug_dir / "02_input_16k.wav", process_sr, audio)
            logger.info(f"Saved: debug_audio/02_input_16k.wav ({process_sr}Hz)")

            # Convert if pipeline is loaded
            if self.app.pipeline is not None:
                self.app.test_status.configure(text="🔄 変換中...", text_color="#66b3ff")
                self.app.update_idletasks()

                # Store state for async completion
                self._pending_audio = audio
                self._pending_out_sr = out_sr
                self._pending_debug_dir = debug_dir

                # Run conversion in background thread
                self._conversion_done.clear()
                self._conversion_result = {"audio": None, "error": None, "model_sr": None}

                thread = threading.Thread(target=self._convert_thread, daemon=True)
                thread.start()

                # Non-blocking poll via after() — return control to mainloop
                self.app.after(100, self._poll_conversion)
                return
            else:
                # No conversion - resample back to output rate for playback
                if process_sr != out_sr:
                    audio = resample(audio, process_sr, out_sr)

            self._finish_playback(audio, out_sr, debug_dir, has_pipeline=False)

        except Exception as e:
            self._handle_error(e)
            self._cleanup()

    def _convert_thread(self) -> None:
        """Run pipeline.infer() in a background thread."""
        import torch

        try:
            audio_tensor = torch.from_numpy(self._pending_audio).float()
            converted = self.app.pipeline.infer(
                audio_tensor,
                pitch_shift=self.app.pitch_control.pitch,
                f0_method=self.app.pitch_control.f0_method,
                index_rate=self.app._get_index_rate(),
                voice_gate_mode=self.app.voice_gate_mode_var.get(),
                energy_threshold=self.app.energy_threshold_slider.get(),
                pre_hubert_pitch_ratio=self.app.pitch_control.pre_hubert_pitch_ratio,
                moe_boost=self.app.pitch_control.moe_boost,
                noise_scale=self.app.pitch_control.noise_scale,
                f0_lowpass_cutoff_hz=self.app.config.inference.f0_lowpass_cutoff_hz,
                enable_octave_flip_suppress=self.app.pitch_control.enable_octave_flip_suppress,
                enable_f0_slew_limit=self.app.pitch_control.enable_f0_slew_limit,
                f0_slew_max_step_st=self.app.pitch_control.f0_slew_max_step_st,
                denoise=self.app.use_denoise_var.get(),
                use_feature_cache=False,
                pad_mode="batch",
            )
            self._conversion_result["audio"] = converted
            self._conversion_result["model_sr"] = self.app.pipeline.sample_rate
        except Exception as e:
            self._conversion_result["error"] = str(e)
        finally:
            self._conversion_done.set()

    def _poll_conversion(self) -> None:
        """Non-blocking poll for conversion completion via after()."""
        if self._conversion_done.is_set():
            self._on_conversion_done()
        else:
            self.app.after(100, self._poll_conversion)

    def _on_conversion_done(self) -> None:
        """Handle conversion result after background thread finishes."""
        try:
            if self._conversion_result.get("error"):
                raise RuntimeError(self._conversion_result["error"])

            audio_converted = self._conversion_result["audio"]
            model_sr = self._conversion_result["model_sr"]
            out_sr = self._pending_out_sr
            debug_dir = self._pending_debug_dir

            # Save converted output at model rate
            wavfile.write(debug_dir / "03_output_model.wav", model_sr, audio_converted)
            logger.info(f"Saved: debug_audio/03_output_model.wav ({model_sr}Hz)")

            # Resample from model rate to output device rate
            if model_sr != out_sr:
                audio = resample(audio_converted, model_sr, out_sr)
            else:
                audio = audio_converted

            self._finish_playback(audio, out_sr, debug_dir, has_pipeline=True)

        except Exception as e:
            self._handle_error(e)
            self._cleanup()

    def _finish_playback(
        self,
        audio: np.ndarray,
        out_sr: int,
        debug_dir: Path,
        *,
        has_pipeline: bool,
    ) -> None:
        """Save final output and play back on main thread."""
        try:
            output_device = self.app.audio_settings.output_device

            # Save final output
            wavfile.write(debug_dir / "04_output_final.wav", out_sr, audio)
            logger.info(f"Saved: debug_audio/04_output_final.wav ({out_sr}Hz)")

            # Playback at output device's native rate (with fallback)
            # Must run on main thread for Windows compatibility
            self.app.test_status.configure(text="🔊 再生中...", text_color="#66ff66")
            self.app.update_idletasks()

            playback_rates = [out_sr, 48000, 44100]
            played = False
            for try_sr in playback_rates:
                try:
                    playback_audio = audio
                    if try_sr != out_sr and not played:
                        playback_audio = resample(audio, out_sr, try_sr)

                    sd.play(playback_audio, samplerate=try_sr, device=output_device)
                    sd.wait()
                    played = True
                    if try_sr != out_sr:
                        logger.warning(f"Playback fallback to {try_sr}Hz (device may not support {out_sr}Hz)")
                    break
                except Exception as e:
                    if try_sr == playback_rates[-1]:
                        raise
                    logger.warning(f"Failed to play at {try_sr}Hz: {e}, trying next rate...")
                    continue

            # Done
            if has_pipeline:
                self.app.test_status.configure(text="✓ 完了 (debug_audio/に保存)", text_color="green")
            else:
                self.app.test_status.configure(text="✓ 完了 (変換なし)", text_color="gray")

        except Exception as e:
            self._handle_error(e)
        finally:
            self._cleanup()

    def _handle_error(self, e: Exception) -> None:
        """Display error message in the GUI."""
        logger.error(f"Audio test failed: {e}")
        error_msg = str(e)

        if "Invalid sample rate" in error_msg or "PaErrorCode -9997" in error_msg:
            self.app.test_status.configure(
                text="エラー: サンプルレート非対応（デバイス設定を確認）",
                text_color="red",
            )
        elif "Invalid number of channels" in error_msg:
            self.app.test_status.configure(
                text="エラー: チャンネル数非対応（デバイス設定を確認）",
                text_color="red",
            )
        else:
            short_msg = error_msg[:50]
            self.app.test_status.configure(text=f"エラー: {short_msg}", text_color="red")

    def _cleanup(self) -> None:
        """Re-enable test button and clear running flag."""
        self._test_running = False
        self.app.test_btn.configure(state="normal")
        self._pending_audio = None
