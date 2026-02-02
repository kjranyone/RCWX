"""Audio test manager for recording and playback testing."""

from __future__ import annotations

import logging
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
            output_sr = out_sr  # Default to output device rate
            if self.app.pipeline is not None:
                self.app.test_status.configure(text="🔄 変換中...", text_color="#66b3ff")
                self.app.update_idletasks()

                # Run conversion in background thread to avoid UI freeze
                import threading
                import torch

                conversion_result = {"audio": None, "error": None, "model_sr": None}
                conversion_done = threading.Event()

                def convert_thread():
                    try:
                        audio_tensor = torch.from_numpy(audio).float()
                        converted = self.app.pipeline.infer(
                            audio_tensor,
                            pitch_shift=self.app.pitch_control.pitch,
                            f0_method=self.app.pitch_control.f0_method,
                            index_rate=self.app._get_index_rate(),
                            voice_gate_mode=self.app.voice_gate_mode_var.get(),
                            energy_threshold=self.app.energy_threshold_slider.get(),
                            use_feature_cache=False,
                        )
                        conversion_result["audio"] = converted
                        conversion_result["model_sr"] = self.app.pipeline.sample_rate
                    except Exception as e:
                        conversion_result["error"] = str(e)
                    finally:
                        conversion_done.set()

                thread = threading.Thread(target=convert_thread, daemon=True)
                thread.start()

                # Wait with UI updates
                while not conversion_done.wait(timeout=0.1):
                    self.app.update_idletasks()

                if conversion_result["error"]:
                    raise RuntimeError(conversion_result["error"])

                audio_converted = conversion_result["audio"]
                model_sr = conversion_result["model_sr"]

                # Save converted output at model rate
                wavfile.write(debug_dir / "03_output_model.wav", model_sr, audio_converted)
                logger.info(f"Saved: debug_audio/03_output_model.wav ({model_sr}Hz)")

                # Resample from model rate to output device rate
                if model_sr != out_sr:
                    audio = resample(audio_converted, model_sr, out_sr)
                else:
                    audio = audio_converted
            else:
                # No conversion - resample back to output rate for playback
                if process_sr != out_sr:
                    audio = resample(audio, process_sr, out_sr)

            # Save final output
            wavfile.write(debug_dir / "04_output_final.wav", out_sr, audio)
            logger.info(f"Saved: debug_audio/04_output_final.wav ({out_sr}Hz)")

            # Playback at output device's native rate (with fallback)
            # Must run on main thread for Windows compatibility
            self.app.test_status.configure(text="🔊 再生中...", text_color="#66ff66")
            self.app.update_idletasks()

            # Try playback with different sample rates if needed
            playback_rates = [out_sr, 48000, 44100]
            played = False
            for try_sr in playback_rates:
                try:
                    # Resample if needed for this playback rate
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
                    if try_sr == playback_rates[-1]:  # Last attempt
                        raise
                    logger.warning(f"Failed to play at {try_sr}Hz: {e}, trying next rate...")
                    continue

            # Done
            if self.app.pipeline is not None:
                self.app.test_status.configure(text="✓ 完了 (debug_audio/に保存)", text_color="green")
            else:
                self.app.test_status.configure(text="✓ 完了 (変換なし)", text_color="gray")

        except Exception as e:
            logger.error(f"Audio test failed: {e}")
            error_msg = str(e)

            # Provide helpful error messages
            if "Invalid sample rate" in error_msg or "PaErrorCode -9997" in error_msg:
                self.app.test_status.configure(
                    text="エラー: サンプルレート非対応（デバイス設定を確認）",
                    text_color="red"
                )
            elif "Invalid number of channels" in error_msg:
                self.app.test_status.configure(
                    text="エラー: チャンネル数非対応（デバイス設定を確認）",
                    text_color="red"
                )
            else:
                short_msg = error_msg[:50]
                self.app.test_status.configure(text=f"エラー: {short_msg}", text_color="red")

        finally:
            self._test_running = False
            self.app.test_btn.configure(state="normal")
