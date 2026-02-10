"""File-based audio conversion manager."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

if TYPE_CHECKING:
    from rcwx.gui.app import RCWXApp

logger = logging.getLogger(__name__)


class FileConverter:
    """
    Manages file-based audio conversion (WAV -> RVC -> WAV).

    Handles:
    - File selection
    - Audio conversion
    - Playback
    - Saving converted audio
    """

    def __init__(self, app: RCWXApp):
        """
        Initialize file converter.

        Args:
            app: Reference to main application
        """
        self.app = app
        self._converted_audio: Optional[np.ndarray] = None
        self._converted_sr: int = 48000
        self._test_playback_stream: Optional[sd.OutputStream] = None
        self._playback_position: int = 0
        self._playback_audio: Optional[np.ndarray] = None

    def browse_file(self) -> None:
        """Open file dialog to select a WAV file."""
        filepath = filedialog.askopenfilename(
            title="WAVファイルを選択",
            filetypes=[
                ("WAV files", "*.wav"),
                ("All files", "*.*"),
            ],
        )
        if filepath:
            self.app.test_file_entry.delete(0, "end")
            self.app.test_file_entry.insert(0, filepath)
            self.app.test_status_label.configure(text="")
            # Disable play/save until converted
            self.app.test_play_btn.configure(state="disabled")
            self.app.test_save_btn.configure(state="disabled")
            self._converted_audio = None

    def convert_file(self) -> None:
        """Convert the selected WAV file."""
        filepath = self.app.test_file_entry.get().strip()
        if not filepath:
            self.app.test_status_label.configure(text="ファイルを選択してください", text_color="orange")
            return

        if not Path(filepath).exists():
            self.app.test_status_label.configure(text="ファイルが見つかりません", text_color="red")
            return

        if not self.app.pipeline:
            self.app.test_status_label.configure(
                text="モデルを先に読み込んでください", text_color="red"
            )
            return

        # Disable buttons during conversion
        self.app.test_convert_btn.configure(state="disabled")
        self.app.test_status_label.configure(text="変換中...", text_color="white")

        def convert_thread():
            try:
                # Read WAV file
                sr_in, audio = wavfile.read(filepath)

                # Convert to float32 [-1, 1]
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                elif audio.dtype == np.float64:
                    audio = audio.astype(np.float32)

                # Convert to mono if stereo
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                # Run conversion
                output = self.app.pipeline.infer(
                    audio,
                    input_sr=sr_in,
                    pitch_shift=self.app.pitch_control.pitch,
                    f0_method=self.app.pitch_control.f0_method,
                    index_rate=self.app._get_index_rate(),
                )

                self._converted_audio = output
                self._converted_sr = self.app.pipeline.sample_rate

                # Update UI
                duration = len(output) / self._converted_sr
                self.app.after(0, lambda d=duration: self._on_conversion_done(d))

            except Exception as e:
                logger.error(f"Conversion failed: {e}")
                error_msg = str(e)
                self.app.after(0, lambda msg=error_msg: self._on_conversion_error(msg))

        thread = threading.Thread(target=convert_thread, daemon=True)
        thread.start()

    def _on_conversion_done(self, duration: float) -> None:
        """Called when conversion completes successfully."""
        self.app.test_convert_btn.configure(state="normal")
        self.app.test_play_btn.configure(state="normal")
        self.app.test_save_btn.configure(state="normal")
        self.app.test_status_label.configure(
            text=f"変換完了 ({duration:.1f}秒, {self._converted_sr}Hz)",
            text_color="green",
        )

    def _on_conversion_error(self, error: str) -> None:
        """Called when conversion fails."""
        self.app.test_convert_btn.configure(state="normal")
        self.app.test_status_label.configure(text=f"エラー: {error}", text_color="red")

    def play_audio(self) -> None:
        """Play the converted audio."""
        if self._converted_audio is None:
            return

        # Stop any existing playback
        self.stop_playback()

        # Normalize for playback
        audio = self._converted_audio.copy()
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9

        # Start playback
        self._playback_position = 0
        self._playback_audio = audio

        def callback(outdata, frames, time, status):
            if status:
                logger.warning(f"Playback status: {status}")

            start = self._playback_position
            end = start + frames

            if start >= len(self._playback_audio):
                outdata.fill(0)
                raise sd.CallbackStop()

            chunk = self._playback_audio[start:end]
            if len(chunk) < frames:
                outdata[: len(chunk), 0] = chunk
                outdata[len(chunk) :, 0] = 0
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = chunk

            self._playback_position = end

        try:
            self._test_playback_stream = sd.OutputStream(
                samplerate=self._converted_sr,
                channels=1,
                callback=callback,
                finished_callback=self._on_playback_finished,
            )
            self._test_playback_stream.start()
            self.app.test_play_btn.configure(state="disabled")
            self.app.test_stop_btn.configure(state="normal")
            self.app.test_status_label.configure(text="再生中...", text_color="cyan")
        except Exception as e:
            logger.error(f"Playback failed: {e}")
            self.app.test_status_label.configure(text=f"再生エラー: {e}", text_color="red")

    def stop_playback(self) -> None:
        """Stop audio playback."""
        if self._test_playback_stream is not None:
            try:
                self._test_playback_stream.abort()
                self._test_playback_stream.close()
            except Exception:
                pass
            self._test_playback_stream = None

        self.app.test_play_btn.configure(
            state="normal" if self._converted_audio is not None else "disabled"
        )
        self.app.test_stop_btn.configure(state="disabled")

    def _on_playback_finished(self) -> None:
        """Called when playback finishes."""
        self.app.after(0, self.stop_playback)
        self.app.after(
            0,
            lambda: self.app.test_status_label.configure(
                text=f"変換完了 ({len(self._converted_audio) / self._converted_sr:.1f}秒, {self._converted_sr}Hz)",
                text_color="green",
            ),
        )

    def save_audio(self) -> None:
        """Save the converted audio to a file."""
        if self._converted_audio is None:
            return

        filepath = filedialog.asksaveasfilename(
            title="変換した音声を保存",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
        )
        if filepath:
            try:
                # Normalize and convert to int16
                audio = self._converted_audio.copy()
                max_val = np.abs(audio).max()
                if max_val > 0:
                    audio = audio / max_val * 0.9
                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(filepath, self._converted_sr, audio_int16)
                self.app.test_status_label.configure(
                    text=f"保存完了: {Path(filepath).name}", text_color="green"
                )
            except Exception as e:
                logger.error(f"Save failed: {e}")
                self.app.test_status_label.configure(text=f"保存エラー: {e}", text_color="red")
