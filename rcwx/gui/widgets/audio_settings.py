"""Audio settings widget."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import customtkinter as ctk
import numpy as np

from rcwx.audio.input import list_input_devices
from rcwx.audio.output import list_output_devices

logger = logging.getLogger(__name__)


class AudioSettingsFrame(ctk.CTkFrame):
    """
    Audio device settings widget.

    Allows users to select input/output devices and configure sample rates.
    """

    def __init__(
        self,
        master: ctk.CTk,
        on_settings_changed: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_settings_changed = on_settings_changed

        # Device lists
        self._input_devices: list[dict] = []
        self._output_devices: list[dict] = []

        # Selected values
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self.input_sample_rate: int = 48000  # Default, auto-detected on device change
        self.output_sample_rate: int = 48000  # Default, auto-detected on device change
        self.input_channels: int = 1  # Default, auto-detected on device change
        self.output_channels: int = 1  # Default, auto-detected on device change
        self.chunk_sec: float = 0.35
        self.input_gain_db: float = 0.0  # Input gain in dB

        self._load_device_lists()
        self._setup_ui()
        self._detect_default_sample_rates()

    def _load_device_lists(self) -> None:
        """Load device lists before UI setup."""
        self._input_devices = list_input_devices()
        self._output_devices = list_output_devices()

    def _format_device_name(self, device: dict) -> str:
        """Format device name with driver and channel information."""
        name = device["name"]
        channels = device["channels"]
        hostapi = device.get("hostapi_name", "Unknown")

        # Channel info
        if channels == 1:
            channel_info = "モノラル"
        elif channels == 2:
            channel_info = "ステレオ"
        else:
            channel_info = f"{channels}ch"

        # Format: "Device Name (Driver, Channels)"
        return f"{name} ({hostapi}, {channel_info})"

    def _extract_device_name(self, formatted_name: str) -> str:
        """Extract original device name from formatted name."""
        # Remove driver and channel info suffix like "(WASAPI, ステレオ)", etc.
        import re
        return re.sub(r'\s*\([^)]+\)$', '', formatted_name)

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Input device section
        self.input_label = ctk.CTkLabel(
            self,
            text="入力デバイス",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.input_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 2))

        # Format device names with channel info
        input_names = ["デフォルト"] + [
            self._format_device_name(d) for d in self._input_devices
        ]
        self.input_var = ctk.StringVar(value="デフォルト")
        self.input_var.trace_add("write", lambda *_: self._on_input_change(self.input_var.get()))
        self.input_dropdown = ctk.CTkOptionMenu(
            self,
            variable=self.input_var,
            values=input_names,
            width=300,
        )
        self.input_dropdown.grid(row=1, column=0, padx=10, pady=2, sticky="ew")

        # Output device section
        self.output_label = ctk.CTkLabel(
            self,
            text="出力デバイス",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.output_label.grid(row=2, column=0, sticky="w", padx=10, pady=(8, 2))

        # Format device names with channel info
        output_names = ["デフォルト"] + [
            self._format_device_name(d) for d in self._output_devices
        ]
        self.output_var = ctk.StringVar(value="デフォルト")
        self.output_var.trace_add("write", lambda *_: self._on_output_change(self.output_var.get()))
        self.output_dropdown = ctk.CTkOptionMenu(
            self,
            variable=self.output_var,
            values=output_names,
            width=300,
        )
        self.output_dropdown.grid(row=3, column=0, padx=10, pady=2, sticky="ew")

        # Note: Chunk size is now managed by LatencySettingsFrame
        # Keep chunk_sec and chunk_options for backwards compatibility
        self.chunk_sec: float = 0.35
        self.chunk_options = [
            ("200ms (低遅延/F0なし)", 0.2),
            ("350ms (バランス)", 0.35),
            ("500ms (高品質)", 0.5),
        ]

        # Input level meter section
        self.level_label = ctk.CTkLabel(
            self,
            text="入力レベル",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.level_label.grid(row=4, column=0, sticky="w", padx=10, pady=(8, 2))

        self.level_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.level_frame.grid(row=5, column=0, padx=10, pady=2, sticky="ew")
        self.level_frame.grid_columnconfigure(0, weight=1)

        self.level_bar = ctk.CTkProgressBar(self.level_frame, width=280, height=20)
        self.level_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.level_bar.set(0)

        self.level_value = ctk.CTkLabel(self.level_frame, text="-∞ dB", width=60)
        self.level_value.grid(row=0, column=1)

        # Monitor and loopback buttons
        self.monitor_loopback_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.monitor_loopback_frame.grid(row=6, column=0, padx=10, pady=(2, 5), sticky="ew")

        self.monitor_btn = ctk.CTkButton(
            self.monitor_loopback_frame,
            text="モニター開始",
            width=120,
            command=self._toggle_monitor,
        )
        self.monitor_btn.grid(row=0, column=0, padx=(0, 5))

        self.loopback_btn = ctk.CTkButton(
            self.monitor_loopback_frame,
            text="ループバック",
            width=120,
            command=self._toggle_loopback,
        )
        self.loopback_btn.grid(row=0, column=1)

        # Channel selection section (for stereo devices)
        self.channel_label = ctk.CTkLabel(
            self,
            text="入力チャンネル選択",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.channel_label.grid(row=7, column=0, sticky="w", padx=10, pady=(8, 2))

        self.channel_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.channel_frame.grid(row=8, column=0, padx=10, pady=2, sticky="ew")

        self.channel_var = ctk.StringVar(value="average")
        self.channel_left_radio = ctk.CTkRadioButton(
            self.channel_frame,
            text="左 (L)",
            variable=self.channel_var,
            value="left",
            command=self._on_channel_change,
        )
        self.channel_left_radio.grid(row=0, column=0, padx=(0, 10))

        self.channel_right_radio = ctk.CTkRadioButton(
            self.channel_frame,
            text="右 (R)",
            variable=self.channel_var,
            value="right",
            command=self._on_channel_change,
        )
        self.channel_right_radio.grid(row=0, column=1, padx=(0, 10))

        self.channel_average_radio = ctk.CTkRadioButton(
            self.channel_frame,
            text="両方（平均）",
            variable=self.channel_var,
            value="average",
            command=self._on_channel_change,
        )
        self.channel_average_radio.grid(row=0, column=2)

        # Initially disable if device is mono
        self._update_channel_selection_state()

        # Input gain section
        self.gain_label = ctk.CTkLabel(
            self,
            text="入力ゲイン補正",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.gain_label.grid(row=9, column=0, sticky="w", padx=10, pady=(8, 2))

        self.gain_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.gain_frame.grid(row=10, column=0, padx=10, pady=2, sticky="ew")

        self.gain_slider = ctk.CTkSlider(
            self.gain_frame,
            from_=-12,
            to=24,
            number_of_steps=36,
            width=200,
            command=self._on_gain_change,
        )
        self.gain_slider.set(0)
        self.gain_slider.grid(row=0, column=0, padx=(0, 10))

        self.gain_value_label = ctk.CTkLabel(self.gain_frame, text="0 dB", width=50)
        self.gain_value_label.grid(row=0, column=1)

        # Recommended gain display
        self.recommended_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.recommended_frame.grid(row=11, column=0, padx=10, pady=(2, 5), sticky="ew")

        self.recommended_label = ctk.CTkLabel(
            self.recommended_frame,
            text="推奨: -- dB",
            text_color="gray",
        )
        self.recommended_label.grid(row=0, column=0, sticky="w")

        self.apply_recommended_btn = ctk.CTkButton(
            self.recommended_frame,
            text="推奨値をセット",
            width=100,
            command=self._apply_recommended_gain,
        )
        self.apply_recommended_btn.grid(row=0, column=1, padx=(10, 0))

        # Configure grid
        self.grid_columnconfigure(0, weight=1)

        # Monitor state
        self._monitoring = False
        self._monitor_stream = None
        self._peak_db = -60.0
        self._recommended_gain = 0.0

        # Loopback state
        self._loopback_running = False
        self._loopback_input_stream = None
        self._loopback_output_stream = None
        self._loopback_queue = None

    def _refresh_devices(self) -> None:
        """Refresh all device lists."""
        self._refresh_input_devices()
        self._refresh_output_devices()

    def _refresh_input_devices(self) -> None:
        """Refresh input device list."""
        self._input_devices = list_input_devices()
        input_names = ["デフォルト"] + [
            self._format_device_name(d) for d in self._input_devices
        ]
        self.input_dropdown.configure(values=input_names)

    def _refresh_output_devices(self) -> None:
        """Refresh output device list."""
        self._output_devices = list_output_devices()
        output_names = ["デフォルト"] + [
            self._format_device_name(d) for d in self._output_devices
        ]
        self.output_dropdown.configure(values=output_names)

    def _detect_default_sample_rates(self) -> None:
        """Detect sample rates and channels for default devices."""
        import sounddevice as sd
        try:
            # Get default input device info
            default_input = sd.query_devices(kind="input")
            self.input_sample_rate = int(default_input["default_samplerate"])
            self.input_channels = int(default_input["max_input_channels"])
        except Exception:
            self.input_sample_rate = 48000
            self.input_channels = 1

        try:
            # Get default output device info
            default_output = sd.query_devices(kind="output")
            self.output_sample_rate = int(default_output["default_samplerate"])
            self.output_channels = int(default_output["max_output_channels"])
        except Exception:
            self.output_sample_rate = 48000
            self.output_channels = 1

    def _on_input_change(self, value: str) -> None:
        """Handle input device change."""
        if value == "デフォルト":
            self.input_device = None
            self._detect_default_sample_rates()
        else:
            # Extract original device name (remove channel info)
            device_name = self._extract_device_name(value)

            # デバイスを検索、見つからなければNone
            self.input_device = None
            for device in self._input_devices:
                if device["name"] == device_name:
                    self.input_device = device["index"]
                    self.input_sample_rate = int(device["sample_rate"])
                    self.input_channels = int(device["channels"])
                    break

        # Update channel selection UI state
        self._update_channel_selection_state()

        if self.on_settings_changed:
            self.on_settings_changed()

    def _on_output_change(self, value: str) -> None:
        """Handle output device change."""
        if value == "デフォルト":
            self.output_device = None
            self._detect_default_sample_rates()
        else:
            # Extract original device name (remove channel info)
            device_name = self._extract_device_name(value)

            # デバイスを検索、見つからなければNone
            self.output_device = None
            for device in self._output_devices:
                if device["name"] == device_name:
                    self.output_device = device["index"]
                    self.output_sample_rate = int(device["sample_rate"])
                    self.output_channels = int(device["channels"])
                    break

        if self.on_settings_changed:
            self.on_settings_changed()

    def _toggle_monitor(self) -> None:
        """Toggle input level monitoring."""
        if self._monitoring:
            self._stop_monitor()
        else:
            self._start_monitor()

    def _start_monitor(self) -> None:
        """Start input level monitoring."""
        import sounddevice as sd

        self._monitoring = True
        self.monitor_btn.configure(text="モニター停止", fg_color="#cc3333")

        def audio_callback(indata, frames, time, status):
            if not self._monitoring:
                return

            # Convert to mono based on channel selection
            if indata.ndim > 1 and indata.shape[1] > 1:
                # Stereo input - apply channel selection
                channel_selection = self.get_channel_selection()
                if channel_selection == "left":
                    audio = indata[:, 0]
                elif channel_selection == "right":
                    audio = indata[:, 1]
                else:  # "average"
                    audio = np.mean(indata, axis=1)
            else:
                # Mono input
                audio = indata[:, 0] if indata.ndim > 1 else indata

            # Calculate RMS level
            rms = np.sqrt(np.mean(audio ** 2))
            # Calculate peak level
            peak = np.max(np.abs(audio))
            # Convert to dB (with floor at -60 dB)
            rms_db = 20 * np.log10(max(rms, 1e-6))
            peak_db = 20 * np.log10(max(peak, 1e-6))
            rms_db = max(rms_db, -60)
            peak_db = max(peak_db, -60)
            # Normalize to 0-1 range (-60 to 0 dB)
            level = (rms_db + 60) / 60
            # Update UI from main thread
            self.after(0, lambda l=level, r=rms_db, p=peak_db: self._update_level(l, r, p))

        # Try to start monitoring with device's native rate and channels, fallback if needed
        common_rates = [self.input_sample_rate, 48000, 44100, 16000]
        started = False

        for try_sr in common_rates:
            try:
                blocksize = int(try_sr * 0.1)  # 100ms blocks
                # Use actual device channel count (mono or stereo)
                self._monitor_stream = sd.InputStream(
                    device=self.input_device,
                    channels=self.input_channels,
                    samplerate=try_sr,
                    blocksize=blocksize,
                    callback=audio_callback,
                )
                self._monitor_stream.start()
                started = True
                if try_sr != self.input_sample_rate:
                    # Update the detected sample rate for future use
                    self.input_sample_rate = try_sr
                    self.level_value.configure(text=f"{try_sr}Hz")
                break
            except Exception as e:
                if try_sr == common_rates[-1]:  # Last attempt
                    self._monitoring = False
                    self.monitor_btn.configure(text="モニター開始", fg_color=["#3B8ED0", "#1F6AA5"])
                    self.level_value.configure(text="デバイスエラー")
                continue

    def _stop_monitor(self) -> None:
        """Stop input level monitoring."""
        self._monitoring = False
        if self._monitor_stream:
            self._monitor_stream.stop()
            self._monitor_stream.close()
            self._monitor_stream = None
        self.monitor_btn.configure(text="モニター開始", fg_color=["#3B8ED0", "#1F6AA5"])
        self.level_bar.set(0)
        self.level_value.configure(text="-∞ dB")

    def _update_level(self, level: float, rms_db: float, peak_db: float) -> None:
        """Update level meter display and recommended gain."""
        self.level_bar.set(min(level, 1.0))
        if rms_db <= -60:
            self.level_value.configure(text="-∞ dB")
        else:
            self.level_value.configure(text=f"{rms_db:.0f} dB")

        # Track peak and calculate recommended gain
        # Target peak: -6 dB (leaving headroom)
        self._peak_db = peak_db
        target_peak = -6.0
        self._recommended_gain = target_peak - peak_db

        # Clamp to reasonable range
        self._recommended_gain = max(-12, min(24, self._recommended_gain))

        if peak_db <= -60:
            self.recommended_label.configure(text="推奨: -- dB (信号なし)")
        else:
            self.recommended_label.configure(text=f"推奨: {self._recommended_gain:+.0f} dB (ピーク: {peak_db:.0f} dB)")

    def _on_gain_change(self, value: float) -> None:
        """Handle input gain slider change."""
        self.input_gain_db = round(value)
        self.gain_value_label.configure(text=f"{self.input_gain_db:+.0f} dB")
        if self.on_settings_changed:
            self.on_settings_changed()

    def _apply_recommended_gain(self) -> None:
        """Apply recommended gain value."""
        self.gain_slider.set(self._recommended_gain)
        self._on_gain_change(self._recommended_gain)

    def _on_channel_change(self) -> None:
        """Handle channel selection change."""
        if self.on_settings_changed:
            self.on_settings_changed()

    def _update_channel_selection_state(self) -> None:
        """Enable/disable channel selection based on device channels."""
        if self.input_channels > 1:
            # Stereo device - enable selection
            self.channel_left_radio.configure(state="normal")
            self.channel_right_radio.configure(state="normal")
            self.channel_average_radio.configure(state="normal")
        else:
            # Mono device - disable selection and default to left (only channel)
            self.channel_left_radio.configure(state="disabled")
            self.channel_right_radio.configure(state="disabled")
            self.channel_average_radio.configure(state="disabled")
            self.channel_var.set("left")  # Mono only has one channel

    def get_channel_selection(self) -> str:
        """Get the currently selected channel mode."""
        return self.channel_var.get()

    def stop_monitor(self) -> None:
        """Public method to stop monitoring (called when closing app)."""
        if self._monitoring:
            self._stop_monitor()
        if self._loopback_running:
            self._stop_loopback()

    def _toggle_loopback(self) -> None:
        """Toggle loopback test."""
        if self._loopback_running:
            self._stop_loopback()
        else:
            self._start_loopback()

    def _start_loopback(self) -> None:
        """Start loopback test (input -> output passthrough)."""
        import sounddevice as sd
        import queue

        if self.input_device is None or self.output_device is None:
            logger.warning("Input or output device not selected")
            return

        self._loopback_running = True
        self._loopback_queue = queue.Queue(maxsize=10)
        self.loopback_btn.configure(text="停止中...", fg_color="#cc3333")

        def input_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Loopback input status: {status}")
            try:
                # Apply channel selection
                if indata.ndim > 1 and indata.shape[1] > 1:
                    channel_selection = self.get_channel_selection()
                    if channel_selection == "left":
                        audio = indata[:, 0]
                    elif channel_selection == "right":
                        audio = indata[:, 1]
                    else:  # "average"
                        audio = np.mean(indata, axis=1)
                else:
                    audio = indata[:, 0] if indata.ndim > 1 else indata

                # Apply gain
                if self.input_gain_db != 0.0:
                    gain_linear = 10 ** (self.input_gain_db / 20)
                    audio = audio * gain_linear

                self._loopback_queue.put_nowait(audio.copy())
            except queue.Full:
                pass  # Drop if queue is full

        def output_callback(outdata, frames, time_info, status):
            if status:
                logger.warning(f"Loopback output status: {status}")
            try:
                audio = self._loopback_queue.get_nowait()
                # Ensure correct length
                if len(audio) < len(outdata):
                    # Pad with zeros
                    padded = np.zeros(len(outdata), dtype=np.float32)
                    padded[:len(audio)] = audio
                    audio = padded
                elif len(audio) > len(outdata):
                    # Trim
                    audio = audio[:len(outdata)]

                outdata[:] = audio.reshape(-1, 1)
            except queue.Empty:
                outdata.fill(0)  # Output silence if no data

        try:
            # Use same sample rate for both
            sr = 48000
            blocksize = 2048

            logger.info(f"Starting loopback: input={self.input_device}, output={self.output_device}, sr={sr}, blocksize={blocksize}")

            self._loopback_input_stream = sd.InputStream(
                device=self.input_device,
                channels=self.input_channels,
                samplerate=sr,
                blocksize=blocksize,
                dtype=np.float32,
                callback=input_callback,
            )

            self._loopback_output_stream = sd.OutputStream(
                device=self.output_device,
                channels=1,  # Mono output
                samplerate=sr,
                blocksize=blocksize,
                dtype=np.float32,
                callback=output_callback,
            )

            self._loopback_input_stream.start()
            self._loopback_output_stream.start()

            logger.info("Loopback started successfully")

        except Exception as e:
            logger.error(f"Failed to start loopback: {e}")
            self._stop_loopback()
            self.level_value.configure(text="ループバックエラー")

    def _stop_loopback(self) -> None:
        """Stop loopback test."""
        self._loopback_running = False

        if self._loopback_input_stream:
            try:
                self._loopback_input_stream.stop()
                self._loopback_input_stream.close()
            except Exception as e:
                logger.warning(f"Error stopping loopback input: {e}")
            self._loopback_input_stream = None

        if self._loopback_output_stream:
            try:
                self._loopback_output_stream.stop()
                self._loopback_output_stream.close()
            except Exception as e:
                logger.warning(f"Error stopping loopback output: {e}")
            self._loopback_output_stream = None

        self._loopback_queue = None
        self.loopback_btn.configure(text="ループバック", fg_color=["#3B8ED0", "#1F6AA5"])
        logger.info("Loopback stopped")

    def get_input_device_name(self) -> str:
        """Get the currently selected input device name."""
        return self.input_var.get()

    def get_output_device_name(self) -> str:
        """Get the currently selected output device name."""
        return self.output_var.get()

    def set_input_device(self, name: str) -> None:
        """Set input device by name (for restoring saved settings)."""
        # Find device and use formatted name for GUI
        for device in self._input_devices:
            if device["name"] == name:
                formatted_name = self._format_device_name(device)
                self.input_var.set(formatted_name)
                # _on_input_change will be called via trace
                return
        # If not found, keep default
        logger.warning(f"Input device '{name}' not found, using default")

    def set_output_device(self, name: str) -> None:
        """Set output device by name (for restoring saved settings)."""
        # Find device and use formatted name for GUI
        for device in self._output_devices:
            if device["name"] == name:
                formatted_name = self._format_device_name(device)
                self.output_var.set(formatted_name)
                # _on_output_change will be called via trace
                return
        # If not found, keep default
        logger.warning(f"Output device '{name}' not found, using default")
