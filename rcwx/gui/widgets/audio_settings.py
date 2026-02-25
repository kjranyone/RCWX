"""Audio settings widget."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import customtkinter as ctk
import numpy as np

from rcwx.audio.input import list_input_devices, select_channel
from rcwx.audio.output import list_output_devices
from rcwx.audio.stream_base import is_device_on_asio, query_asio_channel_names

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
        self.chunk_sec: float = 0.5
        self.input_gain_db: float = 0.0  # Input gain in dB

        self._load_device_lists()
        self._setup_ui()
        self._detect_default_sample_rates()

    def _load_device_lists(self) -> None:
        """Load device lists before UI setup."""
        # Load all devices (will be filtered by hostapi in GUI)
        self._all_input_devices = list_input_devices()
        self._all_output_devices = list_output_devices()

        # Initialize with default filter (WASAPI)
        self._input_hostapi_filter = "WASAPI"
        self._output_hostapi_filter = "WASAPI"

        # Apply initial filter
        self._input_devices = self._filter_devices_by_hostapi(
            self._all_input_devices, self._input_hostapi_filter
        )
        self._output_devices = self._filter_devices_by_hostapi(
            self._all_output_devices, self._output_hostapi_filter
        )

    def _filter_devices_by_hostapi(self, devices: list[dict], hostapi: str) -> list[dict]:
        """Filter devices by hostapi name."""
        if hostapi == "すべて":
            return devices
        return [dev for dev in devices if dev.get("hostapi_name") == hostapi]

    def _get_available_hostapis(self, devices: list[dict]) -> list[str]:
        """Get list of unique hostapi names from devices."""
        hostapis = set(dev.get("hostapi_name", "Unknown") for dev in devices)
        # Sort: WASAPI, ASIO, DirectSound, MME, WDM-KS, others
        priority = {"WASAPI": 0, "ASIO": 1, "DirectSound": 2, "MME": 3}
        return sorted(hostapis, key=lambda x: (priority.get(x, 99), x))

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
        # Only remove if it contains a known host API name or channel info
        import re
        # Match patterns like "(WASAPI, ステレオ)", "(ASIO, モノラル)", "(MME, 2ch)", etc.
        # These always have a comma separating hostapi and channel info
        pattern = r'\s*\((WASAPI|ASIO|MME|DirectSound|WDM-KS|Unknown),\s*[^)]+\)$'
        return re.sub(pattern, '', formatted_name)

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Input device section
        self.input_label = ctk.CTkLabel(
            self,
            text="入力デバイス",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.input_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 2))

        # Input API filter
        self.input_api_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_api_frame.grid(row=1, column=0, padx=10, pady=(2, 2), sticky="ew")

        ctk.CTkLabel(self.input_api_frame, text="API:", width=40).grid(row=0, column=0, padx=(0, 5))

        # Get available APIs for input
        input_apis = self._get_available_hostapis(self._all_input_devices)
        input_apis_with_all = ["すべて"] + input_apis

        self.input_api_var = ctk.StringVar(value=self._input_hostapi_filter)
        self.input_api_menu = ctk.CTkOptionMenu(
            self.input_api_frame,
            variable=self.input_api_var,
            values=input_apis_with_all,
            width=120,
            command=self._on_input_api_change,
        )
        self.input_api_menu.grid(row=0, column=1)

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
        self.input_dropdown.grid(row=2, column=0, padx=10, pady=2, sticky="ew")

        # Output device section
        self.output_label = ctk.CTkLabel(
            self,
            text="出力デバイス",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.output_label.grid(row=3, column=0, sticky="w", padx=10, pady=(8, 2))

        # Output API filter
        self.output_api_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.output_api_frame.grid(row=4, column=0, padx=10, pady=(2, 2), sticky="ew")

        ctk.CTkLabel(self.output_api_frame, text="API:", width=40).grid(row=0, column=0, padx=(0, 5))

        # Get available APIs for output
        output_apis = self._get_available_hostapis(self._all_output_devices)
        output_apis_with_all = ["すべて"] + output_apis

        self.output_api_var = ctk.StringVar(value=self._output_hostapi_filter)
        self.output_api_menu = ctk.CTkOptionMenu(
            self.output_api_frame,
            variable=self.output_api_var,
            values=output_apis_with_all,
            width=120,
            command=self._on_output_api_change,
        )
        self.output_api_menu.grid(row=0, column=1)

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
        self.output_dropdown.grid(row=5, column=0, padx=10, pady=2, sticky="ew")

        # Output channel selection (for ASIO multi-channel devices)
        self.output_channel_label = ctk.CTkLabel(
            self,
            text="出力チャンネル選択",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.output_channel_label.grid(row=6, column=0, sticky="w", padx=10, pady=(8, 2))

        self.output_channel_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.output_channel_frame.grid(row=7, column=0, padx=10, pady=2, sticky="ew")

        self.output_channel_var = ctk.StringVar(value="auto")
        self.output_channel_dropdown = ctk.CTkOptionMenu(
            self.output_channel_frame,
            variable=self.output_channel_var,
            values=["自動 (Ch 1-2)"],
            width=300,
            command=self._on_output_channel_dropdown_change,
        )
        self.output_channel_dropdown.grid(row=0, column=0)

        # Initially hidden — shown when ASIO output with >2 channels is selected
        self.output_channel_label.grid_remove()
        self.output_channel_frame.grid_remove()

        # Build initial output channel options
        self._update_output_channel_selection_state()

        # Input level meter section
        self.level_label = ctk.CTkLabel(
            self,
            text="入力レベル",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.level_label.grid(row=8, column=0, sticky="w", padx=10, pady=(8, 2))

        self.level_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.level_frame.grid(row=9, column=0, padx=10, pady=2, sticky="ew")
        self.level_frame.grid_columnconfigure(0, weight=1)

        self.level_bar = ctk.CTkProgressBar(self.level_frame, width=280, height=20)
        self.level_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.level_bar.set(0)

        self.level_value = ctk.CTkLabel(self.level_frame, text="-∞ dB", width=60)
        self.level_value.grid(row=0, column=1)

        # Monitor controls
        self.monitor_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.monitor_frame.grid(row=10, column=0, padx=10, pady=(2, 5), sticky="ew")

        self.monitor_btn = ctk.CTkButton(
            self.monitor_frame,
            text="モニター開始",
            width=120,
            command=self._toggle_monitor,
        )
        self.monitor_btn.grid(row=0, column=0, padx=(0, 10))

        self.loopback_var = ctk.BooleanVar(value=False)
        self.loopback_check = ctk.CTkCheckBox(
            self.monitor_frame,
            text="ループバック出力",
            variable=self.loopback_var,
            command=self._on_loopback_toggle,
        )
        self.loopback_check.grid(row=0, column=1)

        # Channel selection section
        self.channel_label = ctk.CTkLabel(
            self,
            text="入力チャンネル選択",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.channel_label.grid(row=11, column=0, sticky="w", padx=10, pady=(8, 2))

        self.channel_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.channel_frame.grid(row=12, column=0, padx=10, pady=2, sticky="ew")

        # Internal value: "auto", "average", "0", "1", "2", ...
        self.channel_var = ctk.StringVar(value="auto")

        self.channel_dropdown = ctk.CTkOptionMenu(
            self.channel_frame,
            variable=self.channel_var,
            values=["自動"],
            width=200,
            command=self._on_channel_dropdown_change,
        )
        self.channel_dropdown.grid(row=0, column=0)

        # Build initial channel options
        self._update_channel_selection_state()

        # Input gain section
        self.gain_label = ctk.CTkLabel(
            self,
            text="入力ゲイン補正",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.gain_label.grid(row=13, column=0, sticky="w", padx=10, pady=(8, 2))

        self.gain_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.gain_frame.grid(row=14, column=0, padx=10, pady=2, sticky="ew")

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
        self.recommended_frame.grid(row=15, column=0, padx=10, pady=(2, 5), sticky="ew")

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
        self._monitor_output_stream = None  # For loopback
        self._monitor_queue = None  # For loopback
        self._peak_db = -60.0
        self._recommended_gain = 0.0

        # Device list cache (for change detection in auto-refresh)
        self._cached_input_names: list[str] | None = None
        self._cached_output_names: list[str] | None = None


    def _on_input_api_change(self, selected_api: str) -> None:
        """Handle input API filter change."""
        self._input_hostapi_filter = selected_api
        self._input_devices = self._filter_devices_by_hostapi(
            self._all_input_devices, selected_api
        )
        self._cached_input_names = None  # force update
        self._refresh_input_dropdown()

    def _on_output_api_change(self, selected_api: str) -> None:
        """Handle output API filter change."""
        self._output_hostapi_filter = selected_api
        self._output_devices = self._filter_devices_by_hostapi(
            self._all_output_devices, selected_api
        )
        self._cached_output_names = None  # force update
        self._refresh_output_dropdown()

    def _refresh_input_dropdown(self) -> None:
        """Refresh input device dropdown with current filter."""
        input_names = ["デフォルト"] + [
            self._format_device_name(d) for d in self._input_devices
        ]
        if input_names != self._cached_input_names:
            self._cached_input_names = input_names
            self.input_dropdown.configure(values=input_names)
            # Reset to default if current selection is not in new list
            if self.input_var.get() not in input_names:
                self.input_var.set("デフォルト")

    def _refresh_output_dropdown(self) -> None:
        """Refresh output device dropdown with current filter."""
        output_names = ["デフォルト"] + [
            self._format_device_name(d) for d in self._output_devices
        ]
        if output_names != self._cached_output_names:
            self._cached_output_names = output_names
            self.output_dropdown.configure(values=output_names)
            # Reset to default if current selection is not in new list
            if self.output_var.get() not in output_names:
                self.output_var.set("デフォルト")

    def _refresh_devices(self) -> None:
        """Refresh all device lists."""
        # Reload all devices
        self._all_input_devices = list_input_devices()
        self._all_output_devices = list_output_devices()

        # Reapply filters
        self._input_devices = self._filter_devices_by_hostapi(
            self._all_input_devices, self._input_hostapi_filter
        )
        self._output_devices = self._filter_devices_by_hostapi(
            self._all_output_devices, self._output_hostapi_filter
        )

        # Update dropdowns
        self._refresh_input_dropdown()
        self._refresh_output_dropdown()

        # Update API filter menus
        input_apis = self._get_available_hostapis(self._all_input_devices)
        self.input_api_menu.configure(values=["すべて"] + input_apis)

        output_apis = self._get_available_hostapis(self._all_output_devices)
        self.output_api_menu.configure(values=["すべて"] + output_apis)

    def start_auto_refresh(self, interval_ms: int = 2000) -> None:
        """Start polling device list at specified interval.

        Only updates dropdown widgets when the device list actually changes,
        so there is no flicker during normal operation.
        """
        self._auto_refresh_id: str | None = None

        def refresh_loop():
            try:
                self._refresh_devices()
            except Exception as e:
                logger.warning(f"Device refresh failed: {e}")
            self._auto_refresh_id = self.after(interval_ms, refresh_loop)

        self._auto_refresh_id = self.after(interval_ms, refresh_loop)
        logger.debug("Device auto-refresh started (interval=%dms)", interval_ms)

    def stop_auto_refresh(self) -> None:
        """Stop auto-refreshing device list."""
        if getattr(self, "_auto_refresh_id", None) is not None:
            self.after_cancel(self._auto_refresh_id)
            self._auto_refresh_id = None

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

        # Update output channel selection UI
        self._update_output_channel_selection_state()

        if self.on_settings_changed:
            self.on_settings_changed()

    def _toggle_monitor(self) -> None:
        """Toggle input level monitoring."""
        if self._monitoring:
            self._stop_monitor()
        else:
            self._start_monitor()

    def _on_loopback_toggle(self) -> None:
        """Handle loopback checkbox toggle during monitoring."""
        if not self._monitoring:
            return  # Not monitoring, nothing to do

        import sounddevice as sd

        enable_loopback = self.loopback_var.get()
        logger.info(f"Loopback toggle: enable={enable_loopback}")

        if enable_loopback:
            # Start loopback output
            if self._monitor_queue is None:
                import queue
                self._monitor_queue = queue.Queue(maxsize=10)

            if self._monitor_output_stream is None and self._monitor_stream is not None:
                try:
                    # Get current monitoring parameters
                    sr = self._monitor_stream.samplerate
                    blocksize = self._monitor_stream.blocksize

                    # Create output callback (same as in _start_monitor)
                    output_callback_count = [0]

                    def output_callback(outdata, frames, time_info, status):
                        output_callback_count[0] += 1
                        if output_callback_count[0] <= 3:
                            logger.info(f"Output callback #{output_callback_count[0]}: queue_size={self._monitor_queue.qsize() if self._monitor_queue else 'None'}, frames={frames}")

                        if self._monitor_queue is None:
                            outdata.fill(0)
                            return
                        try:
                            audio = self._monitor_queue.get_nowait()
                            if output_callback_count[0] <= 3:
                                logger.info(f"  Got audio from queue: len={len(audio)}")
                            if len(audio) < len(outdata):
                                padded = np.zeros(len(outdata), dtype=np.float32)
                                padded[:len(audio)] = audio
                                audio = padded
                            elif len(audio) > len(outdata):
                                audio = audio[:len(outdata)]
                            outdata[:] = audio.reshape(-1, 1)
                        except Exception:
                            if output_callback_count[0] <= 3:
                                logger.info(f"  Queue empty, outputting silence")
                            outdata.fill(0)

                    device_name = "default" if self.output_device is None else str(self.output_device)
                    logger.info(f"Starting loopback output: device={device_name}, sr={sr}Hz, blocksize={blocksize}")

                    self._monitor_output_stream = sd.OutputStream(
                        device=self.output_device,
                        channels=1,
                        samplerate=sr,
                        blocksize=blocksize,
                        dtype=np.float32,
                        callback=output_callback,
                    )
                    self._monitor_output_stream.start()
                    logger.info(f"Loopback output started successfully on {device_name}")
                except Exception as e:
                    logger.error(f"Failed to start loopback output: {e}", exc_info=True)
        else:
            # Stop loopback output
            if self._monitor_output_stream is not None:
                try:
                    self._monitor_output_stream.stop()
                    self._monitor_output_stream.close()
                    logger.info("Loopback output stopped")
                except Exception as e:
                    logger.warning(f"Error stopping loopback output: {e}")
                finally:
                    self._monitor_output_stream = None

            # Clear queue
            if self._monitor_queue is not None:
                import queue
                while not self._monitor_queue.empty():
                    try:
                        self._monitor_queue.get_nowait()
                    except queue.Empty:
                        break
                self._monitor_queue = None

    def _start_monitor(self) -> None:
        """Start input level monitoring."""
        import sounddevice as sd
        import queue

        self._monitoring = True
        self.monitor_btn.configure(text="モニター停止", fg_color="#cc3333")

        # Initialize loopback queue if enabled
        enable_loopback = self.loopback_var.get()
        logger.info(f"Monitor starting: loopback={enable_loopback}, input_device={self.input_device}, output_device={self.output_device}")
        if enable_loopback:
            self._monitor_queue = queue.Queue(maxsize=10)

        input_callback_count = [0]  # Mutable counter for closure

        def audio_callback(indata, frames, time, status):
            input_callback_count[0] += 1
            if not self._monitoring:
                return

            # Convert to mono based on channel selection
            audio = select_channel(indata, self.get_channel_selection())

            # Apply gain
            if self.input_gain_db != 0.0:
                gain_linear = 10 ** (self.input_gain_db / 20)
                audio = audio * gain_linear

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

            # Send to loopback output if enabled
            if enable_loopback and self._monitor_queue is not None:
                try:
                    self._monitor_queue.put_nowait(audio.copy())
                    if input_callback_count[0] <= 3:
                        logger.info(f"Input callback #{input_callback_count[0]}: Added to queue, size={self._monitor_queue.qsize()}")
                except queue.Full:
                    if input_callback_count[0] <= 3:
                        logger.warning(f"Input callback #{input_callback_count[0]}: Queue full, dropping audio")
                    pass  # Drop if queue is full

        output_callback_count = [0]  # Mutable counter for closure

        def output_callback(outdata, frames, time_info, status):
            output_callback_count[0] += 1
            if output_callback_count[0] <= 3:
                logger.info(f"Output callback #{output_callback_count[0]}: queue_size={self._monitor_queue.qsize() if self._monitor_queue else 'None'}, frames={frames}")

            if self._monitor_queue is None:
                outdata.fill(0)
                return
            try:
                audio = self._monitor_queue.get_nowait()
                if output_callback_count[0] <= 3:
                    logger.info(f"  Got audio from queue: len={len(audio)}")
                # Ensure correct length
                if len(audio) < len(outdata):
                    padded = np.zeros(len(outdata), dtype=np.float32)
                    padded[:len(audio)] = audio
                    audio = padded
                elif len(audio) > len(outdata):
                    audio = audio[:len(outdata)]
                outdata[:] = audio.reshape(-1, 1)
            except queue.Empty:
                if output_callback_count[0] <= 3:
                    logger.info(f"  Queue empty, outputting silence")
                outdata.fill(0)  # Output silence if no data

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

                # Start loopback output stream if enabled
                if enable_loopback:
                    device_name = "default" if self.output_device is None else str(self.output_device)
                    logger.info(f"Starting loopback output: device={device_name}, sr={try_sr}Hz, blocksize={blocksize}")
                    try:
                        self._monitor_output_stream = sd.OutputStream(
                            device=self.output_device,  # None is OK - uses system default
                            channels=1,  # Mono output
                            samplerate=try_sr,
                            blocksize=blocksize,
                            dtype=np.float32,
                            callback=output_callback,
                        )
                        self._monitor_output_stream.start()
                        logger.info(f"Loopback output started successfully on {device_name}")
                    except Exception as e:
                        logger.error(f"Failed to start loopback output: {e}", exc_info=True)
                        # Continue without loopback

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
        if self._monitor_output_stream:
            try:
                self._monitor_output_stream.stop()
                self._monitor_output_stream.close()
            except Exception as e:
                logger.warning(f"Error stopping loopback output: {e}")
            self._monitor_output_stream = None
        self._monitor_queue = None
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

    # --- Channel display ↔ internal value mapping ---

    _CHANNEL_DISPLAY_AUTO = "自動"
    _CHANNEL_DISPLAY_AVERAGE = "平均"

    def _channel_display_to_value(self, display: str) -> str:
        """Convert dropdown display text to internal value."""
        if display == self._CHANNEL_DISPLAY_AUTO:
            return "auto"
        if display == self._CHANNEL_DISPLAY_AVERAGE:
            return "average"
        # "Ch 1" or "Ch 1: ASIO Name" → "0"
        if display.startswith("Ch "):
            try:
                # Extract number before optional ": name" suffix
                num_part = display[3:].split(":")[0].strip()
                return str(int(num_part) - 1)
            except ValueError:
                pass
        return "auto"

    def _channel_value_to_display(self, value: str) -> str:
        """Convert internal value to dropdown display text."""
        if value == "auto":
            return self._CHANNEL_DISPLAY_AUTO
        if value == "average":
            return self._CHANNEL_DISPLAY_AVERAGE
        # Legacy compat
        if value == "left":
            return "Ch 1"
        if value == "right":
            return "Ch 2"
        # "0" → "Ch 1", "1" → "Ch 2", ...
        try:
            return f"Ch {int(value) + 1}"
        except ValueError:
            return self._CHANNEL_DISPLAY_AUTO

    def _on_channel_dropdown_change(self, display_value: str) -> None:
        """Handle channel dropdown selection."""
        internal = self._channel_display_to_value(display_value)
        self.channel_var.set(internal)
        # Restore display text (variable trace may overwrite with internal value)
        self.channel_dropdown.set(display_value)
        if self.on_settings_changed:
            self.on_settings_changed()

    def _update_channel_selection_state(self) -> None:
        """Rebuild channel dropdown options based on device channel count."""
        n = self.input_channels
        if n <= 1:
            options = [self._CHANNEL_DISPLAY_AUTO]
            self.channel_dropdown.configure(values=options, state="disabled")
            self.channel_dropdown.set(self._CHANNEL_DISPLAY_AUTO)
            self.channel_var.set("auto")
        else:
            options = [self._CHANNEL_DISPLAY_AUTO]
            # Try to get ASIO channel names
            asio_names: list[str] = []
            if self.input_device is not None:
                asio_names = query_asio_channel_names(self.input_device, "input")
            for i in range(n):
                if i < len(asio_names) and asio_names[i]:
                    options.append(f"Ch {i + 1}: {asio_names[i]}")
                else:
                    options.append(f"Ch {i + 1}")
            options.append(self._CHANNEL_DISPLAY_AVERAGE)
            self.channel_dropdown.configure(values=options, state="normal")
            # Keep current selection if still valid, else reset to auto
            current = self.channel_var.get()
            current_display = self._channel_value_to_display(current)
            # Check if display matches any option (prefix match for ASIO names)
            matched = False
            for opt in options:
                if opt == current_display or opt.startswith(current_display + ":"):
                    self.channel_dropdown.set(opt)
                    matched = True
                    break
            if not matched:
                self.channel_dropdown.set(self._CHANNEL_DISPLAY_AUTO)
                self.channel_var.set("auto")

    def get_channel_selection(self) -> str:
        """Get the currently selected input channel mode."""
        return self.channel_var.get()

    # --- Output channel selection ---

    _OUTPUT_CHANNEL_DISPLAY_AUTO = "自動 (Ch 1-2)"

    def _on_output_channel_dropdown_change(self, display_value: str) -> None:
        """Handle output channel dropdown selection."""
        internal = self._output_channel_display_to_value(display_value)
        self.output_channel_var.set(internal)
        # Restore display text (variable trace may overwrite with internal value)
        self.output_channel_dropdown.set(display_value)
        if self.on_settings_changed:
            self.on_settings_changed()

    def _output_channel_display_to_value(self, display: str) -> str:
        """Convert output channel dropdown text to internal value."""
        if display.startswith("自動"):
            return "auto"
        # "Ch 1-2" or "Ch 1-2: Name A / Name B" → "0,1"
        if display.startswith("Ch "):
            try:
                # Extract "N-M" before optional ": ..."
                pair_part = display[3:].split(":")[0].strip()
                parts = pair_part.split("-")
                if len(parts) == 2:
                    a = int(parts[0]) - 1
                    b = int(parts[1]) - 1
                    return f"{a},{b}"
            except (ValueError, IndexError):
                pass
        return "auto"

    def _output_channel_value_to_display(self, value: str) -> str:
        """Convert internal output channel value to display text."""
        if value == "auto":
            return self._OUTPUT_CHANNEL_DISPLAY_AUTO
        try:
            parts = value.split(",")
            if len(parts) == 2:
                a, b = int(parts[0]), int(parts[1])
                return f"Ch {a + 1}-{b + 1}"
        except (ValueError, IndexError):
            pass
        return self._OUTPUT_CHANNEL_DISPLAY_AUTO

    def _update_output_channel_selection_state(self) -> None:
        """Rebuild output channel dropdown based on output device."""
        n = self.output_channels
        is_asio = (
            self.output_device is not None
            and is_device_on_asio(self.output_device, "output")
        )

        if not is_asio or n <= 2:
            # Hide output channel selection for non-ASIO or <=2ch devices
            self.output_channel_label.grid_remove()
            self.output_channel_frame.grid_remove()
            self.output_channel_var.set("auto")
            return

        # Show output channel selection
        self.output_channel_label.grid()
        self.output_channel_frame.grid()

        # Get ASIO channel names
        asio_names = query_asio_channel_names(self.output_device, "output")

        options = [self._OUTPUT_CHANNEL_DISPLAY_AUTO]
        for i in range(0, n - 1, 2):
            a, b = i, i + 1
            if a < len(asio_names) and b < len(asio_names):
                label = f"Ch {a + 1}-{b + 1}: {asio_names[a]} / {asio_names[b]}"
            else:
                label = f"Ch {a + 1}-{b + 1}"
            options.append(label)

        self.output_channel_dropdown.configure(values=options)

        # Keep current selection if still valid
        current = self.output_channel_var.get()
        current_display = self._output_channel_value_to_display(current)
        matched = False
        for opt in options:
            if opt == current_display or opt.startswith(current_display + ":"):
                self.output_channel_dropdown.set(opt)
                matched = True
                break
        if not matched:
            self.output_channel_dropdown.set(self._OUTPUT_CHANNEL_DISPLAY_AUTO)
            self.output_channel_var.set("auto")

    def get_output_channel_selection(self) -> str:
        """Get the currently selected output channel pair."""
        return self.output_channel_var.get()

    def stop_monitor(self) -> None:
        """Public method to stop monitoring (called when closing app)."""
        if self._monitoring:
            self._stop_monitor()

    def get_input_device_name(self) -> str:
        """Get the currently selected input device name (original, not formatted)."""
        formatted_name = self.input_var.get()
        # "デフォルト" is a special value, return as-is
        if formatted_name == "デフォルト":
            return formatted_name
        return self._extract_device_name(formatted_name)

    def get_output_device_name(self) -> str:
        """Get the currently selected output device name (original, not formatted)."""
        formatted_name = self.output_var.get()
        # "デフォルト" is a special value, return as-is
        if formatted_name == "デフォルト":
            return formatted_name
        return self._extract_device_name(formatted_name)

    def set_input_device(self, name: str) -> None:
        """Set input device by name (for restoring saved settings)."""
        logger.info(f"Restoring input device: '{name}'")

        # "デフォルト" is a special value, set directly
        if name == "デフォルト":
            self.input_var.set("デフォルト")
            return

        # Extract original device name (in case formatted name was saved)
        original_name = self._extract_device_name(name)
        logger.info(f"Extracted device name: '{original_name}'")
        logger.info(f"Available devices: {[d['name'] for d in self._input_devices]}")

        # Find device and use formatted name for GUI
        for device in self._input_devices:
            if device["name"] == original_name:
                formatted_name = self._format_device_name(device)
                logger.info(f"Found device, setting to: '{formatted_name}'")
                self.input_var.set(formatted_name)
                # _on_input_change will be called via trace
                return
        # If not found, keep default
        logger.warning(f"Input device '{original_name}' (from '{name}') not found in {len(self._input_devices)} devices, using default")

    def set_output_device(self, name: str) -> None:
        """Set output device by name (for restoring saved settings)."""
        logger.info(f"Restoring output device: '{name}'")

        # "デフォルト" is a special value, set directly
        if name == "デフォルト":
            self.output_var.set("デフォルト")
            return

        # Extract original device name (in case formatted name was saved)
        original_name = self._extract_device_name(name)
        logger.info(f"Extracted device name: '{original_name}'")
        logger.info(f"Available devices: {[d['name'] for d in self._output_devices]}")

        # Find device and use formatted name for GUI
        for device in self._output_devices:
            if device["name"] == original_name:
                formatted_name = self._format_device_name(device)
                logger.info(f"Found device, setting to: '{formatted_name}'")
                self.output_var.set(formatted_name)
                # _on_output_change will be called via trace
                return
        # If not found, keep default
        logger.warning(f"Output device '{original_name}' (from '{name}') not found in {len(self._output_devices)} devices, using default")
