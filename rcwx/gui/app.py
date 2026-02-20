"""Main RCWX GUI application."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import customtkinter as ctk

from rcwx.audio.denoise import is_ml_denoiser_available
from rcwx.config import RCWXConfig
from rcwx.device import get_device, get_device_name
from rcwx.gui.audio_test import AudioTestManager
from rcwx.gui.file_converter import FileConverter
from rcwx.gui.model_loader import ModelLoader
from rcwx.gui.realtime_controller import RealtimeController
from rcwx.gui.widgets.audio_settings import AudioSettingsFrame
from rcwx.gui.widgets.latency_settings import LatencySettingsFrame
from rcwx.gui.widgets.latency_monitor import LatencyMonitor
from rcwx.gui.widgets.model_selector import ModelSelector
from rcwx.gui.widgets.pitch_control import PitchControl
from rcwx.gui.widgets.postprocess_settings import PostprocessSettingsFrame
from rcwx.pipeline.inference import RVCPipeline

# Set PortAudio API preference for Windows (WASAPI for better compatibility)
# This prevents WDM-KS errors with certain audio drivers
if sys.platform == "win32":
    os.environ.setdefault("PA_USE_WASAPI", "1")

logger = logging.getLogger(__name__)


class RCWXApp(ctk.CTk):
    """
    Main RCWX application window.
    """

    def __init__(self):
        super().__init__()

        # Initialization flag to prevent premature config saves
        self._initializing = True

        # Configure window
        self.title("RCWX - RVC Voice Changer")
        self.geometry("800x550")
        self.minsize(800, 500)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Load configuration
        self.config = RCWXConfig.load()

        # Pipeline components
        self.pipeline: Optional[RVCPipeline] = None

        # State
        self._is_running = False
        self._loading = False

        # Compute device info once
        self._device = get_device(self.config.device)
        self._device_name = get_device_name(self._device)

        # Setup UI
        self._setup_ui()

        # Initialize managers (after UI setup, as they need UI references)
        self.audio_test_manager = AudioTestManager(self)
        self.file_converter = FileConverter(self)
        self.model_loader = ModelLoader(self)
        self.realtime_controller = RealtimeController(self)

        # Initialize status bar device display
        self.status_bar.set_device(self._device_name)

        # Load last model if available
        if self.config.last_model_path and Path(self.config.last_model_path).exists():
            self.model_selector.set_model(self.config.last_model_path)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Initialization complete - allow config saves
        self._initializing = False

    def _setup_ui(self) -> None:
        """Setup the main UI layout."""
        # Create tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(5, 3))

        # Add tabs
        self.tab_main = self.tabview.add("メイン")
        self.tab_audio = self.tabview.add("オーディオ")
        self.tab_settings = self.tabview.add("詳細設定")

        # Handle tab change events
        self.tabview.configure(command=self._on_tab_changed)

        # Setup main tab
        self._setup_main_tab()

        # Setup audio tab
        self._setup_audio_tab()

        # Update audio device display in main panel
        self._update_audio_device_display()

        # Setup settings tab
        self._setup_settings_tab()

        # Status bar
        self.status_bar = LatencyMonitor(self, height=40)
        self.status_bar.pack(fill="x", padx=10, pady=(3, 5))

    def _configure_scroll_speed(
        self, scrollable_frame: ctk.CTkScrollableFrame, speed: int = 20
    ) -> None:
        """Configure mouse wheel scroll speed for a CTkScrollableFrame.

        Args:
            scrollable_frame: The scrollable frame to configure
            speed: Multiplier for scroll speed (default: 20, higher = faster)
        """
        canvas = scrollable_frame._parent_canvas

        def _on_mousewheel(event):
            # Scroll by speed * units (negative for natural scrolling direction)
            canvas.yview_scroll(-speed * int(event.delta / 120), "units")
            return "break"  # Prevent event propagation

        # Bind to the canvas
        canvas.bind("<MouseWheel>", _on_mousewheel)
        # Also bind to enter/leave events to ensure scrolling works when hovering
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    def _setup_main_tab(self) -> None:
        """Setup the main tab content with 2-column layout."""
        # Scrollable container
        self.main_scroll = ctk.CTkScrollableFrame(self.tab_main, fg_color="transparent")
        self.main_scroll.pack(fill="both", expand=True)
        self._configure_scroll_speed(self.main_scroll, speed=30)  # Fast scroll speed

        # 2-column container
        self.main_columns = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        self.main_columns.pack(fill="both", expand=True, padx=3, pady=3)
        self.main_columns.grid_columnconfigure(0, weight=1)
        self.main_columns.grid_columnconfigure(1, weight=1)

        # === Left column ===
        self.left_column = ctk.CTkFrame(self.main_columns, fg_color="transparent")
        self.left_column.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)

        # Model selector
        self.model_selector = ModelSelector(
            self.left_column,
            on_model_selected=self._on_model_selected,
            models_dir=self.config.rvc_models_dir,
        )
        self.model_selector.pack(fill="x", pady=(0, 5))

        # Pitch control
        self.pitch_control = PitchControl(
            self.left_column,
            on_pitch_changed=self._on_pitch_changed,
            on_f0_mode_changed=self._on_f0_mode_changed,
            on_f0_method_changed=self._on_f0_method_changed,
            on_pre_hubert_pitch_changed=self._on_pre_hubert_pitch_changed,
            on_moe_boost_changed=self._on_moe_boost_changed,
            on_noise_scale_changed=self._on_noise_scale_changed,
            on_fixed_harmonics_changed=self._on_fixed_harmonics_changed,
            on_octave_flip_suppress_changed=self._on_octave_flip_suppress_changed,
            on_f0_slew_limit_changed=self._on_f0_slew_limit_changed,
            on_f0_slew_max_step_changed=self._on_f0_slew_max_step_changed,
        )
        self.pitch_control.pack(fill="x", pady=(0, 5))

        # Restore saved F0 method
        self.pitch_control.set_f0_method(self.config.inference.f0_method)
        self.pitch_control.set_pitch(self.config.inference.pitch_shift)

        # Restore saved pre-HuBERT pitch setting
        self.pitch_control.set_pre_hubert_pitch_ratio(self.config.inference.pre_hubert_pitch_ratio)
        self.pitch_control.set_moe_boost(self.config.inference.moe_boost)
        self.pitch_control.set_noise_scale(self.config.inference.noise_scale)
        self.pitch_control.set_fixed_harmonics(self.config.inference.fixed_harmonics)
        self.pitch_control.set_enable_octave_flip_suppress(
            self.config.inference.enable_octave_flip_suppress
        )
        self.pitch_control.set_enable_f0_slew_limit(self.config.inference.enable_f0_slew_limit)
        self.pitch_control.set_f0_slew_max_step_st(self.config.inference.f0_slew_max_step_st)

        # Index control
        self.index_frame = ctk.CTkFrame(self.left_column)
        self.index_frame.pack(fill="x", pady=(0, 5))

        self.index_label = ctk.CTkLabel(
            self.index_frame,
            text="■ Index検索",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.index_label.pack(anchor="w", padx=10, pady=(5, 3))

        self.use_index_var = ctk.BooleanVar(value=self.config.inference.use_index)
        self.use_index_cb = ctk.CTkCheckBox(
            self.index_frame,
            text="Indexを使用",
            variable=self.use_index_var,
            command=self._on_index_changed,
        )
        self.use_index_cb.pack(anchor="w", padx=10, pady=3)

        self.index_ratio_frame = ctk.CTkFrame(self.index_frame, fg_color="transparent")
        self.index_ratio_frame.pack(fill="x", padx=10, pady=(0, 5))

        self.index_ratio_label = ctk.CTkLabel(
            self.index_ratio_frame,
            text="Index率:",
            font=ctk.CTkFont(size=11),
        )
        self.index_ratio_label.grid(row=0, column=0, padx=(0, 5))

        self.index_ratio_slider = ctk.CTkSlider(
            self.index_ratio_frame,
            from_=0,
            to=1,
            number_of_steps=20,
            width=120,
            command=self._on_index_ratio_changed,
        )
        self.index_ratio_slider.set(self.config.inference.index_ratio)
        self.index_ratio_slider.grid(row=0, column=1, padx=5)

        self.index_ratio_value = ctk.CTkLabel(
            self.index_ratio_frame,
            text=f"{self.config.inference.index_ratio:.2f}",
            width=40,
        )
        self.index_ratio_value.grid(row=0, column=2)

        self.index_status = ctk.CTkLabel(
            self.index_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        )
        self.index_status.pack(anchor="w", padx=10, pady=(0, 3))

        # Noise cancellation control
        self.denoise_frame = ctk.CTkFrame(self.left_column)
        self.denoise_frame.pack(fill="x", pady=(0, 5))

        self.denoise_label = ctk.CTkLabel(
            self.denoise_frame,
            text="■ ノイズキャンセリング",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.denoise_label.pack(anchor="w", padx=10, pady=(5, 3))

        self.use_denoise_var = ctk.BooleanVar(value=self.config.inference.denoise.enabled)
        self.use_denoise_cb = ctk.CTkCheckBox(
            self.denoise_frame,
            text="ノイズキャンセリングを有効化",
            variable=self.use_denoise_var,
            command=self._on_denoise_changed,
        )
        self.use_denoise_cb.pack(anchor="w", padx=10, pady=3)

        # Method selection
        self.denoise_method_frame = ctk.CTkFrame(self.denoise_frame, fg_color="transparent")
        self.denoise_method_frame.pack(fill="x", padx=10, pady=(0, 3))

        self.denoise_method_label = ctk.CTkLabel(
            self.denoise_method_frame,
            text="方式:",
            font=ctk.CTkFont(size=11),
        )
        self.denoise_method_label.grid(row=0, column=0, padx=(0, 5))

        self.denoise_method_var = ctk.StringVar(value=self.config.inference.denoise.method)
        self.denoise_method_menu = ctk.CTkOptionMenu(
            self.denoise_method_frame,
            variable=self.denoise_method_var,
            values=["auto", "ml", "spectral"],
            width=120,
            command=lambda _: self._on_denoise_changed(),
        )
        self.denoise_method_menu.grid(row=0, column=1, padx=5)

        # Status label
        ml_status = "✓ 利用可能" if is_ml_denoiser_available() else "✗ 未インストール"
        self.denoise_status = ctk.CTkLabel(
            self.denoise_frame,
            text=f"ML Denoiser: {ml_status}",
            font=ctk.CTkFont(size=10),
            text_color="green" if is_ml_denoiser_available() else "gray",
        )
        self.denoise_status.pack(anchor="w", padx=10, pady=(0, 5))

        # Voice gate control
        self.voice_gate_frame = ctk.CTkFrame(self.left_column)
        self.voice_gate_frame.pack(fill="x", pady=(0, 5))

        self.voice_gate_label = ctk.CTkLabel(
            self.voice_gate_frame,
            text="■ Voice Gate",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.voice_gate_label.pack(anchor="w", padx=10, pady=(5, 3))

        self.voice_gate_mode_frame = ctk.CTkFrame(self.voice_gate_frame, fg_color="transparent")
        self.voice_gate_mode_frame.pack(fill="x", padx=10, pady=(0, 3))

        self.voice_gate_mode_label = ctk.CTkLabel(
            self.voice_gate_mode_frame,
            text="モード:",
            font=ctk.CTkFont(size=11),
        )
        self.voice_gate_mode_label.grid(row=0, column=0, padx=(0, 5))

        self.voice_gate_mode_var = ctk.StringVar(value=self.config.inference.voice_gate_mode)
        self.voice_gate_mode_menu = ctk.CTkOptionMenu(
            self.voice_gate_mode_frame,
            variable=self.voice_gate_mode_var,
            values=["off", "strict", "expand", "energy"],
            width=120,
            command=lambda _: self._on_voice_gate_mode_changed(),
        )
        self.voice_gate_mode_menu.grid(row=0, column=1, padx=5)

        # Energy threshold slider (only visible when mode is "energy")
        self.energy_threshold_frame = ctk.CTkFrame(self.voice_gate_frame, fg_color="transparent")
        self.energy_threshold_frame.pack(fill="x", padx=10, pady=(3, 0))

        self.energy_threshold_label = ctk.CTkLabel(
            self.energy_threshold_frame,
            text="閾値:",
            font=ctk.CTkFont(size=11),
        )
        self.energy_threshold_label.grid(row=0, column=0, padx=(0, 5))

        self.energy_threshold_slider = ctk.CTkSlider(
            self.energy_threshold_frame,
            from_=0.01,
            to=0.20,
            number_of_steps=19,
            width=120,
            command=self._on_energy_threshold_changed,
        )
        self.energy_threshold_slider.set(self.config.inference.energy_threshold)
        self.energy_threshold_slider.grid(row=0, column=1, padx=5)

        self.energy_threshold_value = ctk.CTkLabel(
            self.energy_threshold_frame,
            text=f"{self.config.inference.energy_threshold:.2f}",
            width=40,
        )
        self.energy_threshold_value.grid(row=0, column=2)

        # Show/hide based on current mode
        if self.config.inference.voice_gate_mode != "energy":
            self.energy_threshold_frame.pack_forget()

        self.voice_gate_desc = ctk.CTkLabel(
            self.voice_gate_frame,
            text="off=全通過 / strict=F0のみ / expand=破裂音対応 / energy=エネルギー併用",
            font=ctk.CTkFont(size=9),
            text_color="gray",
        )
        self.voice_gate_desc.pack(anchor="w", padx=10, pady=(0, 5))

        # === Right column ===
        self.right_column = ctk.CTkFrame(self.main_columns, fg_color="transparent")
        self.right_column.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)

        # Audio device info section
        self.device_frame = ctk.CTkFrame(self.right_column)
        self.device_frame.pack(fill="x", pady=(0, 5))

        self.device_section_label = ctk.CTkLabel(
            self.device_frame,
            text="■ オーディオデバイス",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.device_section_label.pack(anchor="w", padx=10, pady=(5, 3))

        # Input device (microphone)
        self.mic_label = ctk.CTkLabel(
            self.device_frame,
            text="🎤 デフォルト",
            font=ctk.CTkFont(size=11),
        )
        self.mic_label.pack(anchor="w", padx=15, pady=(0, 2))

        # Output device (speaker)
        self.speaker_label = ctk.CTkLabel(
            self.device_frame,
            text="🔊 デフォルト",
            font=ctk.CTkFont(size=11),
        )
        self.speaker_label.pack(anchor="w", padx=15, pady=(0, 5))

        # Inference device (GPU/CPU)
        self.inference_device_label = ctk.CTkLabel(
            self.device_frame,
            text=f"⚡ {self._device_name}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.inference_device_label.pack(anchor="w", padx=15, pady=(0, 5))

        # Test section
        self.test_frame = ctk.CTkFrame(self.right_column)
        self.test_frame.pack(fill="x", pady=(0, 5))

        self.test_label = ctk.CTkLabel(
            self.test_frame,
            text="■ オーディオテスト",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.test_label.pack(anchor="w", padx=10, pady=(5, 3))

        self.test_btn = ctk.CTkButton(
            self.test_frame,
            text="🎤 テスト (3秒録音→再生)",
            command=self._run_audio_test,
        )
        self.test_btn.pack(fill="x", padx=10, pady=(0, 3))

        self.test_status = ctk.CTkLabel(
            self.test_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.test_status.pack(anchor="w", padx=10, pady=(0, 3))

        # WAV file loop input (inside test frame)
        self.use_wav_input_var = ctk.BooleanVar(value=False)
        self.use_wav_input_cb = ctk.CTkCheckBox(
            self.test_frame,
            text="WAVファイルをループ再生",
            variable=self.use_wav_input_var,
            command=self._on_wav_input_toggled,
        )
        self.use_wav_input_cb.pack(anchor="w", padx=10, pady=(0, 3))

        self.wav_input_file_frame = ctk.CTkFrame(self.test_frame, fg_color="transparent")
        # Initially hidden; shown when checkbox is checked

        self.wav_input_path_var = ctk.StringVar(value="")
        self.wav_input_entry = ctk.CTkEntry(
            self.wav_input_file_frame,
            textvariable=self.wav_input_path_var,
            placeholder_text="WAVファイルを選択...",
        )
        self.wav_input_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.wav_input_browse_btn = ctk.CTkButton(
            self.wav_input_file_frame,
            text="参照",
            width=50,
            command=self._browse_wav_input,
        )
        self.wav_input_browse_btn.pack(side="right")

        # Start/Stop button
        self.control_frame = ctk.CTkFrame(self.right_column)
        self.control_frame.pack(fill="x", pady=(0, 5))

        self.start_btn = ctk.CTkButton(
            self.control_frame,
            text="▶ 開始",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=60,
            command=self._toggle_running,
        )
        self.start_btn.pack(fill="x", padx=10, pady=5)

    def _setup_audio_tab(self) -> None:
        """Setup the audio settings tab."""
        # Scrollable container
        self.audio_scroll = ctk.CTkScrollableFrame(self.tab_audio, fg_color="transparent")
        self.audio_scroll.pack(fill="both", expand=True)
        self._configure_scroll_speed(self.audio_scroll, speed=30)  # Fast scroll speed

        self.audio_settings = AudioSettingsFrame(
            self.audio_scroll,
            on_settings_changed=self._on_audio_settings_changed,
        )
        self.audio_settings.pack(fill="x", padx=10, pady=5)

        # Post-processing settings (treble boost + limiter)
        self.postprocess_settings = PostprocessSettingsFrame(
            self.audio_scroll,
            config=self.config.inference.postprocess,
            on_settings_changed=self._on_postprocess_settings_changed,
        )
        self.postprocess_settings.pack(fill="x", padx=10, pady=5)

        # Latency settings (mode selection + advanced controls)
        self.latency_settings = LatencySettingsFrame(
            self.audio_scroll,
            on_settings_changed=self._on_latency_settings_changed,
        )
        self.latency_settings.pack(fill="x", padx=10, pady=5)

        # Restore saved latency settings
        self._restore_latency_settings()

        # Restore saved audio settings
        saved_gain = self.config.audio.input_gain_db
        if saved_gain != 0.0:
            self.audio_settings.gain_slider.set(saved_gain)
            self.audio_settings.input_gain_db = saved_gain
            self.audio_settings.gain_value_label.configure(text=f"{saved_gain:+.0f} dB")

        # Restore saved channel selection
        saved_channel = self.config.audio.input_channel_selection
        self.audio_settings.channel_var.set(saved_channel)

        # Restore saved API filters and devices
        # Important: Set API filter first without triggering device reset
        if hasattr(self.config.audio, "input_hostapi_filter"):
            # Set filter directly and update device list only
            self.audio_settings._input_hostapi_filter = self.config.audio.input_hostapi_filter
            self.audio_settings.input_api_var.set(self.config.audio.input_hostapi_filter)
            self.audio_settings._input_devices = self.audio_settings._filter_devices_by_hostapi(
                self.audio_settings._all_input_devices, self.config.audio.input_hostapi_filter
            )
            # Update dropdown without resetting selection
            input_names = ["デフォルト"] + [
                self.audio_settings._format_device_name(d)
                for d in self.audio_settings._input_devices
            ]
            self.audio_settings.input_dropdown.configure(values=input_names)

        if hasattr(self.config.audio, "output_hostapi_filter"):
            # Set filter directly and update device list only
            self.audio_settings._output_hostapi_filter = self.config.audio.output_hostapi_filter
            self.audio_settings.output_api_var.set(self.config.audio.output_hostapi_filter)
            self.audio_settings._output_devices = self.audio_settings._filter_devices_by_hostapi(
                self.audio_settings._all_output_devices, self.config.audio.output_hostapi_filter
            )
            # Update dropdown without resetting selection
            output_names = ["デフォルト"] + [
                self.audio_settings._format_device_name(d)
                for d in self.audio_settings._output_devices
            ]
            self.audio_settings.output_dropdown.configure(values=output_names)

        # Now restore saved device selections (after API filters are set)
        if self.config.audio.input_device_name:
            self.audio_settings.set_input_device(self.config.audio.input_device_name)
        if self.config.audio.output_device_name:
            self.audio_settings.set_output_device(self.config.audio.output_device_name)

    def _setup_settings_tab(self) -> None:
        """Setup the advanced settings tab."""
        # Scrollable container
        self.settings_scroll = ctk.CTkScrollableFrame(self.tab_settings, fg_color="transparent")
        self.settings_scroll.pack(fill="both", expand=True)
        self._configure_scroll_speed(self.settings_scroll, speed=30)  # Fast scroll speed

        # Compile option (not available on Windows - Triton not supported)
        compile_default = False if sys.platform == "win32" else self.config.inference.use_compile
        self.compile_var = ctk.BooleanVar(value=compile_default)
        if sys.platform != "win32":
            self.compile_cb = ctk.CTkCheckBox(
                self.settings_scroll,
                text="torch.compileを使用 (初回起動時にコンパイル)",
                variable=self.compile_var,
                command=self._save_config,
            )
            self.compile_cb.pack(anchor="w", padx=20, pady=5)

        # Device selection
        self.device_label = ctk.CTkLabel(
            self.settings_scroll,
            text="デバイス選択",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.device_label.pack(anchor="w", padx=20, pady=(10, 3))

        self.device_var = ctk.StringVar(value=self.config.device)
        self.device_menu = ctk.CTkOptionMenu(
            self.settings_scroll,
            variable=self.device_var,
            values=["auto", "xpu", "cuda", "cpu"],
            command=lambda _: self._save_config(),
        )
        self.device_menu.pack(anchor="w", padx=20, pady=3)

        # Data type selection
        self.dtype_label = ctk.CTkLabel(
            self.settings_scroll,
            text="データ型",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.dtype_label.pack(anchor="w", padx=20, pady=(10, 3))

        self.dtype_var = ctk.StringVar(value=self.config.dtype)
        self.dtype_menu = ctk.CTkOptionMenu(
            self.settings_scroll,
            variable=self.dtype_var,
            values=["float16", "float32", "bfloat16"],
            command=lambda _: self._save_config(),
        )
        self.dtype_menu.pack(anchor="w", padx=20, pady=3)

        # RVC models directory
        self.rvc_models_dir_label = ctk.CTkLabel(
            self.settings_scroll,
            text="RVCモデルディレクトリ",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.rvc_models_dir_label.pack(anchor="w", padx=20, pady=(10, 3))

        self.rvc_models_dir_frame = ctk.CTkFrame(self.settings_scroll, fg_color="transparent")
        self.rvc_models_dir_frame.pack(fill="x", padx=20, pady=3)

        self.rvc_models_dir_entry = ctk.CTkEntry(
            self.rvc_models_dir_frame,
            width=350,
            placeholder_text="ディレクトリを選択...",
        )
        self.rvc_models_dir_entry.pack(side="left", padx=(0, 5))
        if self.config.rvc_models_dir:
            self.rvc_models_dir_entry.insert(0, self.config.rvc_models_dir)

        self.rvc_models_dir_browse_btn = ctk.CTkButton(
            self.rvc_models_dir_frame,
            text="参照",
            width=60,
            command=self._browse_rvc_models_dir,
        )
        self.rvc_models_dir_browse_btn.pack(side="left")

        self.rvc_models_dir_desc = ctk.CTkLabel(
            self.settings_scroll,
            text="指定ディレクトリ内の .pth をスキャンしてモデル選択ドロップダウンに列挙します",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        )
        self.rvc_models_dir_desc.pack(anchor="w", padx=20, pady=(0, 5))

        # Models directory (HuBERT/RMVPE)
        self.models_dir_label = ctk.CTkLabel(
            self.settings_scroll,
            text="HuBERT・RMVPEなど推論モデルのディレクトリ",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.models_dir_label.pack(anchor="w", padx=20, pady=(10, 3))

        self.models_dir_entry = ctk.CTkEntry(
            self.settings_scroll,
            width=400,
        )
        self.models_dir_entry.pack(anchor="w", padx=20, pady=3)
        self.models_dir_entry.insert(0, self.config.models_dir)
        self.models_dir_entry.bind("<FocusOut>", lambda _: self._save_config())

        # Apply button
        self.apply_btn = ctk.CTkButton(
            self.settings_scroll,
            text="設定を適用 (モデル再読込)",
            command=self._apply_settings,
        )
        self.apply_btn.pack(anchor="w", padx=20, pady=(10, 5))

        # Settings info label
        self.settings_info = ctk.CTkLabel(
            self.settings_scroll,
            text="※ デバイス/データ型の変更はモデル再読込後に反映されます",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.settings_info.pack(anchor="w", padx=20, pady=3)

        # === Audio Test Section ===
        self._setup_audio_test_section()

    def _setup_audio_test_section(self) -> None:
        """Setup audio test section for file-based conversion."""
        # Separator
        separator = ctk.CTkFrame(self.settings_scroll, height=2, fg_color="gray50")
        separator.pack(fill="x", padx=20, pady=(15, 5))

        # Section label
        test_label = ctk.CTkLabel(
            self.settings_scroll,
            text="オーディオテスト",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        test_label.pack(anchor="w", padx=20, pady=(5, 3))

        test_desc = ctk.CTkLabel(
            self.settings_scroll,
            text="WAVファイルを選択して変換テストを実行できます",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        test_desc.pack(anchor="w", padx=20, pady=(0, 5))

        # File selection frame
        file_frame = ctk.CTkFrame(self.settings_scroll, fg_color="transparent")
        file_frame.pack(fill="x", padx=20, pady=3)

        self.test_file_entry = ctk.CTkEntry(
            file_frame,
            width=350,
            placeholder_text="WAVファイルを選択...",
        )
        self.test_file_entry.pack(side="left", padx=(0, 10))

        self.test_browse_btn = ctk.CTkButton(
            file_frame,
            text="参照",
            width=60,
            command=self._browse_test_file,
        )
        self.test_browse_btn.pack(side="left")

        # Control buttons frame
        ctrl_frame = ctk.CTkFrame(self.settings_scroll, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=20, pady=5)

        self.test_convert_btn = ctk.CTkButton(
            ctrl_frame,
            text="変換",
            width=80,
            command=self._convert_test_file,
        )
        self.test_convert_btn.pack(side="left", padx=(0, 10))

        self.test_play_btn = ctk.CTkButton(
            ctrl_frame,
            text="再生",
            width=80,
            command=self._play_converted_audio,
            state="disabled",
        )
        self.test_play_btn.pack(side="left", padx=(0, 10))

        self.test_stop_btn = ctk.CTkButton(
            ctrl_frame,
            text="停止",
            width=80,
            command=self._stop_test_playback,
            state="disabled",
        )
        self.test_stop_btn.pack(side="left", padx=(0, 10))

        self.test_save_btn = ctk.CTkButton(
            ctrl_frame,
            text="保存",
            width=80,
            command=self._save_converted_audio,
            state="disabled",
        )
        self.test_save_btn.pack(side="left")

        # Status label
        self.test_status_label = ctk.CTkLabel(
            self.settings_scroll,
            text="",
            font=ctk.CTkFont(size=11),
        )
        self.test_status_label.pack(anchor="w", padx=20, pady=3)

    def _browse_test_file(self) -> None:
        """Open file dialog to select a WAV file."""
        self.file_converter.browse_file()

    def _convert_test_file(self) -> None:
        """Convert the selected WAV file."""
        self.file_converter.convert_file()

    def _play_converted_audio(self) -> None:
        """Play the converted audio."""
        self.file_converter.play_audio()

    def _stop_test_playback(self) -> None:
        """Stop audio playback."""
        self.file_converter.stop_playback()

    def _save_converted_audio(self) -> None:
        """Save the converted audio to a file."""
        self.file_converter.save_audio()

    def _browse_rvc_models_dir(self) -> None:
        """Open folder dialog to select RVC models directory."""
        from tkinter import filedialog

        dir_path = filedialog.askdirectory(title="RVCモデルディレクトリを選択")
        if dir_path:
            self.rvc_models_dir_entry.delete(0, "end")
            self.rvc_models_dir_entry.insert(0, dir_path)
            self.config.rvc_models_dir = dir_path
            self.config.save()
            self.model_selector.scan_directory(dir_path)

    def _apply_settings(self) -> None:
        """Apply settings and reload model."""
        self.model_loader.apply_settings()

    def _on_model_selected(self, path: str) -> None:
        """Handle model selection."""
        logger.info(f"Model selected: {path}")

        # Save to config
        self.config.last_model_path = path
        self.config.save()

        # Unload current pipeline if running
        if self._is_running:
            self.realtime_controller.stop()

        # Load new model in background
        self.model_loader.load_model(path)

    def _on_pitch_changed(self, value: int) -> None:
        """Handle pitch change."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_pitch_shift(value)

    def _on_f0_mode_changed(self, use_f0: bool) -> None:
        """Handle F0 mode change."""
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_f0_mode(use_f0)

    def _on_f0_method_changed(self, method: str) -> None:
        """Handle F0 method change (rmvpe/fcpe/none)."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_f0_method(method)

    def _on_pre_hubert_pitch_changed(self, ratio: float) -> None:
        """Handle pre-HuBERT pitch shift toggle."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_pre_hubert_pitch_ratio(ratio)

    def _on_moe_boost_changed(self, strength: float) -> None:
        """Handle moe boost change."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_moe_boost(strength)

    def _on_noise_scale_changed(self, scale: float) -> None:
        """Handle synthesizer noise scale change."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_noise_scale(scale)

    def _on_fixed_harmonics_changed(self, enabled: bool) -> None:
        """Handle fixed harmonics toggle."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_fixed_harmonics(enabled)

    def _on_octave_flip_suppress_changed(self, enabled: bool) -> None:
        """Handle octave-flip suppress toggle."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_enable_octave_flip_suppress(enabled)

    def _on_f0_slew_limit_changed(self, enabled: bool) -> None:
        """Handle F0 slew limiter toggle."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_enable_f0_slew_limit(enabled)

    def _on_f0_slew_max_step_changed(self, value: float) -> None:
        """Handle F0 slew max-step slider change."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_f0_slew_max_step_st(value)

    def _on_index_changed(self) -> None:
        """Handle index checkbox change."""
        self._save_config()
        # Update voice changer if running
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_index_rate(self._get_index_rate())
        # Update status bar
        index_loaded = self.pipeline is not None and self.pipeline.faiss_index is not None
        self.status_bar.set_index_status(index_loaded, self._get_index_rate())

    def _on_denoise_changed(self) -> None:
        """Handle denoise settings change."""
        self._save_config()
        # Update voice changer if running
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_denoise(
                self.use_denoise_var.get(),
                self.denoise_method_var.get(),
            )

    def _on_index_ratio_changed(self, value: float) -> None:
        """Handle index ratio slider change."""
        self.index_ratio_value.configure(text=f"{value:.2f}")
        self._save_config()
        # Update voice changer if running
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_index_rate(self._get_index_rate())
        # Update status bar
        index_loaded = self.pipeline is not None and self.pipeline.faiss_index is not None
        self.status_bar.set_index_status(index_loaded, self._get_index_rate())

    def _on_voice_gate_mode_changed(self) -> None:
        """Handle voice gate mode change."""
        mode = self.voice_gate_mode_var.get()
        # Show/hide energy threshold slider
        if mode == "energy":
            self.energy_threshold_frame.pack(fill="x", padx=10, pady=(3, 0))
            # Re-pack description after slider
            self.voice_gate_desc.pack_forget()
            self.voice_gate_desc.pack(anchor="w", padx=10, pady=(0, 5))
        else:
            self.energy_threshold_frame.pack_forget()
        self._save_config()
        # Update voice changer if running
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_voice_gate_mode(mode)

    def _on_energy_threshold_changed(self, value: float) -> None:
        """Handle energy threshold slider change."""
        self.energy_threshold_value.configure(text=f"{value:.2f}")
        self._save_config()
        # Update voice changer if running
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_energy_threshold(value)

    def _get_index_rate(self) -> float:
        """Get current index rate (0 if disabled)."""
        if self.use_index_var.get():
            return self.index_ratio_slider.get()
        return 0.0

    def _restore_latency_settings(self) -> None:
        """Restore latency settings from config.

        Only chunk_sec is restored; other parameters are auto-derived.
        """
        if not hasattr(self, "latency_settings"):
            return

        self.latency_settings.set_values(chunk_sec=self.config.audio.chunk_sec)

    def _on_audio_settings_changed(self) -> None:
        """Handle audio settings change."""
        # Update device display in main panel
        self._update_audio_device_display()
        # Save immediately
        self._save_config()

    def _on_latency_settings_changed(self) -> None:
        """Handle latency settings change."""
        # Save immediately
        self._save_config()
        # Apply changes in real-time if voice changer is running
        if hasattr(self, "latency_settings") and self.realtime_controller.voice_changer:
            settings = self.latency_settings.get_settings()
            logger.debug(f"Latency settings changed: {settings}")

            # Update all config values BEFORE triggering restart.
            # set_chunk_sec() restarts the pipeline, so overlap/crossfade/
            # prebuffer/margin must be set first to take effect.
            vc = self.realtime_controller.voice_changer
            vc.set_prebuffer_chunks(settings["prebuffer_chunks"])
            vc.set_buffer_margin(settings["buffer_margin"])
            vc.set_overlap(settings["overlap_sec"])
            vc.set_crossfade(settings["crossfade_sec"])
            vc.set_chunk_sec(settings["chunk_sec"])

    def _on_postprocess_settings_changed(self) -> None:
        """Handle post-processing settings change."""
        self._save_config()
        if hasattr(self, "postprocess_settings") and self.realtime_controller.voice_changer:
            cfg = self.postprocess_settings.get_config()
            vc = self.realtime_controller.voice_changer
            if hasattr(vc, "set_postprocess_config"):
                vc.set_postprocess_config(cfg)

    def _save_config(self) -> None:
        """Save all config settings immediately."""
        # Skip saving during initialization to avoid accessing uninitialized attributes
        if self._initializing:
            return

        try:
            self.config.device = self.device_var.get()
            self.config.dtype = self.dtype_var.get()
            self.config.models_dir = self.models_dir_entry.get()
            rvc_dir = self.rvc_models_dir_entry.get().strip()
            self.config.rvc_models_dir = rvc_dir if rvc_dir else None
            self.config.inference.use_compile = self.compile_var.get()
            self.config.inference.use_index = self.use_index_var.get()
            self.config.inference.index_ratio = self.index_ratio_slider.get()
            self.config.inference.pitch_shift = self.pitch_control.pitch
            self.config.inference.f0_method = self.pitch_control.f0_method
            self.config.inference.pre_hubert_pitch_ratio = self.pitch_control.pre_hubert_pitch_ratio
            self.config.inference.moe_boost = self.pitch_control.moe_boost
            self.config.inference.noise_scale = self.pitch_control.noise_scale
            self.config.inference.fixed_harmonics = self.pitch_control.fixed_harmonics
            self.config.inference.enable_octave_flip_suppress = (
                self.pitch_control.enable_octave_flip_suppress
            )
            self.config.inference.enable_f0_slew_limit = self.pitch_control.enable_f0_slew_limit
            self.config.inference.f0_slew_max_step_st = self.pitch_control.f0_slew_max_step_st
            self.config.inference.denoise.enabled = self.use_denoise_var.get()
            self.config.inference.denoise.method = self.denoise_method_var.get()
            self.config.inference.voice_gate_mode = self.voice_gate_mode_var.get()
            self.config.inference.energy_threshold = self.energy_threshold_slider.get()
            # Save latency settings (all from LatencySettingsFrame)
            if hasattr(self, "latency_settings"):
                latency = self.latency_settings.get_settings()
                self.config.audio.chunk_sec = latency["chunk_sec"]
                self.config.audio.prebuffer_chunks = latency["prebuffer_chunks"]
                self.config.audio.buffer_margin = latency["buffer_margin"]
                self.config.inference.overlap_sec = latency["overlap_sec"]
                self.config.inference.crossfade_sec = latency["crossfade_sec"]
                self.config.inference.use_sola = latency["use_sola"]
            self.config.audio.input_gain_db = self.audio_settings.input_gain_db
            self.config.audio.input_channel_selection = self.audio_settings.get_channel_selection()
            self.config.audio.input_hostapi_filter = self.audio_settings.input_api_var.get()
            self.config.audio.output_hostapi_filter = self.audio_settings.output_api_var.get()
            self.config.audio.input_device_name = self.audio_settings.get_input_device_name()
            self.config.audio.output_device_name = self.audio_settings.get_output_device_name()
            self.config.save()
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _on_tab_changed(self) -> None:
        """Handle tab change event."""
        if not hasattr(self, "audio_settings"):
            return

        tab_name = self.tabview.get()
        if tab_name == "オーディオ":
            # Start auto-refresh when audio tab is selected
            self.audio_settings.start_auto_refresh(interval_ms=1000)
        else:
            # Stop auto-refresh when leaving audio tab
            self.audio_settings.stop_auto_refresh()

    def _update_audio_device_display(self) -> None:
        """Update audio device labels in main panel."""
        if hasattr(self, "audio_settings"):
            input_name = self.audio_settings.get_input_device_name()
            output_name = self.audio_settings.get_output_device_name()
            # Truncate long names
            if len(input_name) > 35:
                input_name = input_name[:32] + "..."
            if len(output_name) > 35:
                output_name = output_name[:32] + "..."
            self.mic_label.configure(text=f"🎤 {input_name}")
            self.speaker_label.configure(text=f"🔊 {output_name}")

    def _toggle_running(self) -> None:
        """Toggle voice changer on/off."""
        self.realtime_controller.toggle()

    def _run_audio_test(self) -> None:
        """Run audio test: record -> convert -> playback."""
        self.audio_test_manager.run_test()

    def _on_wav_input_toggled(self) -> None:
        """Show/hide WAV file selector based on checkbox state."""
        if self.use_wav_input_var.get():
            self.wav_input_file_frame.pack(fill="x", padx=10, pady=(0, 5))
        else:
            self.wav_input_file_frame.pack_forget()

    def _browse_wav_input(self) -> None:
        """Open file dialog to select a WAV file for loop input."""
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="WAVファイルを選択",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.wav_input_path_var.set(path)

    def _on_close(self) -> None:
        """Handle window close."""
        # Stop voice changer
        if self._is_running:
            self.realtime_controller.stop()

        # Stop test playback
        self.file_converter.stop_playback()

        # Stop audio monitor
        self.audio_settings.stop_monitor()

        # Stop device auto-refresh
        self.audio_settings.stop_auto_refresh()

        # Save config
        self._save_config()

        # Destroy window
        self.destroy()

    def run(self) -> None:
        """Run the application."""
        self.mainloop()


def run_gui() -> None:
    """Entry point for GUI application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = RCWXApp()
    app.run()


if __name__ == "__main__":
    run_gui()
