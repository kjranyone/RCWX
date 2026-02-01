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
from rcwx.pipeline.inference import RVCPipeline
from rcwx.pipeline.realtime import RealtimeStats

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
        )
        self.model_selector.pack(fill="x", pady=(0, 5))

        # Pitch control
        self.pitch_control = PitchControl(
            self.left_column,
            on_pitch_changed=self._on_pitch_changed,
            on_f0_mode_changed=self._on_f0_mode_changed,
            on_f0_method_changed=self._on_f0_method_changed,
        )
        self.pitch_control.pack(fill="x", pady=(0, 5))

        # Restore saved F0 method
        self.pitch_control.set_f0_method(self.config.inference.f0_method)

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

        # Chunk processing options
        self.chunk_frame = ctk.CTkFrame(self.left_column)
        self.chunk_frame.pack(fill="x", pady=(0, 5))

        self.chunk_label = ctk.CTkLabel(
            self.chunk_frame,
            text="■ チャンク処理",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.chunk_label.pack(anchor="w", padx=10, pady=(5, 3))

        self.use_feature_cache_var = ctk.BooleanVar(value=self.config.inference.use_feature_cache)
        self.use_feature_cache_cb = ctk.CTkCheckBox(
            self.chunk_frame,
            text="特徴量キャッシュ (HuBERT/F0継続性)",
            variable=self.use_feature_cache_var,
            command=self._on_feature_cache_changed,
        )
        self.use_feature_cache_cb.pack(anchor="w", padx=10, pady=(2, 5))

        # Note: Context, Lookahead, SOLA settings are in the Latency Settings panel (Audio tab)

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
        self.test_status.pack(anchor="w", padx=10, pady=(0, 5))

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

        # Restore saved device selections
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

        # Models directory
        self.models_dir_label = ctk.CTkLabel(
            self.settings_scroll,
            text="モデルディレクトリ",
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

    def _on_feature_cache_changed(self) -> None:
        """Handle feature cache toggle change."""
        self._save_config()
        if self.realtime_controller.voice_changer:
            self.realtime_controller.voice_changer.set_feature_cache(self.use_feature_cache_var.get())

    def _get_index_rate(self) -> float:
        """Get current index rate (0 if disabled)."""
        if self.use_index_var.get():
            return self.index_ratio_slider.get()
        return 0.0

    def _restore_latency_settings(self) -> None:
        """Restore latency settings from config."""
        if not hasattr(self, "latency_settings"):
            return

        # Restore saved values
        self.latency_settings.set_values(
            chunk_sec=self.config.audio.chunk_sec,
            prebuffer_chunks=self.config.audio.prebuffer_chunks,
            buffer_margin=self.config.audio.buffer_margin,
            context_sec=self.config.inference.context_sec,
            lookahead_sec=self.config.inference.lookahead_sec,
            crossfade_sec=self.config.inference.crossfade_sec,
            use_sola=self.config.inference.use_sola,
        )

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
            # Apply real-time settings
            self.realtime_controller.voice_changer.set_chunk_sec(settings["chunk_sec"])
            self.realtime_controller.voice_changer.set_prebuffer_chunks(settings["prebuffer_chunks"])
            self.realtime_controller.voice_changer.set_buffer_margin(settings["buffer_margin"])
            self.realtime_controller.voice_changer.set_context(settings["context_sec"])
            self.realtime_controller.voice_changer.set_lookahead(settings["lookahead_sec"])
            self.realtime_controller.voice_changer.set_crossfade(settings["crossfade_sec"])
            self.realtime_controller.voice_changer.set_sola(settings["use_sola"])

    def _save_config(self) -> None:
        """Save all config settings immediately."""
        # Skip saving during initialization to avoid accessing uninitialized attributes
        if self._initializing:
            return

        try:
            self.config.device = self.device_var.get()
            self.config.dtype = self.dtype_var.get()
            self.config.models_dir = self.models_dir_entry.get()
            self.config.inference.use_compile = self.compile_var.get()
            self.config.inference.use_index = self.use_index_var.get()
            self.config.inference.index_ratio = self.index_ratio_slider.get()
            self.config.inference.f0_method = self.pitch_control.f0_method
            self.config.inference.denoise.enabled = self.use_denoise_var.get()
            self.config.inference.denoise.method = self.denoise_method_var.get()
            self.config.inference.voice_gate_mode = self.voice_gate_mode_var.get()
            self.config.inference.energy_threshold = self.energy_threshold_slider.get()
            self.config.inference.use_feature_cache = self.use_feature_cache_var.get()
            # Save latency settings (all from LatencySettingsFrame)
            if hasattr(self, "latency_settings"):
                latency = self.latency_settings.get_settings()
                self.config.audio.chunk_sec = latency["chunk_sec"]
                self.config.audio.prebuffer_chunks = latency["prebuffer_chunks"]
                self.config.audio.buffer_margin = latency["buffer_margin"]
                self.config.inference.context_sec = latency["context_sec"]
                self.config.inference.lookahead_sec = latency["lookahead_sec"]
                self.config.inference.crossfade_sec = latency["crossfade_sec"]
                self.config.inference.use_sola = latency["use_sola"]
            self.config.audio.input_gain_db = self.audio_settings.input_gain_db
            self.config.audio.input_channel_selection = self.audio_settings.get_channel_selection()
            self.config.audio.input_device_name = self.audio_settings.get_input_device_name()
            self.config.audio.output_device_name = self.audio_settings.get_output_device_name()
            self.config.save()
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

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

    def _on_stats_update(self, stats: RealtimeStats) -> None:
        """Handle stats update from voice changer."""
        # Update UI from main thread
        self.after(0, lambda: self.status_bar.update_stats(stats))

        # Show buffer warnings (first occurrence only, after threshold)
        # Threshold: 5 occurrences to avoid false positives during startup
        BUFFER_WARNING_THRESHOLD = 5

        # Buffer underrun warning
        if stats.buffer_underruns >= BUFFER_WARNING_THRESHOLD and not self._buffer_warning_shown['underrun']:
            self._buffer_warning_shown['underrun'] = True
            self.after(0, self._show_buffer_underrun_warning)

        # Buffer overrun warning
        if stats.buffer_overruns >= BUFFER_WARNING_THRESHOLD and not self._buffer_warning_shown['overrun']:
            self._buffer_warning_shown['overrun'] = True
            self.after(0, self._show_buffer_overrun_warning)

    def _on_inference_error(self, error_msg: str) -> None:
        """Handle inference error from voice changer."""
        # Update UI from main thread
        self.after(0, lambda: self._show_error(error_msg))

    def _show_warning(self, title: str, message: str) -> None:
        """Show warning dialog."""
        import customtkinter as ctk

        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("450x300")
        dialog.transient(self)
        dialog.grab_set()

        # Center on parent
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 450) // 2
        y = self.winfo_y() + (self.winfo_height() - 300) // 2
        dialog.geometry(f"+{x}+{y}")

        label = ctk.CTkLabel(
            dialog,
            text=message,
            justify="left",
            wraplength=410,
        )
        label.pack(pady=20, padx=20)

        btn = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
        btn.pack(pady=10)

    def _show_error(self, error_msg: str) -> None:
        """Show error message in UI."""
        # Show in model selector's status label (truncate long messages)
        short_msg = error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
        self.model_selector.status_label.configure(text=short_msg, text_color="#ff6666")

    def _run_audio_test(self) -> None:
        """Run audio test: record -> convert -> playback."""
        self.audio_test_manager.run_test()

    def _on_close(self) -> None:
        """Handle window close."""
        # Stop voice changer
        if self._is_running:
            self.realtime_controller.stop()

        # Stop test playback
        self.file_converter.stop_playback()

        # Stop audio monitor
        self.audio_settings.stop_monitor()

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
