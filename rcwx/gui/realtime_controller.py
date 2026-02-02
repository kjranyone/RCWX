"""Real-time voice changer controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import customtkinter as ctk
import sounddevice as sd

from rcwx.pipeline.realtime import RealtimeConfig, RealtimeStats, RealtimeVoiceChanger

if TYPE_CHECKING:
    from rcwx.gui.app import RCWXApp

logger = logging.getLogger(__name__)


class RealtimeController:
    """
    Controls real-time voice changer lifecycle.

    Handles:
    - Starting/stopping voice changer
    - Stats updates
    - Error handling
    - Feedback detection warnings
    - Buffer warnings
    """

    def __init__(self, app: RCWXApp):
        """
        Initialize realtime controller.

        Args:
            app: Reference to main application
        """
        self.app = app
        self.voice_changer: Optional[RealtimeVoiceChanger] = None
        self._buffer_warning_shown = {'underrun': False, 'overrun': False}

    def toggle(self) -> None:
        """Toggle voice changer on/off."""
        if self.app._is_running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        """Start voice changer asynchronously."""
        if self.app.pipeline is None:
            logger.warning("No model loaded")
            return

        if self.app._loading:
            return

        # Check for same audio interface (potential feedback)
        if self._check_same_audio_interface():
            logger.warning("Input and output use same audio interface - feedback may occur")
            self._show_warning(
                "フィードバック警告",
                "入力と出力が同じオーディオインターフェースを使用しています。\n"
                "フィードバックループが発生する可能性があります。\n\n"
                "推奨: USBマイクなど、別のインターフェースを入力に使用してください。",
            )

        # Stop audio monitor to avoid device conflict
        self.app.audio_settings.stop_monitor()

        # Disable button and show loading state
        self.app.start_btn.configure(state="disabled", text="⏳ 起動中...")
        self.app._loading = True
        self.app.status_bar.set_loading()

        # NOTE: sounddevice/PortAudio requires audio streams to be created
        # from the main thread on Windows. Running in a separate thread
        # causes "Invalid sample rate" or other errors.
        try:
            # Get latency settings
            latency = self.app.latency_settings.get_settings()

            # Create realtime config with auto-detected sample rates and channels
            rt_config = RealtimeConfig(
                input_device=self.app.audio_settings.input_device,
                output_device=self.app.audio_settings.output_device,
                mic_sample_rate=self.app.audio_settings.input_sample_rate,
                output_sample_rate=self.app.audio_settings.output_sample_rate,
                input_channels=self.app.audio_settings.input_channels,
                output_channels=self.app.audio_settings.output_channels,
                input_channel_selection=self.app.audio_settings.get_channel_selection(),
                # Latency settings (from LatencySettingsFrame)
                chunk_sec=latency["chunk_sec"],
                prebuffer_chunks=latency["prebuffer_chunks"],
                buffer_margin=latency["buffer_margin"],
                context_sec=latency["context_sec"],
                crossfade_sec=latency["crossfade_sec"],
                chunking_mode=latency["chunking_mode"],
                # Pitch settings
                pitch_shift=self.app.pitch_control.pitch,
                use_f0=self.app.pitch_control.use_f0,
                f0_method=self.app.pitch_control.f0_method,
                # Audio settings
                input_gain_db=self.app.audio_settings.input_gain_db,
                index_rate=self.app._get_index_rate(),
                denoise_enabled=self.app.use_denoise_var.get(),
                denoise_method=self.app.denoise_method_var.get(),
                voice_gate_mode=self.app.voice_gate_mode_var.get(),
                energy_threshold=self.app.energy_threshold_slider.get(),
                use_feature_cache=self.app.use_feature_cache_var.get(),
                # w-okada style processing (from LatencySettingsFrame)
                extra_sec=self.app.config.inference.extra_sec,
                lookahead_sec=latency["lookahead_sec"],
                use_sola=latency["use_sola"],
            )

            # Create voice changer with warmup progress callback
            self.voice_changer = RealtimeVoiceChanger(
                self.app.pipeline,
                config=rt_config,
                on_warmup_progress=self._on_warmup_progress,
            )
            self.voice_changer.on_stats_update = self._on_stats_update
            self.voice_changer.on_error = self._on_inference_error

            # Update UI before starting (may take a moment for warmup)
            self.app.update_idletasks()

            # Start (this calls pipeline.load() internally)
            # Must be called from main thread due to sounddevice limitations
            self.voice_changer.start()

            # Success
            self._on_started()
        except Exception as e:
            logger.error(f"Failed to start voice changer: {e}")
            self._on_start_error(str(e))

    def _on_warmup_progress(self, current: int, total: int, message: str) -> None:
        """Called during warmup to show progress."""
        self.app.start_btn.configure(text=f"⏳ {message}")
        self.app.update_idletasks()  # Force UI update

    def _on_started(self) -> None:
        """Called when voice changer starts successfully."""
        self.app._loading = False
        self.app._is_running = True
        self.app.start_btn.configure(text="■ 停止", fg_color="#cc3333", state="normal")
        self.app.status_bar.set_running(True)

    def _on_start_error(self, error_msg: str) -> None:
        """Called when voice changer fails to start."""
        self.app._loading = False
        self.app.start_btn.configure(text="▶ 開始", fg_color=["#3B8ED0", "#1F6AA5"], state="normal")
        self.app.status_bar.set_running(False)

        # Provide helpful message for common errors
        if "WdmSyncIoctl" in error_msg or "WDM-KS" in error_msg:
            self._show_warning(
                "オーディオデバイスエラー",
                f"オーディオデバイスでエラーが発生しました。\n\n"
                f"詳細: {error_msg[:100]}...\n\n"
                "解決策:\n"
                "1. 別のオーディオデバイスを試してください\n"
                "2. チャンクサイズを大きくしてください（オーディオタブ）\n"
                "3. Windowsの「サウンド設定」でデバイスを確認してください",
            )
        elif (
            "Output size is too small" in error_msg
            or "size" in error_msg.lower()
            and "0" in error_msg
        ):
            self._show_warning(
                "チャンクサイズエラー",
                "チャンクサイズが小さすぎます。\n\n"
                "オーディオタブでチャンクサイズを増やしてください。\n"
                "推奨: 350ms以上",
            )
        else:
            self._show_error(f"起動エラー: {error_msg}")

    def stop(self) -> None:
        """Stop the voice changer."""
        if self.voice_changer:
            self.voice_changer.stop()
            self.voice_changer = None

        self.app._is_running = False
        self.app.start_btn.configure(text="▶ 開始", fg_color=["#3B8ED0", "#1F6AA5"])
        self.app.status_bar.set_running(False)

        # Reset buffer warning flags so warnings show again on next start
        self._buffer_warning_shown = {'underrun': False, 'overrun': False}

    def _on_stats_update(self, stats: RealtimeStats) -> None:
        """Handle stats update from voice changer."""
        # Update UI from main thread
        self.app.after(0, lambda: self.app.status_bar.update_stats(stats))

        # Show buffer warnings (first occurrence only, after threshold)
        # Threshold: 5 occurrences to avoid false positives during startup
        BUFFER_WARNING_THRESHOLD = 5

        # Buffer underrun warning
        if stats.buffer_underruns >= BUFFER_WARNING_THRESHOLD and not self._buffer_warning_shown['underrun']:
            self._buffer_warning_shown['underrun'] = True
            self.app.after(0, self._show_buffer_underrun_warning)

        # Buffer overrun warning
        if stats.buffer_overruns >= BUFFER_WARNING_THRESHOLD and not self._buffer_warning_shown['overrun']:
            self._buffer_warning_shown['overrun'] = True
            self.app.after(0, self._show_buffer_overrun_warning)

    def _on_inference_error(self, error_msg: str) -> None:
        """Handle inference error from voice changer."""
        # Update UI from main thread
        self.app.after(0, lambda: self._show_error(error_msg))

    def _show_buffer_underrun_warning(self) -> None:
        """Show buffer underrun warning."""
        self._show_warning(
            "バッファアンダーラン検出",
            "音声処理が入力に追いついていません（音切れが発生）。\n\n"
            "対策:\n"
            "1. チャンクサイズを増やす (オーディオタブ)\n"
            "   • RMVPE: 350ms → 400ms\n"
            "   • FCPE: 150ms → 200ms\n\n"
            "2. F0方式を変更 (メインタブ)\n"
            "   • RMVPE → FCPE (より高速)\n"
            "   • または F0なしモード\n\n"
            "3. ノイズキャンセリングを無効化\n\n"
            "4. バッファマージンを増やす (オーディオタブ)\n"
            "   • 0.5 → 1.0"
        )

    def _show_buffer_overrun_warning(self) -> None:
        """Show buffer overrun warning."""
        self._show_warning(
            "バッファオーバーラン検出",
            "入力キューが満杯です（チャンクがドロップされています）。\n\n"
            "対策:\n"
            "1. チャンクサイズを増やす (オーディオタブ)\n"
            "   • 現在値から +50ms 程度増やす\n\n"
            "2. F0方式を変更 (メインタブ)\n"
            "   • RMVPE → FCPE (より高速)\n"
            "   • または F0なしモード\n\n"
            "3. 出力デバイスを確認\n"
            "   • サンプルレートが正しいか確認\n"
            "   • 別のデバイスを試す\n\n"
            "4. デノイズを無効化してテスト"
        )

    def _check_same_audio_interface(self) -> bool:
        """Check if input and output use the same audio interface (potential feedback)."""
        try:
            devices = sd.query_devices()

            input_idx = self.app.audio_settings.input_device
            output_idx = self.app.audio_settings.output_device

            # Use defaults if None
            if input_idx is None:
                input_idx = sd.default.device[0]
            if output_idx is None:
                output_idx = sd.default.device[1]

            if input_idx is None or output_idx is None:
                return False

            input_name = devices[input_idx]["name"].lower()
            output_name = devices[output_idx]["name"].lower()

            # Check for common interface indicators
            # "High Definition Audio" is the typical onboard audio
            hda_keywords = ["high definition audio", "realtek", "hd audio"]
            input_is_hda = any(kw in input_name for kw in hda_keywords)
            output_is_hda = any(kw in output_name for kw in hda_keywords)

            return input_is_hda and output_is_hda
        except Exception:
            return False

    def _show_warning(self, title: str, message: str) -> None:
        """Show warning dialog."""
        dialog = ctk.CTkToplevel(self.app)
        dialog.title(title)
        dialog.geometry("450x300")
        dialog.transient(self.app)
        dialog.grab_set()

        # Center on parent
        dialog.update_idletasks()
        x = self.app.winfo_x() + (self.app.winfo_width() - 450) // 2
        y = self.app.winfo_y() + (self.app.winfo_height() - 300) // 2
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
        self.app.model_selector.status_label.configure(text=short_msg, text_color="#ff6666")
