"""Real-time voice changer controller."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Callable, Optional

import customtkinter as ctk
import sounddevice as sd

from rcwx.pipeline.realtime_unified import (
    RealtimeConfig,
    RealtimeStats,
    RealtimeVoiceChangerUnified,
)

if TYPE_CHECKING:
    from rcwx.gui.app import RCWXApp

logger = logging.getLogger(__name__)

# Buffer warning dialog policy: startup always produces a few underruns while
# the stream stabilizes, so session-cumulative thresholds fire every single
# session.  Instead, ignore a grace period after (re)start and require the
# threshold to be crossed within a sliding window — i.e. only warn about
# SUSTAINED problems that the user should actually act on.
BUFFER_WARNING_GRACE_SEC = 10.0
BUFFER_WARNING_WINDOW_SEC = 30.0
BUFFER_WARNING_THRESHOLD = 5


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
        self.voice_changer: Optional[RealtimeVoiceChangerUnified] = None
        self._buffer_warning_shown = {"underrun": False, "overrun": False}
        # Sliding-window state for buffer warning dialogs:
        # snapshots of (time, underruns, overruns)
        self._warn_history: deque[tuple[float, int, int]] = deque()
        self._session_start: float = 0.0
        # Latency settings that arrived while an async (re)start was in
        # flight; applied once the pipeline is running again.
        self._pending_latency_settings: Optional[dict] = None

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

        # Check for same audio interface (potential feedback) — skip for WAV input
        if not self.app.use_wav_input_var.get() and self._check_same_audio_interface():
            logger.warning("Input and output use same audio interface - feedback may occur")
            self._show_warning(
                "フィードバック警告",
                "入力と出力が同じオーディオインターフェースを使用しています。\n"
                "フィードバックループが発生する可能性があります。\n\n"
                "推奨: USBマイクなど、別のインターフェースを入力に使用してください。",
            )

        # Disable button and show loading state
        self._begin_loading("起動中...")

        try:
            # Get latency settings
            latency = self.app.latency_settings.get_settings()

            # Create unified realtime config
            rt_config = RealtimeConfig(
                input_device=self.app.audio_settings.input_device,
                output_device=self.app.audio_settings.output_device,
                mic_sample_rate=self.app.audio_settings.input_sample_rate,
                output_sample_rate=self.app.audio_settings.output_sample_rate,
                input_channels=self.app.audio_settings.input_channels,
                output_channels=self.app.audio_settings.output_channels,
                input_channel_selection=self.app.audio_settings.get_channel_selection(),
                output_channel_selection=self.app.audio_settings.get_output_channel_selection(),
                asio_buffer_size=self.app.audio_settings.asio_buffer_size,
                # Latency settings
                chunk_sec=latency["chunk_sec"],
                latency_mode=latency["latency_mode"],
                prebuffer_chunks=latency["prebuffer_chunks"],
                buffer_margin=latency["buffer_margin"],
                overlap_sec=latency["overlap_sec"],
                crossfade_sec=latency["crossfade_sec"],
                sola_search_ms=latency["sola_search_ms"],
                use_sola=latency["use_sola"],
                # Pitch settings
                pitch_shift=self.app.pitch_control.pitch,
                use_f0=self.app.pitch_control.use_f0,
                f0_method=self.app.pitch_control.f0_method,
                moe_boost=self.app.pitch_control.moe_boost,
                hubert_context_sec=self.app.config.inference.hubert_context_sec,
                f0_context_sec=self.app.config.inference.f0_context_sec,
                # Audio settings
                input_gain_db=self.app.audio_settings.input_gain_db,
                output_gain_db=self.app.config.audio.output_gain_db,
                index_rate=self.app._get_index_rate(),
                # Deadline modes bypass denoisers whose p99 exceeds the hop.
                denoise_enabled=(
                    self.app.use_denoise_var.get()
                    and latency["latency_mode"] not in {"sub100", "frontier"}
                ),
                denoise_method=self.app.denoise_method_var.get(),
                denoise_strength=self.app.denoise_strength_slider.get(),
                noise_scale=self.app.pitch_control.noise_scale,
                fixed_harmonics=self.app.pitch_control.fixed_harmonics,
                f0_lowpass_cutoff_hz=self.app.config.inference.f0_lowpass_cutoff_hz,
                enable_octave_flip_suppress=self.app.pitch_control.enable_octave_flip_suppress,
                enable_f0_slew_limit=self.app.pitch_control.enable_f0_slew_limit,
                f0_slew_max_step_st=self.app.pitch_control.f0_slew_max_step_st,
                f0_hole_fill_ms=self.app.config.inference.f0_hole_fill_ms,
                uv_ramp_ms=self.app.config.inference.uv_ramp_ms,
                voice_gate_mode=self.app.voice_gate_mode_var.get(),
                energy_threshold=self.app.energy_threshold_slider.get(),
                # Post-processing
                postprocess_enabled=self.app.config.inference.postprocess.enabled,
                treble_boost_db=self.app.config.inference.postprocess.treble_boost_db,
                treble_cutoff_hz=self.app.config.inference.postprocess.treble_cutoff_hz,
                limiter_threshold_db=self.app.config.inference.postprocess.limiter_threshold_db,
                limiter_release_ms=self.app.config.inference.postprocess.limiter_release_ms,
                # WAV file input
                wav_input_path=(
                    self.app.wav_input_path_var.get() if self.app.use_wav_input_var.get() else ""
                ),
            )

            # Heavy construction (model load check + warmup) runs on a
            # worker thread so the GUI stays responsive; all Tk state was
            # already read into rt_config above on the main thread.
            def _factory() -> RealtimeVoiceChangerUnified:
                vc = RealtimeVoiceChangerUnified(
                    self.app.pipeline,
                    config=rt_config,
                    on_warmup_progress=self._on_warmup_progress,
                )
                vc.on_stats_update = self._on_stats_update
                vc.on_error = self._on_inference_error
                return vc

            self._prepare_and_start_async(_factory)
        except Exception as e:
            logger.error(f"Failed to start voice changer: {e}")
            self._on_start_error(str(e))

    def _begin_loading(self, message: str) -> None:
        """Put the UI into loading state (main thread)."""
        self.app.start_btn.configure(state="disabled", text=f"⏳ {message}")
        self.app._loading = True
        self.app.status_bar.set_loading()

    def _ui(self, fn: Callable[[], None]) -> None:
        """Schedule a callable on the Tk main thread (safe from workers)."""
        try:
            self.app.after(0, fn)
        except Exception:
            logger.debug("UI update skipped (app closing)")

    def _prepare_and_start_async(
        self, vc_factory: Callable[[], RealtimeVoiceChangerUnified]
    ) -> None:
        """Warm up on a background thread, then open streams on the main one.

        Only audio stream creation must run on the main thread on Windows
        (PortAudio fails with "Invalid sample rate" etc. otherwise); the
        expensive part — model load and XPU kernel warmup — has no such
        constraint and previously froze the GUI for ~10s.
        """

        def _worker() -> None:
            try:
                vc = vc_factory()
                vc.prepare()
            except Exception as e:
                logger.error(f"Voice changer warmup failed: {e}", exc_info=True)
                self._ui(lambda err=str(e): self._on_start_error(err))
                return

            def _open_streams(vc: RealtimeVoiceChangerUnified = vc) -> None:
                try:
                    # Fast: warmups already done, only opens audio streams.
                    vc.start()
                    self.voice_changer = vc
                    self._on_started()
                except Exception as e:
                    logger.error(f"Failed to start voice changer: {e}")
                    self._on_start_error(str(e))

            self._ui(_open_streams)

        threading.Thread(target=_worker, daemon=True, name="RCWX-Warmup").start()

    def _on_warmup_progress(self, current: int, total: int, message: str) -> None:
        """Show warmup progress (called from the warmup worker thread)."""
        self._ui(lambda m=message: self.app.start_btn.configure(text=f"⏳ {m}"))

    def _on_started(self) -> None:
        """Called when voice changer starts successfully."""
        self.app._loading = False
        self.app._is_running = True
        self.app.start_btn.configure(text="■ 停止", fg_color="#cc3333", state="normal")
        self.app.status_bar.set_running(True)
        self._session_start = time.monotonic()
        self._warn_history.clear()

        # Latency settings changed while (re)starting — apply them now.
        pending = self._pending_latency_settings
        self._pending_latency_settings = None
        if pending is not None:
            self.apply_latency_settings(pending)

    def _on_start_error(self, error_msg: str) -> None:
        """Called when voice changer fails to start (or restart)."""
        self.app._loading = False
        self.app._is_running = False
        self._pending_latency_settings = None
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
        self.app.reset_output_meter()
        self.app.audio_settings.reset_input_meter()

        # Reset buffer warning flags so warnings show again on next start
        self._buffer_warning_shown = {"underrun": False, "overrun": False}
        self._warn_history.clear()
        self._session_start = 0.0

    # ======== Runtime parameter passthrough ========
    # Forward live parameter changes to the running voice changer, guarding on
    # whether one exists. app.py calls these instead of reaching through
    # `.voice_changer.set_*()` directly (Law of Demeter): the controller owns
    # the voice changer lifecycle, so it owns the runtime-update surface too.

    @property
    def is_running(self) -> bool:
        """True when a voice changer is active."""
        return self.voice_changer is not None

    def set_pitch_shift(self, semitones: int) -> None:
        if self.voice_changer:
            self.voice_changer.set_pitch_shift(semitones)

    def set_f0_mode(self, enabled: bool) -> None:
        if self.voice_changer:
            self.voice_changer.set_f0_mode(enabled)

    def set_f0_method(self, method: str) -> None:
        if self.voice_changer:
            self.voice_changer.set_f0_method(method)

    def set_moe_boost(self, strength: float) -> None:
        if self.voice_changer:
            self.voice_changer.set_moe_boost(strength)

    def set_noise_scale(self, scale: float) -> None:
        if self.voice_changer:
            self.voice_changer.set_noise_scale(scale)

    def set_fixed_harmonics(self, enabled: bool) -> None:
        if self.voice_changer:
            self.voice_changer.set_fixed_harmonics(enabled)

    def set_enable_octave_flip_suppress(self, enabled: bool) -> None:
        if self.voice_changer:
            self.voice_changer.set_enable_octave_flip_suppress(enabled)

    def set_enable_f0_slew_limit(self, enabled: bool) -> None:
        if self.voice_changer:
            self.voice_changer.set_enable_f0_slew_limit(enabled)

    def set_f0_slew_max_step_st(self, step_st: float) -> None:
        if self.voice_changer:
            self.voice_changer.set_f0_slew_max_step_st(step_st)

    def set_index_rate(self, rate: float) -> None:
        if self.voice_changer:
            self.voice_changer.set_index_rate(rate)

    def set_denoise(self, enabled: bool, method: str, strength: float) -> None:
        if self.voice_changer:
            deadline_mode = self.voice_changer.config.latency_mode in {"sub100", "frontier"}
            self.voice_changer.set_denoise(enabled and not deadline_mode, method, strength)

    def set_voice_gate_mode(self, mode: str) -> None:
        if self.voice_changer:
            self.voice_changer.set_voice_gate_mode(mode)

    def set_energy_threshold(self, value: float) -> None:
        if self.voice_changer:
            self.voice_changer.set_energy_threshold(value)

    def set_input_gain_db(self, gain_db: float) -> None:
        if self.voice_changer:
            self.voice_changer.set_input_gain_db(gain_db)

    def set_output_gain_db(self, gain_db: float) -> None:
        if self.voice_changer:
            self.voice_changer.set_output_gain_db(gain_db)

    def apply_latency_settings(self, settings: dict) -> None:
        """Apply latency settings; chunk changes restart asynchronously.

        Non-restarting parameters (overlap/crossfade/prebuffer/margin) are
        applied directly. Chunk or latency-mode changes require a pipeline
        restart because the callback size and Graph shapes may change. Fresh
        warmup runs through the same async path as start(), and settings
        arriving mid-restart are stashed until the pipeline is running.
        """
        if not self.voice_changer:
            return
        if self.app._loading:
            self._pending_latency_settings = settings
            return

        vc = self.voice_changer
        mode_changed = settings["latency_mode"] != vc.config.latency_mode
        chunk_changed = abs(settings["chunk_sec"] - vc.config.chunk_sec) >= 1e-9

        def apply_values() -> None:
            vc.set_latency_mode(settings["latency_mode"])
            vc.set_denoise(
                self.app.use_denoise_var.get()
                and settings["latency_mode"] not in {"sub100", "frontier"},
                self.app.denoise_method_var.get(),
                self.app.denoise_strength_slider.get(),
            )
            vc.set_prebuffer_chunks(settings["prebuffer_chunks"])
            vc.set_buffer_margin(settings["buffer_margin"])
            vc.set_overlap(settings["overlap_sec"])
            vc.set_crossfade(settings["crossfade_sec"])

        if not vc.is_running:
            apply_values()
            vc.set_chunk_sec(settings["chunk_sec"])
            return
        if not chunk_changed and not mode_changed:
            apply_values()
            return  # no chunk change → no restart needed

        # Restart with new chunk size: stop synchronously (fast, streams
        # must close on the main thread), then re-warm asynchronously.
        self._begin_loading("再起動中...")
        vc.stop()
        apply_values()
        vc.set_chunk_sec(settings["chunk_sec"])  # not running → just stores
        self._prepare_and_start_async(lambda vc=vc: vc)

    def apply_postprocess_config(self, cfg) -> None:
        if self.voice_changer and hasattr(self.voice_changer, "set_postprocess_config"):
            self.voice_changer.set_postprocess_config(cfg)

    def _on_stats_update(self, stats: RealtimeStats) -> None:
        """Handle stats update from voice changer."""

        # Update UI from main thread (status bar + input/output level meters).
        # The input meter is driven from pipeline stats so it works even on
        # exclusive ASIO devices, where a standalone monitor stream cannot open.
        def _apply_ui(s: RealtimeStats = stats) -> None:
            self.app.status_bar.update_stats(s)
            self.app.update_output_meter(s)
            self.app.audio_settings.set_input_level_db(s.input_rms_db, s.input_peak_db)

        self.app.after(0, _apply_ui)

        # Show buffer warning dialogs only for SUSTAINED problems (see the
        # module-level policy constants).  Session-cumulative counts fired a
        # modal dialog every session due to the unavoidable startup transient.
        now = time.monotonic()
        if self._session_start == 0.0:
            self._session_start = now

        # Counters shrinking means the pipeline restarted internally
        # (e.g. chunk size change): resync and re-arm the grace period,
        # since a restart has its own startup transient.
        if self._warn_history and (
            stats.buffer_underruns < self._warn_history[-1][1]
            or stats.buffer_overruns < self._warn_history[-1][2]
        ):
            self._warn_history.clear()
            self._session_start = now

        if now - self._session_start < BUFFER_WARNING_GRACE_SEC:
            return

        self._warn_history.append((now, stats.buffer_underruns, stats.buffer_overruns))
        while (
            len(self._warn_history) > 1
            and now - self._warn_history[0][0] > BUFFER_WARNING_WINDOW_SEC
        ):
            self._warn_history.popleft()
        base = self._warn_history[0]
        window_underruns = stats.buffer_underruns - base[1]
        window_overruns = stats.buffer_overruns - base[2]

        # Buffer underrun warning
        if (
            window_underruns >= BUFFER_WARNING_THRESHOLD
            and not self._buffer_warning_shown["underrun"]
        ):
            self._buffer_warning_shown["underrun"] = True
            self.app.after(0, self._show_buffer_underrun_warning)

        # Buffer overrun warning
        if (
            window_overruns >= BUFFER_WARNING_THRESHOLD
            and not self._buffer_warning_shown["overrun"]
        ):
            self._buffer_warning_shown["overrun"] = True
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
            "   • 0.5 → 1.0",
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
            "4. デノイズを無効化してテスト",
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
