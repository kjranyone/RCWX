"""Latency settings widget with advanced controls."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk


class LatencySettingsFrame(ctk.CTkFrame):
    """
    Latency settings widget with chunking mode and parameter controls.

    Provides direct access to all latency-related parameters.
    """

    def __init__(
        self,
        master: ctk.CTk,
        on_settings_changed: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_settings_changed = on_settings_changed

        # Default settings
        self.chunk_sec = 0.25
        self.prebuffer_chunks = 1
        self.buffer_margin = 1.0
        self.context_sec = 0.10
        self.lookahead_sec = 0.0
        self.crossfade_sec = 0.05
        self.use_sola = True
        self.chunking_mode = "wokada"

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Header
        header = ctk.CTkLabel(
            self,
            text="レイテンシ設定",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        header.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(5, 2))

        # Chunking mode selection
        chunking_label = ctk.CTkLabel(
            self,
            text="チャンキング方式",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        chunking_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 2))

        # Description label
        chunking_desc = ctk.CTkLabel(
            self,
            text="w-okada: 低レイテンシ | RVC WebUI: 完全連続 | hybrid: RVC hop + w-okada context",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        )
        chunking_desc.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=0)

        self.chunking_mode_var = ctk.StringVar(value=self.chunking_mode)
        self.chunking_mode_selector = ctk.CTkSegmentedButton(
            self,
            values=["wokada", "rvc_webui", "hybrid"],
            variable=self.chunking_mode_var,
            command=self._on_chunking_mode_change,
        )
        self.chunking_mode_selector.grid(
            row=3, column=0, columnspan=2, padx=10, pady=(2, 5), sticky="ew"
        )

        # Advanced settings frame (always visible)
        self.advanced_frame = ctk.CTkFrame(self)
        self.advanced_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=(5, 3), sticky="ew")
        self._setup_advanced_controls()

        # Configure grid
        self.grid_columnconfigure(0, weight=1)

    def _create_slider_row(
        self,
        parent: ctk.CTkFrame,
        label: str,
        row: int,
        from_: float,
        to: float,
        steps: int,
        value: float,
        value_text: str,
        command: Callable,
        label_width: int = 50,
    ) -> tuple[ctk.CTkSlider, ctk.CTkLabel]:
        """Create a labeled slider row. Returns (slider, value_label)."""
        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=11)).grid(
            row=row, column=0, padx=10, pady=(5, 0), sticky="w"
        )
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row + 1, column=0, padx=10, pady=(0, 2), sticky="ew")

        slider = ctk.CTkSlider(
            frame, from_=from_, to=to, number_of_steps=steps, width=180, command=command
        )
        slider.set(value)
        slider.grid(row=0, column=0, padx=(0, 10))

        value_label = ctk.CTkLabel(frame, text=value_text, width=label_width)
        value_label.grid(row=0, column=1)

        return slider, value_label

    def _setup_advanced_controls(self) -> None:
        """Setup advanced control sliders."""
        frame = self.advanced_frame

        self.chunk_slider, self.chunk_value = self._create_slider_row(
            frame,
            "チャンクサイズ",
            0,
            100,
            600,
            50,
            self.chunk_sec * 1000,
            f"{int(self.chunk_sec * 1000)}ms",
            self._on_chunk_change,
        )
        self.prebuf_slider, self.prebuf_value = self._create_slider_row(
            frame,
            "プリバッファ",
            2,
            0,
            3,
            3,
            self.prebuffer_chunks,
            f"{self.prebuffer_chunks}チャンク",
            self._on_prebuf_change,
            70,
        )
        self.margin_slider, self.margin_value = self._create_slider_row(
            frame,
            "バッファマージン",
            4,
            0.3,
            2.0,
            17,
            self.buffer_margin,
            f"{self.buffer_margin:.1f}x",
            self._on_margin_change,
        )
        self.context_slider, self.context_value = self._create_slider_row(
            frame,
            "コンテキスト",
            6,
            0,
            100,
            20,
            self.context_sec * 1000,
            f"{int(self.context_sec * 1000)}ms",
            self._on_context_change,
        )
        self.crossfade_slider, self.crossfade_value = self._create_slider_row(
            frame,
            "クロスフェード",
            8,
            0,
            100,
            20,
            self.crossfade_sec * 1000,
            f"{int(self.crossfade_sec * 1000)}ms",
            self._on_crossfade_change,
        )
        self.lookahead_slider, self.lookahead_value = self._create_slider_row(
            frame,
            "右コンテキスト (+遅延)",
            10,
            0,
            100,
            20,
            self.lookahead_sec * 1000,
            f"{int(self.lookahead_sec * 1000)}ms",
            self._on_lookahead_change,
        )

        # SOLA checkbox
        self.sola_var = ctk.BooleanVar(value=self.use_sola)
        self.sola_checkbox = ctk.CTkCheckBox(
            frame,
            text="SOLA (位相揃えクロスフェード)",
            variable=self.sola_var,
            command=self._on_sola_change,
        )
        self.sola_checkbox.grid(row=12, column=0, padx=10, pady=(5, 3), sticky="w")

        # Estimated latency display
        self.estimate_label = ctk.CTkLabel(
            frame,
            text="推定レイテンシ: --ms",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#66b3ff",
        )
        self.estimate_label.grid(row=13, column=0, padx=10, pady=(5, 5), sticky="w")

        frame.grid_columnconfigure(0, weight=1)
        self._update_estimate()

    def _on_chunking_mode_change(self, value: str) -> None:
        """Handle chunking mode change."""
        self.chunking_mode = value
        self._notify_change()

    def _on_chunk_change(self, value: float) -> None:
        """Handle chunk size slider change."""
        self.chunk_sec = value / 1000
        self.chunk_value.configure(text=f"{int(value)}ms")
        self._update_estimate()
        self._notify_change()

    def _on_prebuf_change(self, value: float) -> None:
        """Handle prebuffer slider change."""
        self.prebuffer_chunks = int(round(value))
        self.prebuf_value.configure(text=f"{self.prebuffer_chunks}チャンク")
        self._update_estimate()
        self._notify_change()

    def _on_margin_change(self, value: float) -> None:
        """Handle buffer margin slider change."""
        self.buffer_margin = round(value, 1)
        self.margin_value.configure(text=f"{self.buffer_margin:.1f}x")
        self._update_estimate()
        self._notify_change()

    def _on_context_change(self, value: float) -> None:
        """Handle context size slider change."""
        self.context_sec = value / 1000
        self.context_value.configure(text=f"{int(value)}ms")
        self._notify_change()

    def _on_crossfade_change(self, value: float) -> None:
        """Handle crossfade size slider change."""
        self.crossfade_sec = value / 1000
        self.crossfade_value.configure(text=f"{int(value)}ms")
        self._notify_change()

    def _on_lookahead_change(self, value: float) -> None:
        """Handle lookahead slider change."""
        self.lookahead_sec = value / 1000
        self.lookahead_value.configure(text=f"{int(value)}ms")
        self._update_estimate()
        self._notify_change()

    def _on_sola_change(self) -> None:
        """Handle SOLA checkbox change."""
        self.use_sola = self.sola_var.get()
        self._notify_change()

    def _update_estimate(self) -> None:
        """Update estimated latency display."""
        # Estimate: chunk + lookahead + inference(~50ms) + buffer
        # Buffer size ≈ prebuffer * chunk + margin * chunk
        inference_est = 50  # ms
        buffer_est = (self.prebuffer_chunks + self.buffer_margin) * self.chunk_sec * 1000
        lookahead_est = self.lookahead_sec * 1000
        total_est = self.chunk_sec * 1000 + lookahead_est + inference_est + buffer_est

        self.estimate_label.configure(text=f"推定レイテンシ: ~{int(total_est)}ms")

    def _notify_change(self) -> None:
        """Notify that settings have changed."""
        if self.on_settings_changed:
            self.on_settings_changed()

    def get_settings(self) -> dict:
        """Get current latency settings as a dictionary."""
        return {
            "chunk_sec": self.chunk_sec,
            "prebuffer_chunks": self.prebuffer_chunks,
            "buffer_margin": self.buffer_margin,
            "context_sec": self.context_sec,
            "lookahead_sec": self.lookahead_sec,
            "crossfade_sec": self.crossfade_sec,
            "use_sola": self.use_sola,
            "chunking_mode": self.chunking_mode,
        }

    def set_values(
        self,
        chunk_sec: float,
        prebuffer_chunks: int,
        buffer_margin: float,
        context_sec: float,
        lookahead_sec: float,
        crossfade_sec: float,
        use_sola: bool = True,
        chunking_mode: str = "wokada",
    ) -> None:
        """Set individual values directly (for restoring saved settings)."""
        self.chunk_sec = chunk_sec
        self.prebuffer_chunks = prebuffer_chunks
        self.buffer_margin = buffer_margin
        self.context_sec = context_sec
        self.lookahead_sec = lookahead_sec
        self.crossfade_sec = crossfade_sec
        self.use_sola = use_sola
        self.chunking_mode = chunking_mode

        # Update chunking mode UI
        self.chunking_mode_var.set(chunking_mode)

        # Update sliders
        self.chunk_slider.set(chunk_sec * 1000)
        self.chunk_value.configure(text=f"{int(chunk_sec * 1000)}ms")
        self.prebuf_slider.set(prebuffer_chunks)
        self.prebuf_value.configure(text=f"{prebuffer_chunks}チャンク")
        self.margin_slider.set(buffer_margin)
        self.margin_value.configure(text=f"{buffer_margin:.1f}x")
        self.context_slider.set(context_sec * 1000)
        self.context_value.configure(text=f"{int(context_sec * 1000)}ms")
        self.crossfade_slider.set(crossfade_sec * 1000)
        self.crossfade_value.configure(text=f"{int(crossfade_sec * 1000)}ms")
        self.lookahead_slider.set(lookahead_sec * 1000)
        self.lookahead_value.configure(text=f"{int(lookahead_sec * 1000)}ms")
        self.sola_var.set(use_sola)
        self._update_estimate()
