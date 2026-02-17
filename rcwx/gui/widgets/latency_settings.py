"""Latency settings widget with auto-derived parameters."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk


def _auto_params(chunk_sec: float) -> dict:
    """Derive latency parameters automatically from chunk_sec.

    Returns dict with overlap_sec, crossfade_sec, sola_search_ms,
    prebuffer_chunks, buffer_margin, use_sola.
    """
    # overlap: 100% of chunk for maximum HuBERT context continuity,
    # clamped [60ms, 300ms], rounded to 20ms (HuBERT frame boundary)
    overlap_ms = max(60, min(300, chunk_sec * 1000))
    overlap_ms = round(overlap_ms / 20) * 20

    # crossfade: 25% of chunk, clamped [10ms, 80ms], rounded to 10ms
    crossfade_ms = min(80, chunk_sec * 1000 * 0.25)
    crossfade_ms = round(crossfade_ms / 10) * 10
    crossfade_ms = max(10, crossfade_ms)

    # SOLA search window = chunk length (doesn't affect latency)
    sola_search_ms = chunk_sec * 1000

    return {
        "overlap_sec": overlap_ms / 1000,
        "crossfade_sec": crossfade_ms / 1000,
        "sola_search_ms": sola_search_ms,
        "prebuffer_chunks": 1,
        "buffer_margin": 0.5,
        "use_sola": True,
    }


class LatencySettingsFrame(ctk.CTkFrame):
    """
    Latency settings widget.

    Only chunk_sec is user-controllable.  All other latency parameters
    are derived automatically and shown as read-only labels.
    """

    def __init__(
        self,
        master: ctk.CTk,
        on_settings_changed: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_settings_changed = on_settings_changed

        # Default chunk size
        self.chunk_sec = 0.15

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        header = ctk.CTkLabel(
            self,
            text="Latency Settings",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        header.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(5, 2))

        self.advanced_frame = ctk.CTkFrame(self)
        self.advanced_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(5, 3), sticky="ew")
        self._setup_controls()

        self.grid_columnconfigure(0, weight=1)

    def _setup_controls(self) -> None:
        """Setup chunk slider and read-only auto-parameter labels."""
        frame = self.advanced_frame

        # --- Chunk size slider (only user control) ---
        ctk.CTkLabel(frame, text="チャンクサイズ", font=ctk.CTkFont(size=11)).grid(
            row=0, column=0, padx=10, pady=(5, 0), sticky="w"
        )
        slider_frame = ctk.CTkFrame(frame, fg_color="transparent")
        slider_frame.grid(row=1, column=0, padx=10, pady=(0, 2), sticky="ew")

        # Step=20ms to align with HuBERT frame boundary (320 samples @ 16kHz)
        self.chunk_slider = ctk.CTkSlider(
            slider_frame,
            from_=100,
            to=600,
            number_of_steps=25,
            width=180,
            command=self._on_chunk_change,
        )
        rounded_ms = self._round_to_frame_boundary(self.chunk_sec * 1000)
        self.chunk_slider.set(rounded_ms)
        self.chunk_slider.grid(row=0, column=0, padx=(0, 10))

        self.chunk_value = ctk.CTkLabel(slider_frame, text=f"{int(rounded_ms)}ms", width=50)
        self.chunk_value.grid(row=0, column=1)

        # --- Separator ---
        separator = ctk.CTkFrame(frame, height=1, fg_color="gray50")
        separator.grid(row=2, column=0, padx=10, pady=(6, 2), sticky="ew")

        # --- Auto settings header ---
        auto_header = ctk.CTkLabel(
            frame,
            text="自動設定:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray70",
        )
        auto_header.grid(row=3, column=0, padx=10, pady=(2, 0), sticky="w")

        # --- Read-only auto parameter labels ---
        auto = _auto_params(self.chunk_sec)

        # Each auto param: label + value in a row
        self.auto_labels: dict[str, ctk.CTkLabel] = {}

        auto_display = [
            ("overlap", "オーバーラップ", f"{int(auto['overlap_sec'] * 1000)}ms"),
            ("crossfade", "クロスフェード", f"{int(auto['crossfade_sec'] * 1000)}ms"),
            ("prebuffer", "プリバッファ", f"{auto['prebuffer_chunks']}チャンク"),
            ("margin", "バッファマージン", f"{auto['buffer_margin']:.1f}x"),
        ]

        for i, (key, label_text, value_text) in enumerate(auto_display):
            row = 4 + i
            row_frame = ctk.CTkFrame(frame, fg_color="transparent")
            row_frame.grid(row=row, column=0, padx=10, pady=(1, 1), sticky="ew")

            ctk.CTkLabel(
                row_frame,
                text=label_text,
                font=ctk.CTkFont(size=11),
                text_color="gray70",
                width=110,
                anchor="w",
            ).grid(row=0, column=0)

            val_label = ctk.CTkLabel(
                row_frame,
                text=value_text,
                font=ctk.CTkFont(size=11),
                text_color="gray70",
                width=80,
                anchor="w",
            )
            val_label.grid(row=0, column=1)
            self.auto_labels[key] = val_label

        # --- Separator ---
        separator2 = ctk.CTkFrame(frame, height=1, fg_color="gray50")
        separator2.grid(row=8, column=0, padx=10, pady=(4, 2), sticky="ew")

        # --- Estimated latency display ---
        self.estimate_label = ctk.CTkLabel(
            frame,
            text="推定レイテンシ: --ms",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#66b3ff",
        )
        self.estimate_label.grid(row=9, column=0, padx=10, pady=(3, 5), sticky="w")

        frame.grid_columnconfigure(0, weight=1)
        self._update_estimate()

    def _round_to_frame_boundary(self, ms: float) -> int:
        """Round milliseconds to nearest HuBERT frame boundary (20ms)."""
        return int(round(ms / 20) * 20)

    def _on_chunk_change(self, value: float) -> None:
        """Handle chunk size slider change."""
        rounded_ms = self._round_to_frame_boundary(value)
        self.chunk_sec = rounded_ms / 1000
        self.chunk_value.configure(text=f"{rounded_ms}ms")
        self._update_auto_labels()
        self._update_estimate()
        self._notify_change()

    def _update_auto_labels(self) -> None:
        """Update read-only auto-parameter labels from current chunk_sec."""
        auto = _auto_params(self.chunk_sec)
        self.auto_labels["overlap"].configure(text=f"{int(auto['overlap_sec'] * 1000)}ms")
        self.auto_labels["crossfade"].configure(text=f"{int(auto['crossfade_sec'] * 1000)}ms")
        self.auto_labels["prebuffer"].configure(text=f"{auto['prebuffer_chunks']}チャンク")
        self.auto_labels["margin"].configure(text=f"{auto['buffer_margin']:.1f}x")

    def _update_estimate(self) -> None:
        """Update estimated latency display.

        Latency components:
        - Input capture: chunk_sec / 2 (average sample position in chunk)
        - Inference: ~50ms (FCPE on XPU estimate)
        - Output buffer: avg ring ≈ 3/8 hop (cycle: 3/4→2/4→1/4→0)
        - SOLA hold-back: crossfade_sec
        """
        auto = _auto_params(self.chunk_sec)
        inference_est = 50  # ms
        buffer_est = self.chunk_sec * 375  # avg ring ≈ 3/8 hop
        sola_est = auto["crossfade_sec"] * 1000
        total_est = self.chunk_sec * 500 + inference_est + buffer_est + sola_est

        self.estimate_label.configure(text=f"推定レイテンシ: ~{int(total_est)}ms")

    def _notify_change(self) -> None:
        """Notify that settings have changed."""
        if self.on_settings_changed:
            self.on_settings_changed()

    def get_settings(self) -> dict:
        """Get current latency settings as a dictionary.

        Returns all parameters including auto-derived ones.
        """
        auto = _auto_params(self.chunk_sec)
        return {
            "chunk_sec": self.chunk_sec,
            "prebuffer_chunks": auto["prebuffer_chunks"],
            "buffer_margin": auto["buffer_margin"],
            "overlap_sec": auto["overlap_sec"],
            "crossfade_sec": auto["crossfade_sec"],
            "sola_search_ms": auto["sola_search_ms"],
            "use_sola": auto["use_sola"],
        }

    def set_values(self, chunk_sec: float) -> None:
        """Restore chunk_sec from saved settings.

        All other parameters are auto-derived.
        """
        rounded_ms = self._round_to_frame_boundary(chunk_sec * 1000)
        self.chunk_sec = rounded_ms / 1000

        # Update slider
        self.chunk_slider.set(rounded_ms)
        self.chunk_value.configure(text=f"{rounded_ms}ms")

        # Update auto labels and estimate
        self._update_auto_labels()
        self._update_estimate()
