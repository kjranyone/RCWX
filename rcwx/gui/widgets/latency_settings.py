"""Latency settings widget with auto-derived parameters."""

from __future__ import annotations

import math
from typing import Callable, Optional

import customtkinter as ctk


def _minimum_chunk_ms(f0_method: str, latency_mode: str = "sub100") -> int:
    """Return the supported micro-hop floor for an F0 backend."""
    if latency_mode == "frontier" and f0_method in {"swiftf0", "none"}:
        return 20
    return {"rmvpe": 320, "fcpe": 100}.get(f0_method, 40)


def _chunk_slider_spec(
    f0_method: str,
    latency_mode: str,
) -> tuple[int, int, int]:
    """Return the effective slider minimum, maximum, and step in ms."""
    minimum = _minimum_chunk_ms(f0_method, latency_mode)
    if latency_mode == "frontier" and f0_method in {"swiftf0", "none"}:
        return minimum, 100, 20
    return minimum, 600, 20


def _auto_params(chunk_sec: float, latency_mode: str = "balanced") -> dict:
    """Derive latency parameters automatically from chunk_sec.

    Returns dict with overlap_sec, crossfade_sec, sola_search_ms,
    prebuffer_chunks, buffer_margin, use_sola.
    """
    # overlap: 100% of chunk for maximum HuBERT context continuity,
    # clamped [60ms, 300ms], rounded to 20ms (HuBERT frame boundary)
    overlap_ms = max(60, min(300, chunk_sec * 1000))
    overlap_ms = round(overlap_ms / 20) * 20

    aggressive = latency_mode in {"aggressive", "sub100", "frontier"}
    sub100 = latency_mode == "sub100"
    frontier = latency_mode == "frontier"

    # Aggressive keeps enough overlap for a short SOLA splice while avoiding
    # the 50-80ms hold-back used by larger balanced chunks.
    crossfade_ratio = 0.10 if aggressive else 0.25
    crossfade_max_ms = 20 if aggressive else 80
    crossfade_ms = min(crossfade_max_ms, chunk_sec * 1000 * crossfade_ratio)
    crossfade_ms = round(crossfade_ms / 10) * 10
    crossfade_ms = max(10, crossfade_ms)

    # SOLA search window: must cover one period of the lowest expected
    # output F0 so the splice can phase-align (70Hz -> 14.3ms, + margin).
    # Doesn't affect latency; larger windows only add sola_extra compute.
    sola_search_ms = 15.0

    return {
        "overlap_sec": overlap_ms / 1000,
        "crossfade_sec": crossfade_ms / 1000,
        "sola_search_ms": sola_search_ms,
        "latency_mode": (
            "frontier"
            if frontier
            else "sub100"
            if sub100
            else "aggressive"
            if aggressive
            else "balanced"
        ),
        "prebuffer_chunks": 3 if frontier else 2 if sub100 else 1,
        "buffer_margin": 0.1 if frontier or sub100 else 0.25 if aggressive else 0.5,
        "use_sola": True,
    }


class LatencySettingsFrame(ctk.CTkFrame):
    """
    Latency settings widget.

    Chunk size and latency mode are user-controllable. Other latency
    parameters are derived automatically and shown as read-only labels.
    """

    def __init__(
        self,
        master: ctk.CTk,
        on_settings_changed: Optional[Callable[[], None]] = None,
        f0_method: str = "rmvpe",
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_settings_changed = on_settings_changed
        self.f0_method = f0_method

        # Default chunk size
        self.chunk_sec = max(0.16, _minimum_chunk_ms(f0_method) / 1000)
        self.latency_mode = "balanced"

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

        # --- Latency mode ---
        mode_frame = ctk.CTkFrame(frame, fg_color="transparent")
        mode_frame.grid(row=0, column=0, padx=10, pady=(5, 3), sticky="ew")
        ctk.CTkLabel(
            mode_frame,
            text="レイテンシモード",
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, padx=(0, 10), sticky="w")
        self.mode_control = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Balanced", "Aggressive", "Sub-100", "Frontier"],
            command=self._on_mode_change,
            width=300,
        )
        self.mode_control.set("Balanced")
        self.mode_control.grid(row=0, column=1, sticky="e")
        mode_frame.grid_columnconfigure(1, weight=1)

        # --- Chunk size slider ---
        ctk.CTkLabel(frame, text="チャンクサイズ", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, padx=10, pady=(2, 0), sticky="w"
        )
        slider_frame = ctk.CTkFrame(frame, fg_color="transparent")
        slider_frame.grid(row=2, column=0, padx=10, pady=(0, 2), sticky="ew")

        min_chunk_ms, max_chunk_ms, step_ms = _chunk_slider_spec(
            self.f0_method,
            self.latency_mode,
        )
        self.chunk_slider = ctk.CTkSlider(
            slider_frame,
            from_=min_chunk_ms,
            to=max_chunk_ms,
            number_of_steps=(max_chunk_ms - min_chunk_ms) // step_ms,
            width=320,
            command=self._on_chunk_change,
        )
        rounded_ms = self._round_to_frame_boundary(self.chunk_sec * 1000)
        self.chunk_slider.set(rounded_ms)
        self.chunk_slider.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        self.chunk_value = ctk.CTkLabel(slider_frame, text=f"{int(rounded_ms)}ms", width=50)
        self.chunk_value.grid(row=0, column=1)
        slider_frame.grid_columnconfigure(0, weight=1, minsize=320)

        # --- Separator ---
        separator = ctk.CTkFrame(frame, height=1, fg_color="gray50")
        separator.grid(row=3, column=0, padx=10, pady=(6, 2), sticky="ew")

        # --- Auto settings header ---
        auto_header = ctk.CTkLabel(
            frame,
            text="自動設定:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray70",
        )
        auto_header.grid(row=4, column=0, padx=10, pady=(2, 0), sticky="w")

        # --- Read-only auto parameter labels ---
        auto = _auto_params(self.chunk_sec, self.latency_mode)

        # Each auto param: label + value in a row
        self.auto_labels: dict[str, ctk.CTkLabel] = {}

        auto_display = [
            ("overlap", "オーバーラップ", f"{int(auto['overlap_sec'] * 1000)}ms"),
            ("crossfade", "クロスフェード", f"{int(auto['crossfade_sec'] * 1000)}ms"),
            ("prebuffer", "プリバッファ", f"{auto['prebuffer_chunks']}チャンク"),
            ("margin", "バッファマージン", f"{auto['buffer_margin']:.1f}x"),
        ]

        for i, (key, label_text, value_text) in enumerate(auto_display):
            row = 5 + i
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
        separator2.grid(row=9, column=0, padx=10, pady=(4, 2), sticky="ew")

        # --- Estimated latency display ---
        self.estimate_label = ctk.CTkLabel(
            frame,
            text="推定レイテンシ: --ms",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#66b3ff",
        )
        self.estimate_label.grid(row=10, column=0, padx=10, pady=(3, 5), sticky="w")

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

    def _on_mode_change(self, value: str) -> None:
        """Switch between the stable and low-buffer latency policies."""
        self.latency_mode = "sub100" if value == "Sub-100" else value.lower()
        min_ms, _, _ = self._configure_chunk_slider()
        if self.latency_mode in {"sub100", "frontier"}:
            target_ms = min_ms
            self.chunk_sec = target_ms / 1000
            self.chunk_slider.set(target_ms)
            self.chunk_value.configure(text=f"{target_ms}ms")
        self._update_auto_labels()
        self._update_estimate()
        self._notify_change()

    def set_f0_method(self, method: str) -> bool:
        """Apply the backend-specific hop floor and return whether it clamped."""
        self.f0_method = method
        min_ms, max_ms, _ = self._configure_chunk_slider()
        current_ms = self._round_to_frame_boundary(self.chunk_sec * 1000)
        clamped = current_ms < min_ms or current_ms > max_ms
        if clamped:
            current_ms = max(min_ms, min(max_ms, current_ms))
            self.chunk_sec = current_ms / 1000
        self.chunk_slider.set(current_ms)
        self.chunk_value.configure(text=f"{current_ms}ms")
        self._update_auto_labels()
        self._update_estimate()
        return clamped

    def _configure_chunk_slider(self) -> tuple[int, int, int]:
        """Apply the range matching the active mode and F0 backend."""
        min_ms, max_ms, step_ms = _chunk_slider_spec(
            self.f0_method,
            self.latency_mode,
        )
        self.chunk_slider.configure(
            from_=min_ms,
            to=max_ms,
            number_of_steps=(max_ms - min_ms) // step_ms,
        )
        return min_ms, max_ms, step_ms

    def _update_auto_labels(self) -> None:
        """Update read-only auto-parameter labels from current chunk_sec."""
        auto = _auto_params(self.chunk_sec, self.latency_mode)
        self.auto_labels["overlap"].configure(text=f"{int(auto['overlap_sec'] * 1000)}ms")
        self.auto_labels["crossfade"].configure(text=f"{int(auto['crossfade_sec'] * 1000)}ms")
        self.auto_labels["prebuffer"].configure(text=f"{auto['prebuffer_chunks']}チャンク")
        self.auto_labels["margin"].configure(text=f"{auto['buffer_margin']:.1f}x")

    def _update_estimate(self) -> None:
        """Update estimated latency display.

        Latency components:
        - Input/output hop: one chunk period
        - Inference: nominal XPU processing estimate
        - Output buffer: policy's persistent floor target
        - SOLA hold-back: crossfade_sec
        """
        auto = _auto_params(self.chunk_sec, self.latency_mode)
        if self.latency_mode == "frontier":
            inference_est = 15
            buffer_est = self.chunk_sec * 500
        elif self.latency_mode == "sub100":
            inference_est = 20
            buffer_est = self.chunk_sec * 500
        elif self.latency_mode == "aggressive":
            inference_est = 35
            buffer_est = self.chunk_sec * 250
        else:
            inference_est = 50
            buffer_est = self.chunk_sec * 1000
        sola_est = auto["crossfade_sec"] * 1000
        if self.latency_mode == "frontier":
            # Frontier retains crossfade+search, rounded to a 10ms model frame.
            sola_est = math.ceil(
                (sola_est + auto["sola_search_ms"]) / 10
            ) * 10
        total_est = self.chunk_sec * 1000 + inference_est + buffer_est + sola_est

        self.estimate_label.configure(text=f"推定レイテンシ: ~{int(total_est)}ms")

    def _notify_change(self) -> None:
        """Notify that settings have changed."""
        if self.on_settings_changed:
            self.on_settings_changed()

    def get_settings(self) -> dict:
        """Get current latency settings as a dictionary.

        Returns all parameters including auto-derived ones.
        """
        auto = _auto_params(self.chunk_sec, self.latency_mode)
        return {
            "chunk_sec": self.chunk_sec,
            "latency_mode": self.latency_mode,
            "prebuffer_chunks": auto["prebuffer_chunks"],
            "buffer_margin": auto["buffer_margin"],
            "overlap_sec": auto["overlap_sec"],
            "crossfade_sec": auto["crossfade_sec"],
            "sola_search_ms": auto["sola_search_ms"],
            "use_sola": auto["use_sola"],
        }

    def set_values(
        self,
        chunk_sec: float,
        latency_mode: str = "balanced",
        f0_method: Optional[str] = None,
    ) -> None:
        """Restore chunk_sec from saved settings.

        All other parameters are auto-derived.
        """
        if f0_method is not None:
            self.f0_method = f0_method
        self.latency_mode = (
            latency_mode
            if latency_mode in {"balanced", "aggressive", "sub100", "frontier"}
            else "balanced"
        )
        min_ms, max_ms, _ = self._configure_chunk_slider()
        rounded_ms = max(
            min_ms,
            self._round_to_frame_boundary(chunk_sec * 1000),
        )
        rounded_ms = min(max_ms, rounded_ms)
        self.chunk_sec = rounded_ms / 1000
        # Update slider
        self.chunk_slider.set(rounded_ms)
        self.chunk_value.configure(text=f"{rounded_ms}ms")
        self.mode_control.set(
            "Sub-100" if self.latency_mode == "sub100" else self.latency_mode.title()
        )

        # Update auto labels and estimate
        self._update_auto_labels()
        self._update_estimate()
