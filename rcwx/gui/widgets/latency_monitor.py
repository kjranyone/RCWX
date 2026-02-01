"""Latency monitor widget."""

from __future__ import annotations

from typing import Optional

import customtkinter as ctk

from rcwx.pipeline.realtime import RealtimeStats


class LatencyMonitor(ctk.CTkFrame):
    """
    Status bar widget showing latency and device info.
    """

    def __init__(
        self,
        master: ctk.CTk,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self._device_name: str = "未検出"
        self._latency_ms: float = 0.0
        self._cpu_percent: float = 0.0
        self._index_loaded: bool = False
        self._index_rate: float = 0.0

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Device label
        self.device_label = ctk.CTkLabel(
            self,
            text="デバイス: 未検出",
            font=ctk.CTkFont(size=11),
        )
        self.device_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Separator
        self.sep1 = ctk.CTkLabel(self, text="|", font=ctk.CTkFont(size=11))
        self.sep1.grid(row=0, column=1, padx=5, pady=5)

        # Latency label
        self.latency_label = ctk.CTkLabel(
            self,
            text="レイテンシ: --ms",
            font=ctk.CTkFont(size=11),
        )
        self.latency_label.grid(row=0, column=2, padx=10, pady=5)

        # Separator
        self.sep2 = ctk.CTkLabel(self, text="|", font=ctk.CTkFont(size=11))
        self.sep2.grid(row=0, column=3, padx=5, pady=5)

        # Inference time label
        self.inference_label = ctk.CTkLabel(
            self,
            text="推論: --ms",
            font=ctk.CTkFont(size=11),
        )
        self.inference_label.grid(row=0, column=4, padx=10, pady=5)

        # Separator
        self.sep3 = ctk.CTkLabel(self, text="|", font=ctk.CTkFont(size=11))
        self.sep3.grid(row=0, column=5, padx=5, pady=5)

        # Index status label
        self.index_label = ctk.CTkLabel(
            self,
            text="Index: --",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.index_label.grid(row=0, column=6, padx=10, pady=5)

        # Separator
        self.sep4 = ctk.CTkLabel(self, text="|", font=ctk.CTkFont(size=11))
        self.sep4.grid(row=0, column=7, padx=5, pady=5)

        # Buffer warning label
        self.buffer_warning_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.buffer_warning_label.grid(row=0, column=8, padx=10, pady=5)

        # Separator
        self.sep5 = ctk.CTkLabel(self, text="|", font=ctk.CTkFont(size=11))
        self.sep5.grid(row=0, column=9, padx=5, pady=5)

        # Status indicator
        self.status_indicator = ctk.CTkLabel(
            self,
            text="●",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        )
        self.status_indicator.grid(row=0, column=10, padx=10, pady=5, sticky="e")

        # Configure grid
        self.grid_columnconfigure(10, weight=1)

    def set_device(self, name: str) -> None:
        """Set the device name."""
        self._device_name = name
        self.device_label.configure(text=f"デバイス: {name}")

    def update_stats(self, stats: RealtimeStats) -> None:
        """Update the display with new stats."""
        self._latency_ms = stats.latency_ms

        self.latency_label.configure(text=f"レイテンシ: {stats.latency_ms:.0f}ms")
        self.inference_label.configure(text=f"推論: {stats.inference_ms:.0f}ms")

        # Update buffer warning display
        if stats.buffer_underruns > 0 or stats.buffer_overruns > 0:
            warning_parts = []
            if stats.buffer_underruns > 0:
                warning_parts.append(f"⚠ UNDER:{stats.buffer_underruns}")
            if stats.buffer_overruns > 0:
                warning_parts.append(f"DROP:{stats.buffer_overruns}")

            self.buffer_warning_label.configure(
                text=" ".join(warning_parts),
                text_color="#ff3333"
            )
            # Hide separator when warning is shown
            self.sep5.grid_remove()
        else:
            self.buffer_warning_label.configure(text="")
            # Show separator when no warning
            self.sep5.grid()

        # Update status color based on latency and buffer issues
        if stats.buffer_underruns > 0 or stats.buffer_overruns > 0:
            color = "#ff3333"  # Red - buffer issues
        elif stats.latency_ms < 150:
            color = "#00ff00"  # Green
        elif stats.latency_ms < 250:
            color = "#ffff00"  # Yellow
        else:
            color = "#ff6600"  # Orange

        self.status_indicator.configure(text_color=color)

    def set_running(self, running: bool) -> None:
        """Set the running status."""
        if running:
            self.status_indicator.configure(text="●", text_color="#00ff00")
        else:
            self.status_indicator.configure(text="●", text_color="gray")
            self.latency_label.configure(text="レイテンシ: --ms")
            self.inference_label.configure(text="推論: --ms")
            self.buffer_warning_label.configure(text="")
            self.sep5.grid()  # Show separator when stopped

    def set_loading(self) -> None:
        """Set loading status."""
        self.status_indicator.configure(text="◐", text_color="#ffff00")
        self.latency_label.configure(text="レイテンシ: 読込中...")
        self.inference_label.configure(text="推論: --ms")
        self.buffer_warning_label.configure(text="")
        self.sep5.grid()  # Show separator when loading

    def set_index_status(self, loaded: bool, index_rate: float = 0.0) -> None:
        """Set the index status.

        Args:
            loaded: Whether FAISS index is loaded
            index_rate: Current index_rate setting (0-1)
        """
        self._index_loaded = loaded
        self._index_rate = index_rate

        if not loaded:
            self.index_label.configure(text="Index: なし", text_color="gray")
        elif index_rate <= 0:
            self.index_label.configure(text="Index: OFF", text_color="gray")
        else:
            # Show index rate and indicate it's active
            color = "#00ff00" if index_rate > 0.5 else "#ffff00"
            self.index_label.configure(
                text=f"Index: {index_rate:.0%}",
                text_color=color,
            )

    def update_index_rate(self, index_rate: float) -> None:
        """Update just the index rate (keeps loaded status)."""
        self.set_index_status(self._index_loaded, index_rate)
