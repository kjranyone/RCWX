"""Latency monitor widget."""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import customtkinter as ctk

from rcwx.pipeline.realtime_unified import RealtimeStats

# Counters are cumulative for the whole session; highlight them only while
# they are actually increasing so a single early hiccup doesn't stay red.
RECENT_ISSUE_SEC = 5.0

# The status bar shows counts within this sliding window (not session
# totals): once a problem stops, the number decays to zero and the
# indicator disappears instead of accumulating forever.
DISPLAY_WINDOW_SEC = 60.0


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

        # Recency tracking for buffer counters (cumulative in RealtimeStats)
        self._prev_underruns: int = 0
        self._prev_overruns: int = 0
        self._prev_trims: int = 0
        self._last_issue_time: float = 0.0  # underrun/overrun increased
        self._last_trim_time: float = 0.0  # drift trim increased
        # Snapshots (time, underruns, overruns, trims) for the sliding window
        self._count_history: deque[tuple[float, int, int, int]] = deque()

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

        # GPU memory label
        self.gpu_label = ctk.CTkLabel(
            self,
            text="GPU: --",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.gpu_label.grid(row=0, column=8, padx=10, pady=5)

        # Separator
        self.sep5 = ctk.CTkLabel(self, text="|", font=ctk.CTkFont(size=11))
        self.sep5.grid(row=0, column=9, padx=5, pady=5)

        # Buffer warning label
        self.buffer_warning_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.buffer_warning_label.grid(row=0, column=10, padx=10, pady=5)

        # Separator
        self.sep6 = ctk.CTkLabel(self, text="|", font=ctk.CTkFont(size=11))
        self.sep6.grid(row=0, column=11, padx=5, pady=5)

        # Status indicator
        self.status_indicator = ctk.CTkLabel(
            self,
            text="●",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        )
        self.status_indicator.grid(row=0, column=12, padx=10, pady=5, sticky="e")

        # Configure grid
        self.grid_columnconfigure(12, weight=1)

    def set_device(self, name: str) -> None:
        """Set the device name."""
        self._device_name = name
        self.device_label.configure(text=f"デバイス: {name}")

    def update_stats(self, stats: RealtimeStats) -> None:
        """Update the display with new stats."""
        self._latency_ms = stats.latency_ms

        self.latency_label.configure(text=f"レイテンシ: {stats.latency_ms:.0f}ms")
        self.inference_label.configure(text=f"推論: {stats.inference_ms:.0f}ms")

        # Update GPU memory display
        pct = stats.gpu_memory_percent
        if pct > 0:
            color = "#ff3333" if pct > 80 else "#ffaa00" if pct > 60 else "#88ff88"
            self.gpu_label.configure(text=f"GPU: {pct:.0f}%", text_color=color)

        # --- Recency tracking ---
        # Counters only ever grow within a session; a decrease means the
        # pipeline restarted (stats.reset()), so resync silently.
        now = time.monotonic()
        if (
            stats.buffer_underruns < self._prev_underruns
            or stats.buffer_overruns < self._prev_overruns
            or stats.buffer_trims < self._prev_trims
        ):
            self._count_history.clear()
            self._last_issue_time = 0.0
            self._last_trim_time = 0.0
        if (
            stats.buffer_underruns > self._prev_underruns
            or stats.buffer_overruns > self._prev_overruns
        ):
            self._last_issue_time = now
        if stats.buffer_trims > self._prev_trims:
            self._last_trim_time = now
        self._prev_underruns = stats.buffer_underruns
        self._prev_overruns = stats.buffer_overruns
        self._prev_trims = stats.buffer_trims

        recent_issue = (
            self._last_issue_time > 0
            and now - self._last_issue_time < RECENT_ISSUE_SEC
        )
        recent_trim = (
            self._last_trim_time > 0
            and now - self._last_trim_time < RECENT_ISSUE_SEC
        )

        # --- Sliding-window counts for display ---
        # Show events within the last DISPLAY_WINDOW_SEC instead of session
        # totals: the numbers decay to zero (and disappear) once resolved.
        self._count_history.append(
            (now, stats.buffer_underruns, stats.buffer_overruns, stats.buffer_trims)
        )
        while (
            len(self._count_history) > 1
            and now - self._count_history[0][0] > DISPLAY_WINDOW_SEC
        ):
            self._count_history.popleft()
        base = self._count_history[0]
        win_underruns = stats.buffer_underruns - base[1]
        win_overruns = stats.buffer_overruns - base[2]
        win_trims = stats.buffer_trims - base[3]

        # Update buffer warning display
        warning_parts = []
        if win_underruns > 0:
            warning_parts.append(f"UNDER:{win_underruns}")
        if win_overruns > 0:
            warning_parts.append(f"DROP:{win_overruns}")
        if win_trims > 0:
            warning_parts.append(f"DRIFT:{win_trims}")

        if warning_parts:
            # Red only while underruns/drops are actively occurring; drift is
            # informational (yellow while active).  Decaying counts are shown
            # dimmed until they leave the window.
            if recent_issue:
                color = "#ff3333"
            elif recent_trim:
                color = "#ffaa00"
            else:
                color = "gray"
            self.buffer_warning_label.configure(
                text=" ".join(warning_parts),
                text_color=color,
            )
            self.sep6.grid_remove()
        else:
            self.buffer_warning_label.configure(text="")
            self.sep6.grid()

        # Update status color based on latency and active buffer issues
        if recent_issue:
            color = "#ff3333"  # Red - buffer issues happening now
        elif stats.latency_ms < 150:
            color = "#00ff00"  # Green
        elif stats.latency_ms < 250:
            color = "#ffff00"  # Yellow
        else:
            color = "#ff6600"  # Orange

        self.status_indicator.configure(text_color=color)

    def set_running(self, running: bool) -> None:
        """Set the running status."""
        # New session: clear recency state so stale warnings don't carry over
        self._prev_underruns = 0
        self._prev_overruns = 0
        self._prev_trims = 0
        self._last_issue_time = 0.0
        self._last_trim_time = 0.0
        self._count_history.clear()
        if running:
            self.status_indicator.configure(text="●", text_color="#00ff00")
        else:
            self.status_indicator.configure(text="●", text_color="gray")
            self.latency_label.configure(text="レイテンシ: --ms")
            self.inference_label.configure(text="推論: --ms")
            self.gpu_label.configure(text="GPU: --", text_color="gray")
            self.buffer_warning_label.configure(text="")
            self.sep6.grid()  # Show separator when stopped

    def set_loading(self) -> None:
        """Set loading status."""
        self.status_indicator.configure(text="◐", text_color="#ffff00")
        self.latency_label.configure(text="レイテンシ: 読込中...")
        self.inference_label.configure(text="推論: --ms")
        self.gpu_label.configure(text="GPU: --", text_color="gray")
        self.buffer_warning_label.configure(text="")
        self.sep6.grid()  # Show separator when loading

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
