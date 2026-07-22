"""Latency monitor widget."""

from __future__ import annotations

import time
from collections import deque

import customtkinter as ctk

from rcwx.pipeline.realtime_unified import RealtimeStats

# Counters are cumulative for the whole session; highlight them only while
# they are actually increasing so a single early hiccup doesn't stay red.
RECENT_ISSUE_SEC = 5.0

# The status bar shows counts within this sliding window (not session
# totals): once a problem stops, the number decays to zero and the
# indicator disappears instead of accumulating forever.
DISPLAY_WINDOW_SEC = 60.0

# Fixed pixel widths so live text updates do not reflow the whole footer.
_FONT_SIZE = 11
_W_DEVICE = 170
_W_LATENCY = 120
_W_INFERENCE = 170
_W_INDEX = 90
_W_GPU = 80
_W_BUFFER = 170
_W_STATUS = 24
_W_SEP = 12


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

    def _label(
        self,
        text: str,
        *,
        width: int,
        anchor: str = "w",
        text_color: str | tuple[str, str] | None = None,
        size: int = _FONT_SIZE,
    ) -> ctk.CTkLabel:
        kwargs: dict = {
            "text": text,
            "font": ctk.CTkFont(size=size),
            "width": width,
            "anchor": anchor,
        }
        if text_color is not None:
            kwargs["text_color"] = text_color
        return ctk.CTkLabel(self, **kwargs)

    def _setup_ui(self) -> None:
        """Setup the UI components with fixed column geometry."""
        font = ctk.CTkFont(size=_FONT_SIZE)
        col = 0

        def place(widget: ctk.CTkLabel, sticky: str = "w") -> None:
            nonlocal col
            widget.grid(row=0, column=col, padx=(6 if col else 10, 6), pady=5, sticky=sticky)
            col += 1

        def place_sep() -> ctk.CTkLabel:
            nonlocal col
            sep = ctk.CTkLabel(self, text="|", font=font, width=_W_SEP, anchor="center")
            sep.grid(row=0, column=col, padx=2, pady=5)
            col += 1
            return sep

        self.device_label = self._label("デバイス: 未検出", width=_W_DEVICE)
        place(self.device_label)
        self.sep1 = place_sep()

        self.latency_label = self._label("レイテンシ: ---ms", width=_W_LATENCY)
        place(self.latency_label)
        self.sep2 = place_sep()

        self.inference_label = self._label("推論: ---ms (p95 ---ms)", width=_W_INFERENCE)
        place(self.inference_label)
        self.sep3 = place_sep()

        self.index_label = self._label("Index: --", width=_W_INDEX, text_color="gray")
        place(self.index_label)
        self.sep4 = place_sep()

        self.gpu_label = self._label("GPU: --%", width=_W_GPU, text_color="gray")
        place(self.gpu_label)
        self.sep5 = place_sep()

        # Always reserve warning space; never grid_remove neighbors (that reflow).
        self.buffer_warning_label = self._label(
            "",
            width=_W_BUFFER,
            text_color="gray",
        )
        place(self.buffer_warning_label)
        self.sep6 = place_sep()

        self.status_indicator = self._label(
            "●",
            width=_W_STATUS,
            anchor="center",
            text_color="gray",
            size=14,
        )
        place(self.status_indicator, sticky="e")

        # Trailing flex so the status dot stays pinned without shifting left fields.
        self.grid_columnconfigure(col - 1, weight=0)
        self.grid_columnconfigure(col, weight=1)

    @staticmethod
    def _fmt_ms(value: float, *, width: int = 3) -> str:
        """Format milliseconds with a stable digit field (clamped for display)."""
        n = int(round(value))
        if n < 0:
            n = 0
        if n > 9999:
            n = 9999
        return f"{n:{width}d}"

    def set_device(self, name: str) -> None:
        """Set the device name."""
        self._device_name = name
        # Truncate long GPU names so the device column width stays stable.
        display = name if len(name) <= 18 else name[:17] + "…"
        self.device_label.configure(text=f"デバイス: {display}")

    def update_stats(self, stats: RealtimeStats) -> None:
        """Update the display with new stats."""
        self._latency_ms = stats.latency_ms

        self.latency_label.configure(
            text=f"レイテンシ: {self._fmt_ms(stats.latency_ms)}ms"
        )
        self.inference_label.configure(
            text=(
                f"推論: {self._fmt_ms(stats.inference_ms)}ms "
                f"(p95 {self._fmt_ms(stats.inference_p95_ms)}ms)"
            )
        )

        # Update GPU memory display (always same string shape)
        pct = max(0.0, min(100.0, float(stats.gpu_memory_percent)))
        if pct > 0:
            color = "#ff3333" if pct > 80 else "#ffaa00" if pct > 60 else "#88ff88"
            self.gpu_label.configure(text=f"GPU: {pct:3.0f}%", text_color=color)
        else:
            self.gpu_label.configure(text="GPU: --%", text_color="gray")

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

        # Fixed three fields so length does not thrash when one counter appears.
        # Cap display at 99 so the string stays within the reserved width.
        def _cap(n: int) -> int:
            return max(0, min(99, int(n)))

        if win_underruns > 0 or win_overruns > 0 or win_trims > 0:
            if recent_issue:
                color = "#ff3333"
            elif recent_trim:
                color = "#ffaa00"
            else:
                color = "gray"
            self.buffer_warning_label.configure(
                text=(
                    f"U:{_cap(win_underruns):02d} "
                    f"D:{_cap(win_overruns):02d} "
                    f"T:{_cap(win_trims):02d}"
                ),
                text_color=color,
            )
        else:
            self.buffer_warning_label.configure(text="", text_color="gray")

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
            self.latency_label.configure(text="レイテンシ: ---ms")
            self.inference_label.configure(text="推論: ---ms (p95 ---ms)")
            self.gpu_label.configure(text="GPU: --%", text_color="gray")
            self.buffer_warning_label.configure(text="", text_color="gray")

    def set_loading(self) -> None:
        """Set loading status."""
        self.status_indicator.configure(text="◐", text_color="#ffff00")
        self.latency_label.configure(text="レイテンシ: 読込中")
        self.inference_label.configure(text="推論: ---ms (p95 ---ms)")
        self.gpu_label.configure(text="GPU: --%", text_color="gray")
        self.buffer_warning_label.configure(text="", text_color="gray")

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
                text=f"Index: {index_rate:3.0%}",
                text_color=color,
            )

    def update_index_rate(self, index_rate: float) -> None:
        """Update just the index rate (keeps loaded status)."""
        self.set_index_status(self._index_loaded, index_rate)
