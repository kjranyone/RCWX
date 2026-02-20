"""Post-processing settings widget (treble boost + limiter)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import customtkinter as ctk

from rcwx.config import PostprocessConfig

logger = logging.getLogger(__name__)


class PostprocessSettingsFrame(ctk.CTkFrame):
    def __init__(
        self,
        master: ctk.CTk,
        config: PostprocessConfig,
        on_settings_changed: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self._config = config
        self.on_settings_changed = on_settings_changed

        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self) -> None:
        label = ctk.CTkLabel(self, text="ポスト処理", font=ctk.CTkFont(weight="bold"))
        label.pack(anchor="w", padx=10, pady=(5, 3))

        self.enabled_var = ctk.BooleanVar(value=True)
        self.enabled_cb = ctk.CTkCheckBox(
            self,
            text="有効",
            variable=self.enabled_var,
            command=self._on_enabled_change,
        )
        self.enabled_cb.pack(anchor="w", padx=10, pady=2)

        treble_frame = ctk.CTkFrame(self, fg_color="transparent")
        treble_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(treble_frame, text="高域ブースト:", width=90, anchor="w").grid(row=0, column=0)
        self.treble_slider = ctk.CTkSlider(
            treble_frame,
            from_=0,
            to=8,
            number_of_steps=16,
            width=150,
            command=self._on_treble_change,
        )
        self.treble_slider.grid(row=0, column=1, padx=5)
        self.treble_label = ctk.CTkLabel(treble_frame, text="+4.0 dB", width=60)
        self.treble_label.grid(row=0, column=2)

        cutoff_frame = ctk.CTkFrame(self, fg_color="transparent")
        cutoff_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(cutoff_frame, text="カットオフ:", width=90, anchor="w").grid(row=0, column=0)
        self.cutoff_slider = ctk.CTkSlider(
            cutoff_frame,
            from_=1500,
            to=6000,
            number_of_steps=18,
            width=150,
            command=self._on_cutoff_change,
        )
        self.cutoff_slider.grid(row=0, column=1, padx=5)
        self.cutoff_label = ctk.CTkLabel(cutoff_frame, text="2800 Hz", width=60)
        self.cutoff_label.grid(row=0, column=2)

        limiter_frame = ctk.CTkFrame(self, fg_color="transparent")
        limiter_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(limiter_frame, text="リミッター:", width=90, anchor="w").grid(row=0, column=0)
        self.limiter_slider = ctk.CTkSlider(
            limiter_frame,
            from_=-6,
            to=0,
            number_of_steps=12,
            width=150,
            command=self._on_limiter_change,
        )
        self.limiter_slider.grid(row=0, column=1, padx=5)
        self.limiter_label = ctk.CTkLabel(limiter_frame, text="-1 dB", width=60)
        self.limiter_label.grid(row=0, column=2)

        release_frame = ctk.CTkFrame(self, fg_color="transparent")
        release_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(release_frame, text="リリース:", width=90, anchor="w").grid(row=0, column=0)
        self.release_slider = ctk.CTkSlider(
            release_frame,
            from_=20,
            to=200,
            number_of_steps=18,
            width=150,
            command=self._on_release_change,
        )
        self.release_slider.grid(row=0, column=1, padx=5)
        self.release_label = ctk.CTkLabel(release_frame, text="80 ms", width=60)
        self.release_label.grid(row=0, column=2)

        self._update_slider_states()

    def _load_from_config(self) -> None:
        self.enabled_var.set(self._config.enabled)
        self.treble_slider.set(self._config.treble_boost_db)
        self.cutoff_slider.set(self._config.treble_cutoff_hz)
        self.limiter_slider.set(self._config.limiter_threshold_db)
        self.release_slider.set(self._config.limiter_release_ms)
        self._update_labels()
        self._update_slider_states()

    def _update_labels(self) -> None:
        self.treble_label.configure(text=f"+{self._config.treble_boost_db:.1f} dB")
        self.cutoff_label.configure(text=f"{int(self._config.treble_cutoff_hz)} Hz")
        self.limiter_label.configure(text=f"{self._config.limiter_threshold_db:.0f} dB")
        self.release_label.configure(text=f"{int(self._config.limiter_release_ms)} ms")

    def _update_slider_states(self) -> None:
        state = "normal" if self.enabled_var.get() else "disabled"
        self.treble_slider.configure(state=state)
        self.cutoff_slider.configure(state=state)
        self.limiter_slider.configure(state=state)
        self.release_slider.configure(state=state)

    def _on_enabled_change(self) -> None:
        self._config.enabled = self.enabled_var.get()
        self._update_slider_states()
        if self.on_settings_changed:
            self.on_settings_changed()

    def _on_treble_change(self, value: float) -> None:
        self._config.treble_boost_db = round(value, 1)
        self.treble_label.configure(text=f"+{self._config.treble_boost_db:.1f} dB")
        if self.on_settings_changed:
            self.on_settings_changed()

    def _on_cutoff_change(self, value: float) -> None:
        self._config.treble_cutoff_hz = round(value / 100) * 100
        self.cutoff_label.configure(text=f"{int(self._config.treble_cutoff_hz)} Hz")
        if self.on_settings_changed:
            self.on_settings_changed()

    def _on_limiter_change(self, value: float) -> None:
        self._config.limiter_threshold_db = round(value)
        self.limiter_label.configure(text=f"{self._config.limiter_threshold_db:.0f} dB")
        if self.on_settings_changed:
            self.on_settings_changed()

    def _on_release_change(self, value: float) -> None:
        self._config.limiter_release_ms = round(value / 10) * 10
        self.release_label.configure(text=f"{int(self._config.limiter_release_ms)} ms")
        if self.on_settings_changed:
            self.on_settings_changed()

    def get_config(self) -> PostprocessConfig:
        return self._config
