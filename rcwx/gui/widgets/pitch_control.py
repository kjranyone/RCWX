"""Pitch control widget."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk


class PitchControl(ctk.CTkFrame):
    """
    Pitch shift and F0 control widget.

    Also exposes a continuous pre-HuBERT pitch ratio control.
    """

    def __init__(
        self,
        master: ctk.CTk,
        on_pitch_changed: Optional[Callable[[int], None]] = None,
        on_f0_mode_changed: Optional[Callable[[bool], None]] = None,
        on_f0_method_changed: Optional[Callable[[str], None]] = None,
        on_pre_hubert_pitch_changed: Optional[Callable[[float], None]] = None,
        on_moe_boost_changed: Optional[Callable[[float], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_pitch_changed = on_pitch_changed
        self.on_f0_mode_changed = on_f0_mode_changed
        self.on_f0_method_changed = on_f0_method_changed
        self.on_pre_hubert_pitch_changed = on_pre_hubert_pitch_changed
        self.on_moe_boost_changed = on_moe_boost_changed

        self._pitch: int = 0
        self._use_f0: bool = True
        self._f0_method: str = "rmvpe"
        self._pre_hubert_pitch_ratio: float = 0.0
        self._moe_boost: float = 0.0

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup UI components."""
        self.pitch_label = ctk.CTkLabel(
            self,
            text="ピッチシフト",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.pitch_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(5, 2))

        self.min_label = ctk.CTkLabel(self, text="-24", font=ctk.CTkFont(size=10))
        self.min_label.grid(row=1, column=0, padx=(10, 5), pady=2)

        self.pitch_slider = ctk.CTkSlider(
            self,
            from_=-24,
            to=24,
            number_of_steps=48,
            width=250,
            command=self._on_slider_change,
        )
        self.pitch_slider.set(0)
        self.pitch_slider.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        self.max_label = ctk.CTkLabel(self, text="+24", font=ctk.CTkFont(size=10))
        self.max_label.grid(row=1, column=2, padx=(5, 10), pady=2)

        self.value_label = ctk.CTkLabel(
            self,
            text="現在値: 0 半音",
            font=ctk.CTkFont(size=12),
        )
        self.value_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=2)

        self.preset_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.preset_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=(2, 5), sticky="ew")

        presets = [
            ("-12", -12),
            ("-5", -5),
            ("0", 0),
            ("+5", 5),
            ("+12", 12),
        ]
        for i, (label, value) in enumerate(presets):
            btn = ctk.CTkButton(
                self.preset_frame,
                text=label,
                width=50,
                command=lambda v=value: self._set_pitch(v),
            )
            btn.grid(row=0, column=i, padx=3, pady=2)

        self.f0_label = ctk.CTkLabel(
            self,
            text="F0モード",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.f0_label.grid(row=4, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 2))

        self.f0_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.f0_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=(0, 5), sticky="ew")

        self.f0_var = ctk.StringVar(value="rmvpe")
        self.rmvpe_rb = ctk.CTkRadioButton(
            self.f0_frame,
            text="RMVPE (高品質)",
            variable=self.f0_var,
            value="rmvpe",
            command=self._on_f0_change,
        )
        self.rmvpe_rb.grid(row=0, column=0, padx=5, pady=2)

        self.fcpe_rb = ctk.CTkRadioButton(
            self.f0_frame,
            text="FCPE (低遅延)",
            variable=self.f0_var,
            value="fcpe",
            command=self._on_f0_change,
        )
        self.fcpe_rb.grid(row=0, column=1, padx=5, pady=2)

        self.none_rb = ctk.CTkRadioButton(
            self.f0_frame,
            text="なし",
            variable=self.f0_var,
            value="none",
            command=self._on_f0_change,
        )
        self.none_rb.grid(row=0, column=2, padx=5, pady=2)

        self.pre_hubert_label = ctk.CTkLabel(
            self,
            text="プレHuBERTシフト比率",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.pre_hubert_label.grid(row=6, column=0, columnspan=3, sticky="w", padx=10, pady=(5, 2))

        self.pre_hubert_min = ctk.CTkLabel(self, text="0.0", font=ctk.CTkFont(size=10))
        self.pre_hubert_min.grid(row=7, column=0, padx=(10, 5), pady=2)

        self.pre_hubert_slider = ctk.CTkSlider(
            self,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            width=250,
            command=self._on_pre_hubert_change,
        )
        self.pre_hubert_slider.set(0.0)
        self.pre_hubert_slider.grid(row=7, column=1, padx=5, pady=2, sticky="ew")

        self.pre_hubert_value = ctk.CTkLabel(self, text="0.00", width=40, font=ctk.CTkFont(size=11))
        self.pre_hubert_value.grid(row=7, column=2, padx=(5, 10), pady=2)

        self.pre_hubert_hint = ctk.CTkLabel(
            self,
            text="推奨: 0.20〜0.40",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        )
        self.pre_hubert_hint.grid(row=8, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 5))

        self.moe_label = ctk.CTkLabel(
            self,
            text="Moe Boost",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.moe_label.grid(row=9, column=0, columnspan=3, sticky="w", padx=10, pady=(5, 2))

        self.moe_min = ctk.CTkLabel(self, text="0.0", font=ctk.CTkFont(size=10))
        self.moe_min.grid(row=10, column=0, padx=(10, 5), pady=2)

        self.moe_slider = ctk.CTkSlider(
            self,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            width=250,
            command=self._on_moe_boost_change,
        )
        self.moe_slider.set(0.0)
        self.moe_slider.grid(row=10, column=1, padx=5, pady=2, sticky="ew")

        self.moe_value = ctk.CTkLabel(self, text="0.00", width=40, font=ctk.CTkFont(size=11))
        self.moe_value.grid(row=10, column=2, padx=(5, 10), pady=2)

        self.moe_hint = ctk.CTkLabel(
            self,
            text="Recommended: 0.40-0.80",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        )
        self.moe_hint.grid(row=11, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 4))

        self.moe_preset_btn = ctk.CTkButton(
            self,
            text="Apply Moe Preset",
            command=self.apply_moe_preset,
        )
        self.moe_preset_btn.grid(row=12, column=0, columnspan=3, padx=10, pady=(0, 6), sticky="ew")

        self.grid_columnconfigure(1, weight=1)

    def _on_slider_change(self, value: float) -> None:
        """Handle pitch slider change."""
        self._pitch = int(round(value))
        self._update_value_label()
        if self.on_pitch_changed:
            self.on_pitch_changed(self._pitch)

    def _set_pitch(self, value: int) -> None:
        """Set pitch to a specific value."""
        self._pitch = value
        self.pitch_slider.set(value)
        self._update_value_label()
        if self.on_pitch_changed:
            self.on_pitch_changed(self._pitch)

    def _update_value_label(self) -> None:
        """Update pitch value label."""
        sign = "+" if self._pitch > 0 else ""
        self.value_label.configure(text=f"現在値: {sign}{self._pitch} 半音")

    def _on_f0_change(self) -> None:
        """Handle F0 mode change."""
        method = self.f0_var.get()
        self._use_f0 = method != "none"
        self._f0_method = method
        if self.on_f0_mode_changed:
            self.on_f0_mode_changed(self._use_f0)
        if self.on_f0_method_changed:
            self.on_f0_method_changed(method)

    def _on_pre_hubert_change(self, value: float) -> None:
        """Handle pre-HuBERT ratio slider change."""
        self._pre_hubert_pitch_ratio = max(0.0, min(1.0, float(value)))
        self.pre_hubert_value.configure(text=f"{self._pre_hubert_pitch_ratio:.2f}")
        if self.on_pre_hubert_pitch_changed:
            self.on_pre_hubert_pitch_changed(self._pre_hubert_pitch_ratio)

    def _on_moe_boost_change(self, value: float) -> None:
        """Handle moe boost slider change."""
        self._moe_boost = max(0.0, min(1.0, float(value)))
        self.moe_value.configure(text=f"{self._moe_boost:.2f}")
        if self.on_moe_boost_changed:
            self.on_moe_boost_changed(self._moe_boost)

    def set_f0_enabled(self, enabled: bool) -> None:
        """Enable or disable F0 controls based on model support."""
        if enabled:
            self.rmvpe_rb.configure(state="normal")
            self.fcpe_rb.configure(state="normal")
            if self.f0_var.get() == "none":
                self.f0_var.set("rmvpe")
            self._use_f0 = True
        else:
            self.rmvpe_rb.configure(state="disabled")
            self.fcpe_rb.configure(state="disabled")
            self.f0_var.set("none")
            self._use_f0 = False

    def set_f0_method(self, method: str) -> None:
        """Set F0 method (rmvpe, fcpe, or none)."""
        self.f0_var.set(method)
        self._use_f0 = method != "none"
        self._f0_method = method

    def set_pitch(self, value: int) -> None:
        """Set pitch shift without invoking callbacks."""
        self._pitch = max(-24, min(24, int(value)))
        self.pitch_slider.set(self._pitch)
        self._update_value_label()

    def set_pre_hubert_pitch(self, enabled: bool) -> None:
        """Backward-compatible bool setter."""
        self.set_pre_hubert_pitch_ratio(0.35 if enabled else 0.0)

    def set_pre_hubert_pitch_ratio(self, ratio: float) -> None:
        """Set pre-HuBERT pitch ratio (0.0-1.0)."""
        self._pre_hubert_pitch_ratio = max(0.0, min(1.0, float(ratio)))
        self.pre_hubert_slider.set(self._pre_hubert_pitch_ratio)
        self.pre_hubert_value.configure(text=f"{self._pre_hubert_pitch_ratio:.2f}")

    def set_moe_boost(self, strength: float) -> None:
        """Set moe boost strength (0.0-1.0)."""
        self._moe_boost = max(0.0, min(1.0, float(strength)))
        self.moe_slider.set(self._moe_boost)
        self.moe_value.configure(text=f"{self._moe_boost:.2f}")

    def apply_moe_preset(self) -> None:
        """Apply a practical cute voice preset."""
        self._set_pitch(8)
        self.set_f0_method("fcpe")
        self.set_pre_hubert_pitch_ratio(0.15)
        self.set_moe_boost(0.70)

        if self.on_f0_mode_changed:
            self.on_f0_mode_changed(True)
        if self.on_f0_method_changed:
            self.on_f0_method_changed("fcpe")
        if self.on_pre_hubert_pitch_changed:
            self.on_pre_hubert_pitch_changed(self._pre_hubert_pitch_ratio)
        if self.on_moe_boost_changed:
            self.on_moe_boost_changed(self._moe_boost)

    @property
    def pitch(self) -> int:
        """Get current pitch shift."""
        return self._pitch

    @property
    def use_f0(self) -> bool:
        """Get current F0 mode."""
        return self._use_f0

    @property
    def f0_method(self) -> str:
        """Get current F0 method (rmvpe, fcpe, or none)."""
        return self.f0_var.get()

    @property
    def pre_hubert_pitch_ratio(self) -> float:
        """Get current pre-HuBERT pitch ratio."""
        return self._pre_hubert_pitch_ratio

    @property
    def moe_boost(self) -> float:
        """Get current moe boost."""
        return self._moe_boost
