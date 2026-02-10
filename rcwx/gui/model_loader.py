"""Model loading and management."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import customtkinter as ctk

from rcwx.device import get_device_name
from rcwx.pipeline.inference import RVCPipeline

if TYPE_CHECKING:
    from rcwx.gui.app import RCWXApp

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Manages RVC model loading and lifecycle.

    Handles:
    - Asynchronous model loading
    - Model info updates
    - Error handling
    - Device detection
    """

    def __init__(self, app: RCWXApp):
        """
        Initialize model loader.

        Args:
            app: Reference to main application
        """
        self.app = app

    def load_model(self, path: str) -> None:
        """
        Load model asynchronously.

        Args:
            path: Path to model file (.pth)
        """
        if self.app._loading:
            return

        self.app._loading = True
        self.app.status_bar.set_loading()
        self.app.start_btn.configure(state="disabled")

        def load_thread():
            try:
                self.app.pipeline = RVCPipeline(
                    path,
                    device=self.app.device_var.get(),
                    dtype=self.app.dtype_var.get(),
                    # Always load with model F0 capability enabled.
                    # Runtime F0 on/off is controlled by f0_method/use_f0 in realtime config.
                    use_f0=True,
                    use_compile=self.app.compile_var.get(),
                    models_dir=self.app.models_dir_entry.get(),
                )
                self.app.pipeline.load()

                # Update UI from main thread
                self.app.after(0, self._on_model_loaded)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                error_msg = str(e)
                self.app.after(0, lambda msg=error_msg: self._on_model_load_error(msg))

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def _on_model_loaded(self) -> None:
        """Called when model is loaded successfully."""
        self.app._loading = False
        self.app.start_btn.configure(state="normal")
        self.app.status_bar.set_running(False)

        # Update model info and device display
        if self.app.pipeline:
            self.app.model_selector.set_model_info(
                has_f0=self.app.pipeline.has_f0,
                version=self.app.pipeline.synthesizer.version if self.app.pipeline.synthesizer else 2,
            )
            self.app.pitch_control.set_f0_enabled(self.app.pipeline.has_f0)

            # Update device name in status bar
            device_name = get_device_name(self.app.pipeline.device)
            self.app.status_bar.set_device(device_name)

            # Update index status
            index_loaded = self.app.pipeline.faiss_index is not None
            if index_loaded:
                n_vectors = self.app.pipeline.faiss_index.ntotal
                self.app.index_status.configure(
                    text=f"Index読込済 ({n_vectors}ベクトル)",
                    text_color="green",
                )
            else:
                self.app.index_status.configure(
                    text="Indexなし",
                    text_color="gray",
                )

            # Update status bar index indicator
            self.app.status_bar.set_index_status(index_loaded, self.app._get_index_rate())

    def _on_model_load_error(self, error: str) -> None:
        """Called when model loading fails."""
        self.app._loading = False
        self.app.start_btn.configure(state="normal")
        self.app.status_bar.set_running(False)

        # Show error dialog
        dialog = ctk.CTkToplevel(self.app)
        dialog.title("エラー")
        dialog.geometry("400x150")
        dialog.transient(self.app)
        dialog.grab_set()

        label = ctk.CTkLabel(
            dialog,
            text=f"モデルの読み込みに失敗しました:\n{error}",
            wraplength=350,
        )
        label.pack(pady=20)

        btn = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
        btn.pack(pady=10)

    def apply_settings(self) -> None:
        """Apply settings and reload model."""
        if self.app.model_selector.model_path:
            # Stop if running
            if self.app._is_running:
                self.app.realtime_controller.stop()

            # Unload current pipeline
            if self.app.pipeline:
                self.app.pipeline.unload()
                self.app.pipeline = None

            # Reload with new settings
            self.load_model(self.app.model_selector.model_path)
