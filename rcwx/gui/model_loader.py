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

        Replaces any existing pipeline only after the new model loads
        successfully, then unloads the previous one so XPU/CUDA weights,
        FAISS tables, and accelerator graphs are not retained across
        model switches.

        Args:
            path: Path to model file (.pth)
        """
        if self.app._loading:
            return

        self.app._loading = True
        self.app.status_bar.set_loading()
        self.app.start_btn.configure(state="disabled")

        # Snapshot all Tk-bound settings on the main thread; the worker
        # thread must never touch Tcl/Tk state (not thread-safe).
        device = self.app.device_var.get()
        dtype = self.app.dtype_var.get()
        use_compile = self.app.compile_var.get()
        models_dir = self.app.models_dir_entry.get()
        # Capture the previous pipeline on the main thread so a concurrent
        # UI action cannot drop the reference before we unload it.
        old_pipeline = self.app.pipeline

        def load_thread():
            try:
                new_pipeline = RVCPipeline(
                    path,
                    device=device,
                    dtype=dtype,
                    # Always load with model F0 capability enabled.
                    # Runtime F0 on/off is controlled by f0_method/use_f0 in realtime config.
                    use_f0=True,
                    use_compile=use_compile,
                    models_dir=models_dir,
                )
                new_pipeline.load()
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                # Keep the previous pipeline so a failed switch is recoverable.
                error_msg = str(e)
                self.app.after(0, lambda msg=error_msg: self._on_model_load_error(msg))
                return

            # Publish the new pipeline first so a failed unload cannot
            # leave the app without a usable model.
            self.app.pipeline = new_pipeline
            if old_pipeline is not None and old_pipeline is not new_pipeline:
                try:
                    logger.info("Unloading previous pipeline to free device memory")
                    old_pipeline.unload()
                except Exception:
                    logger.exception(
                        "Failed to unload previous pipeline; device memory may leak"
                    )

            # Update UI from main thread (errors here must not look like load failure)
            self.app.after(0, self._on_model_loaded)

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
            # Stop if running; load_model unloads the previous pipeline only
            # after the replacement has loaded so a failed reload keeps the
            # last working model in memory.
            if self.app._is_running:
                self.app.realtime_controller.stop()

            self.load_model(self.app.model_selector.model_path)
