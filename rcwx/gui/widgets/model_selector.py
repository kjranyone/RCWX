"""Model selector widget."""

from __future__ import annotations

import logging
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Dict, Optional

import customtkinter as ctk

logger = logging.getLogger(__name__)


class ModelSelector(ctk.CTkFrame):
    """
    Model selection widget.

    Allows users to select and load RVC models via dropdown or file dialog.
    """

    def __init__(
        self,
        master: ctk.CTk,
        on_model_selected: Optional[Callable[[str], None]] = None,
        models_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_model_selected = on_model_selected
        self._model_path: Optional[str] = None
        self._has_f0: Optional[bool] = None
        self._has_index: bool = False
        self._version: Optional[int] = None
        self._model_map: Dict[str, str] = {}  # display_name -> full_path

        self._setup_ui()

        # Initial scan if directory provided
        if models_dir:
            self.scan_directory(models_dir)

    def _setup_ui(self) -> None:
        """Setup the UI components."""
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="モデル選択",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(5, 2))

        # Model dropdown
        self.model_var = ctk.StringVar(value="モデルを選択...")
        self.model_dropdown = ctk.CTkComboBox(
            self,
            variable=self.model_var,
            values=["モデルを選択..."],
            width=300,
            state="readonly",
            command=self._on_dropdown_change,
        )
        self.model_dropdown.grid(row=1, column=0, padx=10, pady=2, sticky="ew")

        # Button frame
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.grid(row=1, column=1, padx=(0, 10), pady=2)

        # Refresh button
        self.refresh_btn = ctk.CTkButton(
            self.btn_frame,
            text="更新",
            width=50,
            command=self._on_refresh_clicked,
        )
        self.refresh_btn.pack(side="left", padx=(0, 5))

        # Browse button
        self.browse_btn = ctk.CTkButton(
            self.btn_frame,
            text="開く...",
            width=60,
            command=self._browse_model,
        )
        self.browse_btn.pack(side="left")

        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="状態: 未選択",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.status_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 5))

        # Configure grid
        self.grid_columnconfigure(0, weight=1)

    def scan_directory(self, dir_path: str) -> None:
        """Scan directory for .pth files and update dropdown.

        Args:
            dir_path: Directory path to scan for RVC models
        """
        self._models_dir = dir_path
        root = Path(dir_path)
        if not root.is_dir():
            logger.warning(f"Models directory not found: {dir_path}")
            return

        self._model_map.clear()
        for pth in sorted(root.glob("**/*.pth")):
            # Use parent folder name if in subfolder, else use stem
            if pth.parent == root:
                display_name = pth.stem
            else:
                display_name = pth.parent.name
            # Handle duplicate display names by appending stem
            if display_name in self._model_map:
                display_name = f"{display_name} ({pth.stem})"
            self._model_map[display_name] = str(pth)

        # Update dropdown values
        if self._model_map:
            names = list(self._model_map.keys())
            self.model_dropdown.configure(values=names)
            # If current model is in the list, select it
            if self._model_path:
                current_name = self._find_display_name(self._model_path)
                if current_name:
                    self.model_var.set(current_name)
            logger.info(f"Scanned {len(self._model_map)} models from {dir_path}")
        else:
            self.model_dropdown.configure(values=["モデルを選択..."])
            logger.info(f"No .pth files found in {dir_path}")

    def _find_display_name(self, path: str) -> Optional[str]:
        """Find display name for a given model path."""
        norm_path = str(Path(path))
        for name, p in self._model_map.items():
            if str(Path(p)) == norm_path:
                return name
        return None

    def _on_refresh_clicked(self) -> None:
        """Handle refresh button click."""
        if hasattr(self, "_models_dir") and self._models_dir:
            self.scan_directory(self._models_dir)

    def _browse_model(self) -> None:
        """Open file dialog to select a model."""
        filepath = filedialog.askopenfilename(
            title="RVCモデルを選択",
            filetypes=[
                ("PyTorchモデル", "*.pth"),
                ("すべてのファイル", "*.*"),
            ],
        )

        if filepath:
            self.set_model(filepath)

    def _on_dropdown_change(self, value: str) -> None:
        """Handle dropdown selection change."""
        if value == "モデルを選択...":
            return
        # Resolve path from model map
        path = self._model_map.get(value)
        if path:
            self.set_model(path)
        elif self._model_path:
            # Fallback: already loaded model selected again
            if self.on_model_selected:
                self.on_model_selected(self._model_path)

    def set_model(self, path: str) -> None:
        """
        Set the selected model.

        Args:
            path: Path to the model file
        """
        model_path = Path(path)
        if not model_path.exists():
            logger.error(f"Model file not found: {path}")
            return

        self._model_path = str(model_path)

        # Update dropdown display
        display_name = self._find_display_name(self._model_path)
        if display_name:
            self.model_var.set(display_name)
        else:
            # Model not in scanned directory - show stem
            model_name = model_path.stem
            self.model_var.set(model_name)
            # Add to dropdown values temporarily
            current_values = self.model_dropdown.cget("values")
            if model_name not in current_values:
                self.model_dropdown.configure(values=list(current_values) + [model_name])

        # Check for index file
        index_path = model_path.with_suffix(".index")
        if not index_path.exists():
            # Try looking in the same directory
            index_files = list(model_path.parent.glob("*.index"))
            self._has_index = len(index_files) > 0
        else:
            self._has_index = True

        # Update status (actual F0 detection happens after loading)
        self._update_status()

        # Notify callback
        if self.on_model_selected:
            self.on_model_selected(self._model_path)

    def _update_status(self) -> None:
        """Update the status label."""
        if self._model_path is None:
            self.status_label.configure(text="状態: 未選択", text_color="gray")
            return

        parts = []

        # Version info
        if self._version is not None:
            parts.append(f"RVC v{self._version}")

        # F0 info
        if self._has_f0 is not None:
            f0_text = "F0あり" if self._has_f0 else "F0なし"
            parts.append(f0_text)

        # Index info
        index_text = "Index: あり" if self._has_index else "Index: なし"
        parts.append(index_text)

        status_text = "状態: " + " | ".join(parts)
        self.status_label.configure(text=status_text, text_color="white")

    def set_model_info(self, has_f0: bool, version: int = 2) -> None:
        """
        Update model info after loading.

        Args:
            has_f0: Whether model supports F0
            version: RVC version (1 or 2)
        """
        self._has_f0 = has_f0
        self._version = version
        self._update_status()

    @property
    def model_path(self) -> Optional[str]:
        """Get the current model path."""
        return self._model_path
