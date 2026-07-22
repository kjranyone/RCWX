"""Regression: switching models must unload the previous pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rcwx.gui.model_loader import ModelLoader


class _ImmediateThread:
    """Run target synchronously so unit tests need no sleeps/joins."""

    def __init__(self, target=None, daemon=None, **_kwargs):
        self._target = target

    def start(self) -> None:
        if self._target is not None:
            self._target()


class _FakeApp:
    def __init__(self) -> None:
        self._loading = False
        self._is_running = False
        self.pipeline = None
        self.device_var = SimpleNamespace(get=lambda: "cpu")
        self.dtype_var = SimpleNamespace(get=lambda: "float32")
        self.compile_var = SimpleNamespace(get=lambda: False)
        self.models_dir_entry = SimpleNamespace(get=lambda: ".")
        self.status_bar = MagicMock()
        self.start_btn = MagicMock()
        self.model_selector = MagicMock()
        self.pitch_control = MagicMock()
        self.index_status = MagicMock()
        self.realtime_controller = MagicMock()

    def _get_index_rate(self) -> float:
        return 0.0

    def after(self, _ms, fn):
        fn()


def test_load_model_unloads_previous_pipeline_after_success():
    app = _FakeApp()
    old = MagicMock(name="old_pipeline")
    new = MagicMock(name="new_pipeline")
    new.has_f0 = True
    new.synthesizer = SimpleNamespace(version=2)
    new.faiss_index = None
    new.device = "cpu"
    app.pipeline = old

    loader = ModelLoader(app)

    with (
        patch("rcwx.gui.model_loader.RVCPipeline", return_value=new) as ctor,
        patch("rcwx.gui.model_loader.threading.Thread", _ImmediateThread),
        patch("rcwx.gui.model_loader.get_device_name", return_value="CPU"),
    ):
        loader.load_model("model.pth")

    ctor.assert_called_once()
    new.load.assert_called_once()
    assert app.pipeline is new
    old.unload.assert_called_once()
    assert app._loading is False


def test_load_model_keeps_previous_pipeline_on_failure():
    app = _FakeApp()
    old = MagicMock(name="old_pipeline")
    app.pipeline = old

    broken = MagicMock(name="broken_pipeline")
    broken.load.side_effect = RuntimeError("boom")

    loader = ModelLoader(app)

    with (
        patch("rcwx.gui.model_loader.RVCPipeline", return_value=broken),
        patch("rcwx.gui.model_loader.threading.Thread", _ImmediateThread),
        patch.object(loader, "_on_model_load_error") as on_error,
    ):
        loader.load_model("model.pth")

    assert app.pipeline is old
    old.unload.assert_not_called()
    on_error.assert_called_once()
