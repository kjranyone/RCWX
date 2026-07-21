"""Tests for real-time Synthesizer Accelerator Graph integration."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

import rcwx.models.synthesizer as synthesizer_module
from rcwx.models.synthesizer import SynthesizerLoader


class _FakeF0Synthesizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sine_gen = SimpleNamespace(
            phase_mode="legacy",
            fixed_harmonics=False,
            uv_ramp_ms=0.0,
        )
        self.dec = SimpleNamespace(
            m_source=SimpleNamespace(l_sin_gen=sine_gen)
        )

    def infer(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        speaker: torch.Tensor,
        *,
        skip_head: int,
        return_length: int,
        return_length2: int,
        noise_scale: float,
    ) -> torch.Tensor:
        del lengths, pitch, speaker, return_length2
        end = skip_head + return_length
        base = features[:, skip_head:end, :1].transpose(1, 2)
        base = base + pitchf[:, skip_head:end].unsqueeze(1) * 0.001
        return base + torch.randn_like(base) * noise_scale


class _FakeNoF0Synthesizer(nn.Module):
    def infer(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        speaker: torch.Tensor,
        *,
        skip_head: int,
        return_length: int,
        return_length2: int,
        noise_scale: float,
    ) -> torch.Tensor:
        del lengths, speaker, return_length2, noise_scale
        end = skip_head + return_length
        return features[:, skip_head:end, :1].transpose(1, 2)


def _loader(model: nn.Module, *, has_f0: bool = True) -> SynthesizerLoader:
    loader = SynthesizerLoader("unused.pth", device="cpu", dtype=torch.float32)
    loader.model = model
    loader.has_f0 = has_f0
    loader.version = 2
    return loader


def _inputs(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    features = torch.arange(16, dtype=torch.float32, device=device).reshape(1, 8, 2)
    lengths = torch.tensor([8], dtype=torch.long, device=device)
    pitch = torch.full((1, 8), 100, dtype=torch.long, device=device)
    pitchf = torch.full((1, 8), 220.0, dtype=torch.float32, device=device)
    return features, lengths, pitch, pitchf


def test_realtime_graph_namespace_tracks_python_controls(monkeypatch) -> None:
    loader = _loader(_FakeF0Synthesizer())
    features, lengths, pitch, pitchf = _inputs()
    namespaces = []

    def run_graph(owner, namespace, function, *inputs):
        del owner
        namespaces.append(namespace)
        return function(*inputs)

    monkeypatch.setattr(synthesizer_module, "run_accelerator_graph", run_graph)

    kwargs = dict(
        pitch=pitch,
        pitchf=pitchf,
        skip_head=2,
        return_length=4,
        return_length2=4,
        use_accelerator_graph=True,
    )
    first = loader.infer(features, lengths, noise_scale=0.4, **kwargs)
    loader.model.dec.m_source.l_sin_gen.fixed_harmonics = True
    loader.model.dec.m_source.l_sin_gen.uv_ramp_ms = 5.0
    second = loader.infer(features, lengths, noise_scale=0.7, **kwargs)

    assert first.shape == second.shape == (1, 4)
    assert len(namespaces) == 2
    assert namespaces[0] != namespaces[1]
    assert "skip-2-return-4-return2-4" in namespaces[0]
    assert "fixed-1" in namespaces[1]


def test_compile_mode_bypasses_accelerator_graph(monkeypatch) -> None:
    loader = _loader(_FakeNoF0Synthesizer(), has_f0=False)
    loader.use_compile = True
    features, lengths, _, _ = _inputs()

    def unexpected_graph(*args, **kwargs):
        raise AssertionError("torch.compile mode must not use Accelerator Graph")

    monkeypatch.setattr(
        synthesizer_module,
        "run_accelerator_graph",
        unexpected_graph,
    )
    output = loader.infer(
        features,
        lengths,
        skip_head=2,
        return_length=4,
        return_length2=4,
        use_accelerator_graph=True,
    )

    assert output.shape == (1, 4)


def test_xpu_synthesizer_graph_replays_with_fresh_rng() -> None:
    if not torch.xpu.is_available() or not hasattr(torch.accelerator, "Graph"):
        return

    loader = _loader(_FakeF0Synthesizer())
    loader.device = "xpu"
    features, lengths, pitch, pitchf = _inputs("xpu")
    kwargs = dict(
        pitch=pitch,
        pitchf=pitchf,
        noise_scale=0.5,
        skip_head=2,
        return_length=4,
        return_length2=4,
        use_accelerator_graph=True,
    )

    first = loader.infer(features, lengths, **kwargs)
    first_snapshot = first.clone()
    second = loader.infer(features + 1, lengths, **kwargs)
    torch.xpu.synchronize()

    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, first_snapshot)
    assert not torch.equal(first, second)
    stats = loader.graph_stats()
    assert stats["entries"] == 1
    assert stats["captures"] == 1
    assert stats["replays"] == 2
    assert stats["fallbacks"] == 0

    loader.model.dec.m_source.l_sin_gen.fixed_harmonics = True
    _ = loader.infer(features, lengths, **kwargs)
    torch.xpu.synchronize()
    assert loader.graph_stats()["entries"] == 2

    loader.clear_graph_cache()
    assert loader.graph_stats()["entries"] == 0
