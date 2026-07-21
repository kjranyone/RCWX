"""Tests for device graph capture and eager fallback."""

from __future__ import annotations

import torch

from rcwx.accelerator_graph import (
    accelerator_graph_capture_pending,
    clear_accelerator_graph_cache,
    get_accelerator_graph_stats,
    run_accelerator_graph,
)


class _Owner:
    pass


def test_cpu_uses_eager_without_cache() -> None:
    owner = _Owner()
    value = torch.arange(8, dtype=torch.float32)

    output = run_accelerator_graph(owner, "cpu-square", lambda x: x.square(), value)

    assert torch.equal(output, value.square())
    assert get_accelerator_graph_stats(owner)["entries"] == 0


def test_xpu_graph_capture_and_replay() -> None:
    if not torch.xpu.is_available() or not hasattr(torch.accelerator, "Graph"):
        return

    owner = _Owner()
    value = torch.arange(8, device="xpu", dtype=torch.float32)

    def function(x: torch.Tensor) -> torch.Tensor:
        return x.square() + 1

    assert accelerator_graph_capture_pending(owner, "xpu-square", value)
    first = run_accelerator_graph(owner, "xpu-square", function, value)
    second_input = value + 1
    second = run_accelerator_graph(owner, "xpu-square", function, second_input)
    torch.xpu.synchronize()

    assert torch.equal(first.cpu(), value.cpu().square() + 1)
    assert torch.equal(second.cpu(), second_input.cpu().square() + 1)
    assert not accelerator_graph_capture_pending(owner, "xpu-square", value)
    stats = get_accelerator_graph_stats(owner)
    assert stats["entries"] == 1
    assert stats["failures"] == 0
    assert stats["captures"] == 1
    assert stats["replays"] == 2
    assert stats["fallbacks"] == 0
    assert stats["evictions"] == 0
    assert stats["capture_ms"] > 0

    clear_accelerator_graph_cache(owner)
    assert get_accelerator_graph_stats(owner)["entries"] == 0
