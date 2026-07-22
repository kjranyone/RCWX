"""Tests for device and dtype selection."""

import torch

from rcwx.device import get_dtype


def test_get_dtype_accepts_torch_dtype() -> None:
    assert get_dtype("xpu", torch.float32) is torch.float32
    assert get_dtype("xpu", torch.bfloat16) is torch.bfloat16


def test_get_dtype_keeps_cpu_float32_policy() -> None:
    assert get_dtype("cpu", torch.float16) is torch.float32
