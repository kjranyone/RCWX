"""Device selection for XPU/CUDA/CPU."""

from __future__ import annotations

from typing import Literal

import torch

DeviceType = Literal["xpu", "cuda", "cpu"]


def get_device(preferred: str = "auto") -> DeviceType:
    """
    Return the best available device.

    Args:
        preferred: Device preference ("auto", "xpu", "cuda", "cpu")

    Returns:
        Device type string
    """
    if preferred != "auto":
        if preferred == "xpu" and torch.xpu.is_available():
            return "xpu"
        elif preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif preferred == "cpu":
            return "cpu"

    if torch.xpu.is_available():
        return "xpu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype(
    device: DeviceType,
    preferred: str | torch.dtype = "float16",
) -> torch.dtype:
    """
    Return the optimal dtype for the device.

    Args:
        device: Device type
        preferred: Preferred dtype string or torch dtype

    Returns:
        torch.dtype
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if isinstance(preferred, torch.dtype):
        preferred_dtype = preferred
    else:
        preferred_dtype = dtype_map.get(preferred, torch.float16)

    if device == "cpu":
        return torch.float32

    return preferred_dtype


def get_device_name(device: DeviceType) -> str:
    """
    Return a human-readable device name.

    Args:
        device: Device type

    Returns:
        Device name string
    """
    if device == "xpu":
        try:
            return torch.xpu.get_device_name(0)
        except Exception:
            return "Intel XPU"
    elif device == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA GPU"
    return "CPU"


def list_devices() -> list[dict]:
    """
    List all available devices.

    Returns:
        List of device info dictionaries
    """
    devices = []

    devices.append({
        "type": "cpu",
        "index": 0,
        "name": "CPU",
        "available": True,
    })

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append({
                "type": "cuda",
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "available": True,
            })

    if torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            devices.append({
                "type": "xpu",
                "index": i,
                "name": torch.xpu.get_device_name(i),
                "available": True,
            })

    return devices
