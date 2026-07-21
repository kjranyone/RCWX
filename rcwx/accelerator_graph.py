"""Static-shape accelerator graph capture with eager fallback."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

ENV_NAME = "RCWX_ACCELERATOR_GRAPH"
MAX_CACHE_ENV = "RCWX_ACCELERATOR_GRAPH_MAX_CACHE"

_probe_lock = threading.Lock()
_probe_results: dict[str, bool] = {}


def _device_type(device: torch.device | str) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0].lower()


def _device_module(device: torch.device | str) -> Any:
    return getattr(torch, _device_type(device))


def _graph_requested(device: torch.device | str) -> bool:
    explicit = os.environ.get(ENV_NAME)
    if explicit == "0":
        return False
    device_type = _device_type(device)
    if explicit == "1":
        return device_type in {"xpu", "cuda"}
    # RCWX enables graphs automatically only on its primary XPU backend.
    return device_type == "xpu"


def detect_accelerator_graph_support(device: torch.device | str) -> bool:
    """Run a real capture/replay probe instead of trusting API presence."""
    if not _graph_requested(device):
        return False
    if not hasattr(torch, "accelerator") or not hasattr(torch.accelerator, "Graph"):
        return False

    module = _device_module(device)
    if not module.is_available():
        return False

    try:
        current = module.current_stream(device)
        capture_stream = module.Stream(device=device)
        capture_stream.wait_stream(current)
        probe = torch.arange(32, device=device, dtype=torch.float32)

        with module.stream(capture_stream), torch.no_grad():
            for _ in range(3):
                _ = probe.square() + 1
        module.synchronize(device)

        graph = torch.accelerator.Graph()
        with module.stream(capture_stream), torch.no_grad(), graph:
            captured = probe.square() + 1

        with module.stream(capture_stream), torch.no_grad():
            probe.copy_(torch.arange(32, device=device, dtype=torch.float32))
            graph.replay()
            result = captured.clone()
        module.synchronize(device)

        expected = torch.arange(32, dtype=torch.float32).square() + 1
        valid = torch.equal(result.cpu(), expected)
        graph.reset()
        return bool(valid)
    except Exception:
        logger.exception("Accelerator Graph probe failed on %s", device)
        return False


def accelerator_graph_enabled(device: torch.device | str) -> bool:
    if not _graph_requested(device):
        return False
    key = str(device)
    with _probe_lock:
        if key not in _probe_results:
            _probe_results[key] = detect_accelerator_graph_support(device)
            logger.info(
                "Accelerator Graph %s on %s",
                "enabled" if _probe_results[key] else "unavailable",
                device,
            )
        return _probe_results[key]


def _clone_output(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_output(item) for item in value)
    if isinstance(value, list):
        return [_clone_output(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_output(item) for key, item in value.items()}
    return value


def _tensor_signature(
    tensor: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> tuple[Any, ...]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(dtype or tensor.dtype),
        str(tensor.device),
        bool(tensor.requires_grad),
    )


def _call_signature(
    namespace: str,
    inputs: tuple[torch.Tensor, ...],
    dtype_overrides: tuple[torch.dtype, ...] | None = None,
) -> tuple[Any, ...]:
    if dtype_overrides is None:
        dtype_overrides = tuple(value.dtype for value in inputs)
    return (namespace,) + tuple(
        _tensor_signature(value, dtype)
        for value, dtype in zip(inputs, dtype_overrides)
    )


class _CapturedCall:
    def __init__(
        self,
        function: Callable[..., Any],
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        started = time.perf_counter()
        self.lock = threading.RLock()
        self.inputs = tuple(torch.empty_like(value) for value in inputs)
        for static, value in zip(self.inputs, inputs):
            static.copy_(value)

        device = self.inputs[0].device
        self.module = _device_module(device)
        current = self.module.current_stream(device)
        self.stream = self.module.Stream(device=device)
        self.stream.wait_stream(current)

        with self.module.stream(self.stream), torch.no_grad():
            for _ in range(3):
                warmup_output = function(*self.inputs)
        self.module.synchronize(device)

        self.graph = torch.accelerator.Graph()
        with self.module.stream(self.stream), torch.no_grad(), self.graph:
            self.output = function(*self.inputs)
        self.module.synchronize(device)
        self.capture_ms = (time.perf_counter() - started) * 1000.0
        del warmup_output

    def replay(self, inputs: tuple[torch.Tensor, ...]) -> Any:
        with self.lock:
            current = self.module.current_stream(self.inputs[0].device)
            self.stream.wait_stream(current)
            with self.module.stream(self.stream), torch.no_grad():
                for static, value in zip(self.inputs, inputs):
                    static.copy_(value, non_blocking=True)
                self.graph.replay()
                output = _clone_output(self.output)
                done = self.module.Event()
                done.record(self.stream)
            current.wait_event(done)
            return output


class _GraphCache:
    def __init__(self) -> None:
        self.entries: OrderedDict[tuple[Any, ...], _CapturedCall] = OrderedDict()
        self.failures: set[tuple[Any, ...]] = set()
        self.lock = threading.RLock()
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0
        self.eviction_count = 0
        self.capture_ms = 0.0

    def capture_pending(
        self,
        namespace: str,
        inputs: tuple[torch.Tensor, ...],
        dtype_overrides: tuple[torch.dtype, ...] | None = None,
    ) -> bool:
        signature = _call_signature(namespace, inputs, dtype_overrides)
        with self.lock:
            return signature not in self.entries and signature not in self.failures

    def run(
        self,
        namespace: str,
        function: Callable[..., Any],
        inputs: tuple[torch.Tensor, ...],
    ) -> Any:
        signature = _call_signature(namespace, inputs)
        with self.lock:
            if signature in self.failures:
                self.fallback_count += 1
                return function(*inputs)

            entry = self.entries.get(signature)
            if entry is None:
                try:
                    entry = _CapturedCall(function, inputs)
                    self.entries[signature] = entry
                    self.capture_count += 1
                    self.capture_ms += entry.capture_ms
                    max_entries = max(1, int(os.environ.get(MAX_CACHE_ENV, "8")))
                    while len(self.entries) > max_entries:
                        _, evicted = self.entries.popitem(last=False)
                        evicted.graph.reset()
                        self.eviction_count += 1
                    logger.info(
                        "Captured Accelerator Graph %s in %.1f ms",
                        namespace,
                        entry.capture_ms,
                    )
                except Exception:
                    self.failures.add(signature)
                    self.fallback_count += 1
                    logger.exception(
                        "Accelerator Graph capture failed for %s; using eager",
                        namespace,
                    )
                    return function(*inputs)
            else:
                self.entries.move_to_end(signature)

        output = entry.replay(inputs)
        with self.lock:
            self.replay_count += 1
        return output

    def clear(self) -> None:
        with self.lock:
            for entry in self.entries.values():
                entry.graph.reset()
            self.entries.clear()
            self.failures.clear()


def _get_cache(owner: object) -> _GraphCache:
    cache = getattr(owner, "_rcwx_accelerator_graph_cache", None)
    if cache is None:
        cache = _GraphCache()
        setattr(owner, "_rcwx_accelerator_graph_cache", cache)
    return cache


def accelerator_graph_capture_pending(
    owner: object,
    namespace: str,
    *inputs: torch.Tensor,
    dtype_overrides: tuple[torch.dtype, ...] | None = None,
) -> bool:
    if not inputs or not accelerator_graph_enabled(inputs[0].device):
        return False
    return _get_cache(owner).capture_pending(
        namespace,
        tuple(inputs),
        dtype_overrides,
    )


def run_accelerator_graph(
    owner: object,
    namespace: str,
    function: Callable[..., Any],
    *inputs: torch.Tensor,
) -> Any:
    if not inputs or not accelerator_graph_enabled(inputs[0].device):
        return function(*inputs)
    return _get_cache(owner).run(namespace, function, tuple(inputs))


def clear_accelerator_graph_cache(owner: object) -> None:
    cache = getattr(owner, "_rcwx_accelerator_graph_cache", None)
    if cache is not None:
        cache.clear()
        delattr(owner, "_rcwx_accelerator_graph_cache")


def get_accelerator_graph_stats(owner: object) -> dict[str, float | int]:
    cache = getattr(owner, "_rcwx_accelerator_graph_cache", None)
    if cache is None:
        return {
            "entries": 0,
            "failures": 0,
            "captures": 0,
            "replays": 0,
            "fallbacks": 0,
            "evictions": 0,
            "capture_ms": 0.0,
        }
    with cache.lock:
        return {
            "entries": len(cache.entries),
            "failures": len(cache.failures),
            "captures": cache.capture_count,
            "replays": cache.replay_count,
            "fallbacks": cache.fallback_count,
            "evictions": cache.eviction_count,
            "capture_ms": cache.capture_ms,
        }
