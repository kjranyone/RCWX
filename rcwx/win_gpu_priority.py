"""WDDM GPU scheduling priority boost for Windows.

DWM composition shares the GPU with inference; window drag/resize storms
preempt the compute queue for milliseconds at a time, which blows the 20ms
deadline budget in aggressive mode.  Raising the process's WDDM scheduling
priority class makes the scheduler favor our submissions over composition.

HIGH and REALTIME require administrator privileges; ABOVE_NORMAL does not,
so we walk down until one sticks.  Set RCWX_GPU_PRIORITY=0 to disable.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# D3DKMT_SCHEDULINGPRIORITYCLASS
_IDLE = 0
_BELOW_NORMAL = 1
_NORMAL = 2
_ABOVE_NORMAL = 3
_HIGH = 4
_REALTIME = 5

_PRIORITY_NAMES = {
    _ABOVE_NORMAL: "ABOVE_NORMAL",
    _HIGH: "HIGH",
    _REALTIME: "REALTIME",
}

_applied: bool | None = None


def boost_gpu_scheduling_priority() -> bool:
    """Raise this process's WDDM scheduling priority class (idempotent).

    Returns True if any elevated class was applied.
    """
    global _applied
    if _applied is not None:
        return _applied
    _applied = False

    if sys.platform != "win32":
        return False
    if os.environ.get("RCWX_GPU_PRIORITY", "1") == "0":
        logger.info("[GPU-PRIO] Disabled via RCWX_GPU_PRIORITY=0")
        return False

    try:
        import ctypes

        gdi32 = ctypes.WinDLL("gdi32")
        func = gdi32.D3DKMTSetProcessSchedulingPriorityClass
        func.argtypes = [ctypes.c_void_p, ctypes.c_int]
        func.restype = ctypes.c_int  # NTSTATUS
        hproc = ctypes.c_void_p(-1)  # GetCurrentProcess() pseudo-handle

        # HIGH needs admin; ABOVE_NORMAL is available to normal users.
        for cls in (_HIGH, _ABOVE_NORMAL):
            status = func(hproc, cls)
            if status == 0:
                logger.info(
                    "[GPU-PRIO] WDDM scheduling priority set to %s",
                    _PRIORITY_NAMES[cls],
                )
                _applied = True
                return True
        logger.info(
            "[GPU-PRIO] Could not raise WDDM scheduling priority "
            "(last NTSTATUS=0x%08X)",
            status & 0xFFFFFFFF,
        )
    except Exception as e:
        logger.info("[GPU-PRIO] Unavailable: %s", e)
    return False
