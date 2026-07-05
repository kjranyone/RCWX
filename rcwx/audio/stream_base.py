"""Base class for audio streams with robust fallback."""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class AudioStreamBase(ABC):
    """
    Base class for audio input/output streams with robust fallback.

    Implements fallback across multiple audio APIs (WASAPI, DirectSound, MME)
    and configurations to ensure reliable operation on various hardware.
    """

    # Subclasses must set these
    STREAM_TYPE: str = ""  # "input" or "output"
    STREAM_CLASS: type = None  # sd.InputStream or sd.OutputStream

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 48000,
        channels: int = 1,
        blocksize: int = 1024,
        asio_buffer_size: int = 0,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        # Requested ASIO buffer size in frames (0 = driver control-panel
        # preferred).  Only consulted when *device* is on the ASIO host API;
        # honored via _try_open_asio() so a simplex ASIO stream (e.g. ASIO
        # output paired with a WASAPI mic) still gets a controlled buffer
        # instead of the driver default.
        self.asio_buffer_size = asio_buffer_size
        self._stream: Optional[sd.InputStream | sd.OutputStream] = None
        self._actual_sample_rate: Optional[int] = None

    @property
    def actual_sample_rate(self) -> int:
        """Get the actual sample rate used by the stream."""
        return self._actual_sample_rate or self.sample_rate

    @abstractmethod
    def _audio_callback(
        self,
        data: NDArray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal sounddevice callback - must be implemented by subclass."""
        pass

    def _get_api_preferences(self) -> list[tuple[str, Optional[int], Optional[object]]]:
        """
        Get list of audio API preferences with settings.

        Returns:
            List of (api_name, hostapi_index, extra_settings) tuples
        """
        api_preferences: list[tuple[str, Optional[int], Optional[object]]] = []

        if sys.platform == "win32":
            wasapi_hostapi = None
            ds_hostapi = None
            mme_hostapi = None

            for i, hostapi in enumerate(sd.query_hostapis()):
                name = hostapi["name"]
                if "WASAPI" in name:
                    wasapi_hostapi = i
                elif "DirectSound" in name:
                    ds_hostapi = i
                elif "MME" in name:
                    mme_hostapi = i

            # WASAPI: try both with and without WasapiSettings
            if wasapi_hostapi is not None:
                try:
                    wasapi_shared = sd.WasapiSettings(exclusive=False)
                    api_preferences.append(("WASAPI-shared", wasapi_hostapi, wasapi_shared))
                except Exception:
                    pass
                api_preferences.append(("WASAPI", wasapi_hostapi, None))

            # NOTE: ASIO is intentionally excluded from the fallback list.
            # ASIO drivers are exclusive full-duplex — opening a single
            # InputStream or OutputStream claims the driver, blocking the
            # other direction.  ASIO is used only via AsioDuplexStream.

            if ds_hostapi is not None:
                api_preferences.append(("DirectSound", ds_hostapi, None))

            if mme_hostapi is not None:
                api_preferences.append(("MME", mme_hostapi, None))

        api_preferences.append(("default", None, None))
        return api_preferences

    def _find_device_on_api(self, hostapi_index: int, original_device: int) -> Optional[int]:
        """Find equivalent device on a different API by matching device name."""
        try:
            original_info = sd.query_devices(original_device)
            original_name = original_info["name"]
            channel_key = f"max_{self.STREAM_TYPE}_channels"

            for i, dev in enumerate(sd.query_devices()):
                if dev[channel_key] > 0 and dev["hostapi"] == hostapi_index and dev["name"] == original_name:
                    return i
        except Exception:
            pass
        return None

    def _get_device_sample_rates(self, device: Optional[int]) -> list[int]:
        """Get sample rates to try for a device."""
        rates = []
        try:
            if device is not None:
                dev_info = sd.query_devices(device)
            else:
                dev_info = sd.query_devices(kind=self.STREAM_TYPE)
            default_rate = int(dev_info["default_samplerate"])
            rates.append(default_rate)
        except Exception:
            pass

        if self.sample_rate not in rates:
            rates.append(self.sample_rate)

        for rate in [48000, 44100, 16000, 22050, 32000, 96000]:
            if rate not in rates:
                rates.append(rate)

        return rates

    def _get_blocksize_options(self, sample_rate: int) -> list[int]:
        """Get blocksize options to try."""
        rate_ratio = sample_rate / self.sample_rate if self.sample_rate > 0 else 1.0
        scaled_blocksize = int(self.blocksize * rate_ratio)

        blocksizes = [scaled_blocksize, self.blocksize, 2048, 4096, 1024, 512, 8192]
        return list(dict.fromkeys(blocksizes))  # Remove duplicates, preserve order

    def _try_open_stream(
        self,
        device: Optional[int],
        sample_rate: int,
        blocksize: int,
        extra_settings: Optional[object],
    ) -> None:
        """Try to open a stream with given parameters."""
        self._stream = self.STREAM_CLASS(
            device=device,
            samplerate=sample_rate,
            channels=self.channels,
            blocksize=blocksize,
            dtype=np.float32,
            callback=self._audio_callback,
            extra_settings=extra_settings,
        )
        self._stream.start()

    def _get_device_name(self, device: Optional[int]) -> str:
        """Get human-readable device name."""
        if device is not None:
            try:
                return sd.query_devices(device)["name"]
            except Exception:
                return f"device={device}"
        else:
            try:
                default = sd.query_devices(kind=self.STREAM_TYPE)
                return f"システムデフォルト ({default['name']})"
            except Exception:
                return "システムデフォルト"

    def _get_error_message(self, error_msg: str, api_names: list[str]) -> str:
        """Get localized error message."""
        stream_type_jp = "入力" if self.STREAM_TYPE == "input" else "出力"
        device_type_jp = "マイク" if self.STREAM_TYPE == "input" else "スピーカー"
        return (
            f"オーディオ{stream_type_jp}デバイスを開けませんでした。\n"
            f"詳細: {error_msg}\n\n"
            f"試行したAPI: {', '.join(api_names)}\n\n"
            f"解決策:\n"
            f"1. 別のオーディオデバイスを選択してください\n"
            f"2. 他のアプリが{device_type_jp}を排他モードで使用していないか確認してください\n"
            f"3. オーディオドライバーを更新してください\n"
            f"4. PCを再起動してください"
        )

    def _try_open_asio(self) -> bool:
        """Open *self.device* as a dedicated ASIO stream with a controlled buffer.

        The generic fallback in start() intentionally excludes ASIO from its
        API list, but its final ``"default"`` entry still opens an ASIO device
        index with a fixed blocksize — leaving the ASIO hardware buffer at the
        driver's control-panel value (often 2048).  This path instead opens
        with ``blocksize=0`` + ``latency = size / samplerate`` so PortAudio
        builds the ASIO buffers at the requested (or preferred) size, mirroring
        AsioDuplexStream.  Only valid when the device is a *different* device
        from the other direction's stream (else use AsioDuplexStream) — that is
        the case for a mixed config (e.g. ASIO output + WASAPI mic).

        Returns True on success; False (with no stream opened) otherwise, so
        the caller falls through to the generic fallback unchanged.
        """
        if self.device is None or not is_device_on_asio(self.device, self.STREAM_TYPE):
            return False

        target_size = resolve_asio_buffer_size(
            self.device, self.asio_buffer_size, self.STREAM_TYPE
        )

        # Native (control-panel) rate first; ASIO drivers run at a fixed rate.
        rates: list[int] = []
        native = query_asio_native_sample_rate(self.device, self.STREAM_TYPE)
        if native is not None:
            rates.append(native)
        for r in [self.sample_rate, 48000, 44100, 96000]:
            if r not in rates:
                rates.append(r)

        device_name = self._get_device_name(self.device)
        for sr in rates:
            # target_size==0 (query failed): fall back to a ~10ms hint rather
            # than 'high' (=driver max) which sd would otherwise pick.
            latency = (target_size / float(sr)) if target_size > 0 else 0.010
            try:
                self._stream = self.STREAM_CLASS(
                    device=self.device,
                    samplerate=sr,
                    channels=self.channels,
                    blocksize=0,
                    dtype=np.float32,
                    latency=latency,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._actual_sample_rate = int(self._stream.samplerate)
                logger.info(
                    "%s stream started (ASIO): %s, sr=%dHz, asio_buffer=%s, "
                    "latency=%s",
                    self.STREAM_TYPE.capitalize(), device_name,
                    self._actual_sample_rate,
                    target_size if target_size > 0 else "auto",
                    self._stream.latency,
                )
                return True
            except Exception as e:
                logger.debug("ASIO %s open failed (sr=%d): %s", self.STREAM_TYPE, sr, e)
                self._stream = None
        return False

    def start(self) -> None:
        """Start stream with robust fallback."""
        if self._stream is not None:
            return

        # ASIO devices: open with a controlled buffer first (blocksize=0 +
        # explicit latency).  If it fails, fall through to the generic
        # WASAPI/DirectSound/MME/default chain below (behavior unchanged).
        if self._try_open_asio():
            return

        api_preferences = self._get_api_preferences()
        last_error: Optional[Exception] = None
        attempts: list[str] = []

        for api_name, hostapi_index, extra_settings in api_preferences:
            device_to_use = self.device

            if hostapi_index is not None and self.device is not None:
                try:
                    device_info = sd.query_devices(self.device)
                    if device_info["hostapi"] != hostapi_index:
                        mapped_device = self._find_device_on_api(hostapi_index, self.device)
                        if mapped_device is not None:
                            device_to_use = mapped_device
                        else:
                            logger.debug(f"{api_name}: No equivalent device for '{device_info['name']}'")
                            continue
                except Exception as e:
                    logger.debug(f"{api_name}: Failed to query device: {e}")
                    continue

            device_name = self._get_device_name(device_to_use)
            sample_rates = self._get_device_sample_rates(device_to_use)
            logger.info(f"Trying {api_name}: {device_name}")

            for sr in sample_rates:
                blocksizes = self._get_blocksize_options(sr)

                for bs in blocksizes:
                    try:
                        self._try_open_stream(device_to_use, sr, bs, extra_settings)
                        self._actual_sample_rate = int(self._stream.samplerate)
                        self.blocksize = bs
                        logger.info(
                            f"{self.STREAM_TYPE.capitalize()} stream started ({api_name}): "
                            f"{device_name}, sr={self._actual_sample_rate}Hz, blocksize={bs}"
                        )
                        return
                    except Exception as e:
                        last_error = e
                        attempts.append(f"{api_name}/sr={sr}/bs={bs}")
                        logger.debug(f"Failed: {api_name}/sr={sr}/bs={bs}: {e}")
                        self._stream = None

            if last_error:
                logger.warning(f"{api_name} failed: {last_error}")

        logger.error(f"All audio configurations failed. Tried {len(attempts)} configurations. Last error: {last_error}")
        error_msg = str(last_error) if last_error else "Unknown error"
        raise AudioStreamError(self._get_error_message(error_msg, [api for api, _, _ in api_preferences]))

    def stop(self) -> None:
        """Stop stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error stopping {self.STREAM_TYPE} stream: {e}")
            finally:
                self._stream = None

    @property
    def is_active(self) -> bool:
        """Check if stream is active."""
        return self._stream is not None and self._stream.active


class AudioStreamError(Exception):
    """Exception raised when audio stream cannot be opened."""

    pass


def parse_output_channel_pair(selection: str) -> Optional[tuple[int, int]]:
    """Parse a canonical output channel selection ("a,b") into an index pair.

    Returns None for "auto" or any unparseable value.  Callers that receive
    None for a non-"auto" value should treat it as auto AND emit a warning —
    silently opening/broadcasting to all channels is what previously routed
    voice-changer output into ASIO LOOPBACK channels (feedback loop).
    """
    if selection == "auto":
        return None
    try:
        parts = str(selection).split(",")
        if len(parts) == 2:
            a, b = int(parts[0]), int(parts[1])
            if a >= 0 and b >= 0:
                return (a, b)
    except (ValueError, IndexError):
        pass
    return None


def reinit_portaudio() -> bool:
    """Rebuild PortAudio's internal device cache.

    PortAudio enumerates devices only at ``Pa_Initialize`` time, so devices
    plugged in after process start are invisible to ``sd.query_devices()``
    until we terminate and re-initialize. Must NOT be called while any
    sounddevice stream is active — doing so crashes PortAudio.
    Returns True on success.
    """
    try:
        sd._terminate()
        sd._initialize()
        return True
    except Exception as e:
        logger.warning("PortAudio reinit failed: %s", e)
        return False


def list_devices(stream_type: str, wasapi_only: bool = False) -> list[dict]:
    """
    List available audio devices.

    Args:
        stream_type: "input" or "output"
        wasapi_only: If True, only show WASAPI devices (default: False)

    Returns:
        List of device info dictionaries with hostapi_name
    """
    devices = []
    channel_key = f"max_{stream_type}_channels"

    # Build hostapi index -> name mapping
    hostapi_names = {}
    wasapi_hostapi = None
    for i, hostapi in enumerate(sd.query_hostapis()):
        hostapi_names[i] = hostapi["name"]
        if "WASAPI" in hostapi["name"]:
            wasapi_hostapi = i

    for i, dev in enumerate(sd.query_devices()):
        if dev[channel_key] > 0:
            if wasapi_only and wasapi_hostapi is not None:
                if dev["hostapi"] != wasapi_hostapi:
                    continue

            # Get hostapi name (WASAPI, ASIO, MME, etc.)
            hostapi_idx = dev["hostapi"]
            hostapi_name = hostapi_names.get(hostapi_idx, "Unknown")

            # Simplify hostapi name (e.g., "Windows WASAPI" -> "WASAPI")
            if "WASAPI" in hostapi_name:
                hostapi_name = "WASAPI"
            elif "ASIO" in hostapi_name:
                hostapi_name = "ASIO"
            elif "DirectSound" in hostapi_name:
                hostapi_name = "DirectSound"
            elif "MME" in hostapi_name:
                hostapi_name = "MME"

            devices.append({
                "index": i,
                "name": dev["name"],
                "channels": dev[channel_key],
                "sample_rate": dev["default_samplerate"],
                "hostapi_name": hostapi_name,
            })
    return devices


def get_default_device(stream_type: str) -> Optional[int]:
    """Get the default device index."""
    try:
        return sd.query_devices(kind=stream_type)["index"]
    except Exception:
        return None


def query_asio_channel_names(
    device_index: int,
    stream_type: str = "input",
) -> list[str]:
    """Return ASIO channel names for a device via PortAudio's ASIO extension.

    Uses ``PaAsio_GetInputChannelName`` / ``PaAsio_GetOutputChannelName``
    from the PortAudio DLL via ctypes.  Returns an empty list if the
    functions are unavailable or the device is not ASIO.

    Example return: ``["MIC/LINE/INST 1", "MIC/LINE/INST 2", "LOOPBACK Left", ...]``
    """
    import ctypes
    import re

    if not is_device_on_asio(device_index, stream_type):
        return []

    try:
        info = sd.query_devices(device_index)
        channel_key = f"max_{stream_type}_channels"
        n_channels = int(info[channel_key])
        if n_channels == 0:
            return []

        # Load the PortAudio DLL via ctypes to access ASIO-specific functions
        # that are not exposed through sounddevice's cffi declarations.
        m = re.search(r"'([^']+)'", repr(sd._lib))
        if not m:
            return []
        pa = ctypes.CDLL(m.group(1))

        func_name = (
            "PaAsio_GetInputChannelName"
            if stream_type == "input"
            else "PaAsio_GetOutputChannelName"
        )
        func = getattr(pa, func_name)
        func.restype = ctypes.c_int
        func.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]

        names: list[str] = []
        for ch in range(n_channels):
            name_ptr = ctypes.c_char_p()
            err = func(device_index, ch, ctypes.byref(name_ptr))
            if err == 0 and name_ptr.value:
                names.append(name_ptr.value.decode())
            else:
                names.append(f"Ch {ch + 1}")
        return names
    except Exception as e:
        logger.debug("Failed to query ASIO channel names: %s", e)
        return []


def get_asio_hostapi_index() -> Optional[int]:
    """Return the hostapi index for ASIO, or None if unavailable."""
    try:
        for i, hostapi in enumerate(sd.query_hostapis()):
            if "ASIO" in hostapi["name"]:
                return i
    except Exception:
        pass
    return None


def is_device_on_asio(device_index: Optional[int], stream_type: str = "output") -> bool:
    """Check whether a device is on the ASIO host API.

    If *device_index* is None, the system default for *stream_type* is queried.
    """
    asio_idx = get_asio_hostapi_index()
    if asio_idx is None:
        return False
    try:
        if device_index is not None:
            info = sd.query_devices(device_index)
        else:
            info = sd.query_devices(kind=stream_type)
        return info["hostapi"] == asio_idx
    except Exception:
        return False


def query_asio_buffer_sizes(
    device_index: Optional[int],
    stream_type: str = "output",
) -> Optional[tuple[int, int, int, int]]:
    """Return (min, max, preferred, granularity) ASIO buffer sizes in frames.

    Uses ``PaAsio_GetAvailableBufferSizes`` from the PortAudio DLL via
    ctypes.  ``preferred`` is the buffer size configured in the driver's
    control panel — PortAudio's 'low'/'high' latency defaults map to
    min/max buffer size instead, so honoring the panel value requires
    this query.  Returns None if unavailable or the device is not ASIO.
    """
    import ctypes
    import re

    if device_index is None or not is_device_on_asio(device_index, stream_type):
        return None

    try:
        m = re.search(r"'([^']+)'", repr(sd._lib))
        if not m:
            return None
        pa = ctypes.CDLL(m.group(1))

        func = pa.PaAsio_GetAvailableBufferSizes
        func.restype = ctypes.c_int
        func.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
        ]

        min_size = ctypes.c_long()
        max_size = ctypes.c_long()
        preferred = ctypes.c_long()
        granularity = ctypes.c_long()
        err = func(
            device_index,
            ctypes.byref(min_size),
            ctypes.byref(max_size),
            ctypes.byref(preferred),
            ctypes.byref(granularity),
        )
        if err != 0:
            return None
        return (
            int(min_size.value),
            int(max_size.value),
            int(preferred.value),
            int(granularity.value),
        )
    except Exception as e:
        logger.debug("Failed to query ASIO buffer sizes: %s", e)
        return None


def snap_asio_buffer_size(
    requested: int, sizes: tuple[int, int, int, int]
) -> int:
    """Clamp a requested ASIO buffer size to the driver's constraints.

    ``sizes`` is (min, max, preferred, granularity) from
    ``query_asio_buffer_sizes``.  Granularity -1 means power-of-two steps
    (per the ASIO SDK); a positive value means arithmetic steps from min.
    """
    min_size, max_size, _preferred, granularity = sizes
    size = max(min_size, min(max_size, int(requested)))
    if granularity == -1:
        p = min_size if min_size > 0 else 1
        while p * 2 <= size:
            p *= 2
        if (size - p) > (p * 2 - size) and p * 2 <= max_size:
            p *= 2
        size = p
    elif granularity > 0:
        steps = round((size - min_size) / granularity)
        size = min_size + int(steps) * granularity
        size = max(min_size, min(max_size, size))
    return size


def resolve_asio_buffer_size(
    device_index: Optional[int],
    requested_buffer_size: int,
    stream_type: str = "output",
) -> int:
    """Resolve the ASIO host buffer size (frames) to request for a device.

    Intended for use with ``blocksize=0`` + ``latency = size / samplerate`` so
    PortAudio builds the ASIO buffers at exactly this size.  PortAudio's
    'low'/'high' latency classes map to the driver min/max instead, so an
    explicit size is required to honor either the user's choice or the
    control-panel preferred value.

    - ``requested_buffer_size > 0``: snapped to the driver's
      min/max/granularity constraints.
    - ``requested_buffer_size == 0``: the driver's preferred (control-panel)
      size.
    - Returns ``0`` when the size can't be determined (caller should fall back
      to a plain latency hint, e.g. ~10ms).

    Shared by AsioDuplexStream and AudioStreamBase._try_open_asio so both the
    duplex and simplex ASIO paths honor the same setting identically.
    """
    sizes = query_asio_buffer_sizes(device_index, stream_type)
    if sizes is not None:
        logger.info(
            "ASIO buffer sizes (%s): min=%d max=%d preferred=%d granularity=%d",
            stream_type, *sizes,
        )
    if requested_buffer_size and requested_buffer_size > 0:
        target = int(requested_buffer_size)
        if sizes is not None:
            snapped = snap_asio_buffer_size(target, sizes)
            if snapped != target:
                logger.info(
                    "ASIO buffer size %d snapped to %d (driver constraints)",
                    target, snapped,
                )
            target = snapped
        logger.info("ASIO buffer size: %d (user-selected)", target)
        return target
    if sizes is not None and sizes[2] > 0:
        logger.info(
            "ASIO buffer size: %d (driver preferred / control panel)", sizes[2]
        )
        return sizes[2]
    return 0


def enumerate_asio_buffer_sizes(
    sizes: tuple[int, int, int, int]
) -> list[int]:
    """Enumerate a concise sorted list of buffer sizes the driver supports.

    Power-of-two granularity (-1) is enumerated exactly.  Arithmetic
    granularity snaps a ladder of common sizes to the driver's step so
    every listed value is valid (drivers with fine granularity support
    hundreds of sizes — listing them all is useless in a dropdown).
    min, max and preferred are always included.
    """
    min_size, max_size, preferred, granularity = sizes
    result: set[int] = set()
    if granularity == -1:
        v = 1
        while v < min_size:
            v *= 2
        while v <= max_size:
            result.add(v)
            v *= 2
    elif granularity > 0:
        ladder = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
        for t in ladder:
            result.add(snap_asio_buffer_size(t, sizes))
    for v in (min_size, max_size, preferred):
        if v > 0:
            result.add(v)
    return sorted(x for x in result if min_size <= x <= max_size)


def query_asio_native_sample_rate(
    device_index: Optional[int],
    stream_type: str = "output",
) -> Optional[int]:
    """Return the ASIO device's native (default) sample rate, or None.

    ASIO drivers typically operate at a fixed sample rate configured in the
    driver's control panel.  ``sd.query_devices()["default_samplerate"]``
    reflects this value.
    """
    if not is_device_on_asio(device_index, stream_type):
        return None
    try:
        if device_index is not None:
            info = sd.query_devices(device_index)
        else:
            info = sd.query_devices(kind=stream_type)
        return int(info["default_samplerate"])
    except Exception:
        return None
