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
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
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

    def start(self) -> None:
        """Start stream with robust fallback."""
        if self._stream is not None:
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
