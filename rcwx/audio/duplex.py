"""ASIO duplex stream — opens input and output on a single sd.Stream.

ASIO drivers (e.g. MiniFuse ASIO Driver) run in exclusive full-duplex mode.
Opening separate InputStream + OutputStream fails with ``Device unavailable``
because the first stream exclusively claims the driver.  This module wraps
``sd.Stream`` so both directions share a single driver session.

ASIO devices typically have a fixed sample rate and buffer size configured
in the driver's control panel.  This class queries the device's native
sample rate AND its preferred (= control panel) buffer size, then passes
``blocksize=0`` with ``latency = preferred / samplerate`` so PortAudio
creates the ASIO buffers at exactly the panel value.  ``latency`` must be
explicit and exact: PortAudio reports the driver's min/max buffer sizes as
its 'low'/'high' latency defaults, so both sounddevice's default
(``'high'`` → max, e.g. 2048) and a naive ``'low'`` (→ min, e.g. 64)
silently rebuild the ASIO buffers away from the panel setting.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from rcwx.audio.input import select_channel
from rcwx.audio.stream_base import (
    parse_output_channel_pair,
    query_asio_buffer_sizes,
    query_asio_channel_names,
    query_asio_native_sample_rate,
    snap_asio_buffer_size as _snap_buffer_size,
)

logger = logging.getLogger(__name__)


class AsioDuplexStream:
    """Full-duplex ASIO stream bridging separate input/output callbacks.

    Parameters
    ----------
    input_device, output_device:
        PortAudio device indices (both must be on the ASIO host API).
    sample_rate:
        Requested sample rate (Hz).  The device's native rate is tried first;
        *sample_rate* is used as a secondary candidate.
    input_channels, output_channels:
        Channel counts handed to ``sd.Stream``.
    blocksize:
        Unused; ASIO always uses ``blocksize=0`` (driver-preferred).
        Accepted for call-site compatibility.
    input_callback:
        ``(mono_audio: np.ndarray) -> None`` — receives channel-selected mono
        float32 audio, same contract as ``AudioInput``'s callback.
    output_callback:
        ``(frames: int) -> np.ndarray`` — must return *frames* float32 samples,
        same contract as ``AudioOutput``'s callback.
    channel_selection:
        How to down-mix multi-channel input: ``"auto"`` / ``"0"`` /
        ``"1"`` / ``"average"`` / any channel index as string.
    output_channel_selection:
        Which output channel pair to route audio to: ``"auto"`` (broadcast
        to all) / ``"0,1"`` / ``"2,3"`` etc.
    requested_buffer_size:
        Explicit ASIO buffer size in frames.  0 (default) follows the
        driver's control panel (preferredSize).  Non-zero values are
        snapped to the driver's min/max/granularity constraints.
    """

    def __init__(
        self,
        input_device: int,
        output_device: int,
        sample_rate: int,
        input_channels: int,
        output_channels: int,
        blocksize: int,
        input_callback: Callable[[np.ndarray], None],
        output_callback: Callable[[int], np.ndarray],
        channel_selection: str = "auto",
        output_channel_selection: str = "auto",
        requested_buffer_size: int = 0,
    ) -> None:
        self._input_callback = input_callback
        self._output_callback = output_callback
        self._channel_selection = channel_selection

        # Parse output channel selection into index pair
        self._output_ch_indices = parse_output_channel_pair(output_channel_selection)
        if self._output_ch_indices is None and output_channel_selection != "auto":
            logger.warning(
                "Unparseable output_channel_selection %r — treating as auto "
                "(output to Ch 1-2 only)",
                output_channel_selection,
            )

        # ASIO LOOPBACK input channels carry the device's own playback (= the
        # voice changer output).  Exclude them from auto/average input
        # selection so the output can never be auto-picked back into the
        # pipeline as input (digital-delay-like feedback loop).
        self._input_exclude_channels: Optional[frozenset[int]] = None
        try:
            names = query_asio_channel_names(input_device, "input")
            loopback = frozenset(
                i for i, name in enumerate(names) if "loopback" in name.lower()
            )
            if loopback:
                self._input_exclude_channels = loopback
                logger.info(
                    "Excluding ASIO loopback input channels from auto selection: %s",
                    sorted(loopback),
                )
        except Exception as e:
            logger.debug("Could not query ASIO input channel names: %s", e)

        self._stream: Optional[sd.Stream] = None
        self._actual_sample_rate: int = sample_rate
        self._stopped = False

        # --- Build candidate sample rates: device native first ---
        rates: list[int] = []
        for dev_idx, st in [(output_device, "output"), (input_device, "input")]:
            native = query_asio_native_sample_rate(dev_idx, st)
            if native is not None and native not in rates:
                rates.append(native)
        if sample_rate not in rates:
            rates.append(sample_rate)
        for r in [48000, 44100, 96000]:
            if r not in rates:
                rates.append(r)

        logger.info(
            "ASIO duplex: in=%s out=%s candidate rates=%s",
            input_device, output_device, rates,
        )

        # blocksize=0 + latency=size/sr → PortAudio creates the ASIO
        # buffers at exactly that size.  'low'/'high' latency classes map
        # to the driver's min/max buffer instead and would override the
        # control panel in either direction.  The size is the user's
        # explicit choice when set, else the panel value (preferredSize).
        buffer_sizes = query_asio_buffer_sizes(output_device, "output")
        if buffer_sizes is not None:
            logger.info(
                "ASIO buffer sizes: min=%d max=%d preferred=%d granularity=%d",
                *buffer_sizes,
            )
        target_size = 0
        if requested_buffer_size and requested_buffer_size > 0:
            target_size = int(requested_buffer_size)
            if buffer_sizes is not None:
                target_size = _snap_buffer_size(target_size, buffer_sizes)
                if target_size != requested_buffer_size:
                    logger.info(
                        "ASIO buffer size %d snapped to %d (driver constraints)",
                        requested_buffer_size, target_size,
                    )
            logger.info("ASIO buffer size: %d (user-selected)", target_size)
        elif buffer_sizes is not None and buffer_sizes[2] > 0:
            target_size = buffer_sizes[2]
            logger.info(
                "ASIO buffer size: %d (driver preferred / control panel)",
                target_size,
            )
        last_error: Optional[Exception] = None
        for sr in rates:
            if target_size > 0:
                latency = target_size / float(sr)
            else:
                # Size unknown: target ~10ms, a sane middle ground
                # between the min (fragile) and max (laggy).
                latency = 0.010
            try:
                self._stream = sd.Stream(
                    device=(input_device, output_device),
                    samplerate=sr,
                    channels=(input_channels, output_channels),
                    blocksize=0,
                    dtype=np.float32,
                    latency=latency,
                    callback=self._duplex_callback,
                )
                self._actual_sample_rate = int(self._stream.samplerate)
                logger.info(
                    "ASIO duplex stream created: in=%s out=%s sr=%d bs=%s "
                    "latency=%s",
                    input_device, output_device,
                    self._actual_sample_rate, self._stream.blocksize,
                    self._stream.latency,
                )
                return
            except Exception as e:
                last_error = e
                logger.debug("ASIO duplex open failed (sr=%d): %s", sr, e)

        raise RuntimeError(
            f"ASIO duplex stream could not be opened. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Public interface (mirrors AudioInput / AudioOutput)
    # ------------------------------------------------------------------

    @property
    def actual_sample_rate(self) -> int:
        return self._actual_sample_rate

    def start(self) -> None:
        if self._stream is not None and not self._stream.active:
            self._stream.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error stopping ASIO duplex stream: %s", e)
            finally:
                self._stream = None

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    # ------------------------------------------------------------------
    # Duplex callback
    # ------------------------------------------------------------------

    def _duplex_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        # An unhandled exception in a sounddevice callback kills the stream
        # immediately (paAbort) with no notification to the main thread.
        # In duplex mode that would take out both input AND output, so each
        # side is wrapped individually.
        if status:
            logger.warning("ASIO duplex status: %s", status)

        # --- Input side: channel selection → mono callback ---
        try:
            audio = select_channel(
                indata, self._channel_selection, self._input_exclude_channels
            )
            self._input_callback(audio)
        except Exception as e:
            logger.error("ASIO duplex input callback error: %s", e)

        # --- Output side: request audio and fill outdata ---
        try:
            output = self._output_callback(frames)
            if len(output) >= frames:
                mono = output[:frames]
            else:
                mono = np.zeros(frames, dtype=np.float32)
                if len(output) > 0:
                    mono[: len(output)] = output

            # Route mono to selected output channels
            if self._output_ch_indices is not None and outdata.ndim > 1:
                outdata.fill(0)
                ch_a, ch_b = self._output_ch_indices
                if ch_a < outdata.shape[1]:
                    outdata[:, ch_a] = mono
                if ch_b < outdata.shape[1]:
                    outdata[:, ch_b] = mono
            elif outdata.ndim > 1 and outdata.shape[1] > 1:
                # Auto: first stereo pair only.  Broadcasting to ALL channels
                # would also feed ASIO LOOPBACK outputs, whose signal returns
                # on the loopback inputs (feedback loop).
                outdata.fill(0)
                outdata[:, 0] = mono
                outdata[:, 1] = mono
            else:
                outdata[:, 0] = mono
        except Exception as e:
            logger.error("ASIO duplex output callback error: %s", e)
            outdata.fill(0)
