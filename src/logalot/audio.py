"""Step B — RX audio capture from the rig's USB CODEC.

Pulls the received audio off the FTDX10 USB Audio CODEC (a CoreAudio input
device) into a ring buffer: the level meter reads it now, the ASR segmenter
(Step C) will read segments out of it next. RX audio only carries the *received*
signal — your own transmitted voice isn't in this stream — so it is, by
definition, the remote operator; PTT from CAT tells us when to ignore it because
we're transmitting.

``sounddevice`` (PortAudio) + ``numpy`` live in the ``[capture]`` extra and are
imported lazily, so this module is importable without them (the pure helpers and
the dashboard's "audio off" path don't need hardware).
"""
from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass
from typing import Protocol

# The FTDX10 enumerates on macOS as "USB AUDIO  CODEC" (note: vendor strings vary
# in spacing/case), so we match on whitespace-collapsed, lower-cased names.
DEFAULT_DEVICE = "USB Audio CODEC"
DEFAULT_SAMPLERATE = 48000
DBFS_FLOOR = -60.0  # bottom of the meter; quieter than this reads as 0%


class AudioError(RuntimeError):
    """No sounddevice/numpy, or the requested input device can't be opened.
    The dashboard treats this as 'audio off', never a crash."""


class AudioSource(Protocol):
    def level_dbfs(self) -> float: ...
    def read_last(self, seconds: float): ...  # numpy.ndarray at runtime


# --- pure helpers (testable without hardware) --------------------------------

def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def rms_to_dbfs(rms: float) -> float:
    """RMS of float samples in [-1, 1] -> dBFS. Silence floors at DBFS_FLOOR."""
    if rms <= 1e-9:
        return DBFS_FLOOR
    return max(DBFS_FLOOR, 20.0 * math.log10(rms))


def dbfs_to_pct(dbfs: float, floor: float = DBFS_FLOOR) -> int:
    """Map dBFS onto 0–100 for a VU bar (floor → 0, 0 dBFS → 100)."""
    return int(max(0, min(100, (dbfs - floor) / (0.0 - floor) * 100)))


@dataclass(slots=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    samplerate: float


def list_input_devices() -> list[DeviceInfo]:
    """All CoreAudio devices with at least one input channel."""
    import sounddevice as sd

    out: list[DeviceInfo] = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append(DeviceInfo(i, d["name"], d["max_input_channels"],
                                  d["default_samplerate"]))
    return out


def find_input_device(name: str) -> DeviceInfo | None:
    """First input device whose name contains ``name`` (whitespace/case
    insensitive). Handles the FTDX10's "USB AUDIO  CODEC" double-space quirk."""
    want = _norm(name)
    for d in list_input_devices():
        if want in _norm(d.name):
            return d
    return None


# --- ring buffer (numpy, testable without an audio device) -------------------

class RingBuffer:
    """Fixed-capacity circular buffer of mono float32 samples. Thread-safe: the
    PortAudio callback writes while the web/ASR side reads."""

    def __init__(self, capacity: int) -> None:
        import numpy as np

        self._np = np
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()

    def write(self, data) -> None:
        n = len(data)
        if n == 0:
            return
        with self._lock:
            if n >= self._cap:                      # incoming bigger than buffer
                self._buf[:] = data[-self._cap:]
                self._write = 0
                self._filled = self._cap
                return
            end = self._write + n
            if end <= self._cap:
                self._buf[self._write:end] = data
            else:                                   # wrap
                first = self._cap - self._write
                self._buf[self._write:] = data[:first]
                self._buf[: n - first] = data[first:]
            self._write = end % self._cap
            self._filled = min(self._cap, self._filled + n)

    def read_last(self, n: int):
        """Most recent ``n`` samples (or fewer if not yet filled), oldest first."""
        with self._lock:
            n = min(n, self._filled)
            start = (self._write - n) % self._cap
            if start + n <= self._cap:
                return self._buf[start:start + n].copy()
            first = self._cap - start
            out = self._np.empty(n, dtype=self._np.float32)
            out[:first] = self._buf[start:]
            out[first:] = self._buf[: n - first]
            return out


# --- live capture ------------------------------------------------------------

class AudioCapture:
    """RX audio from an input device into a ring buffer, with a live level.

    Mono: if the device exposes >1 input channel (the FTDX10 codec is stereo) we
    take the first channel. Capture runs in PortAudio's callback thread; reads of
    :meth:`level_dbfs` / :meth:`read_last` are safe from other threads.
    """

    def __init__(self, device: str | None = None,
                 samplerate: int = DEFAULT_SAMPLERATE,
                 ring_seconds: float = 30.0) -> None:
        self.device_name = device or DEFAULT_DEVICE
        self.samplerate = samplerate
        self.ring_seconds = ring_seconds
        self._stream = None
        self._device: DeviceInfo | None = None
        self._ring: RingBuffer | None = None
        self._last_rms = 0.0          # float store/read is atomic enough for a meter
        self._overflows = 0
        self._taps: list[queue.Queue] = []   # lossless ordered feeds (e.g. the ASR worker)

    def tap(self) -> queue.Queue:
        """Register a queue that receives every captured block (a copy), in
        order. Unlike the ring buffer (which overwrites), a tap loses nothing
        even if the consumer stalls during a slow transcription."""
        q: queue.Queue = queue.Queue()
        self._taps.append(q)
        return q

    @property
    def device_label(self) -> str:
        return self._device.name if self._device else self.device_name

    def is_running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError as e:
            raise AudioError("audio needs the [capture] extra: pip install -e '.[capture]'") from e

        dev = find_input_device(self.device_name)
        if dev is None:
            raise AudioError(
                f"input device matching {self.device_name!r} not found "
                f"(have: {[d.name for d in list_input_devices()]})"
            )
        self._device = dev
        self._ring = RingBuffer(int(self.ring_seconds * self.samplerate))
        try:
            self._stream = sd.InputStream(
                device=dev.index, channels=1, samplerate=self.samplerate,
                dtype="float32", callback=self._callback,
            )
            self._stream.start()
        except Exception as e:  # sd.PortAudioError and friends
            self._stream = None
            raise AudioError(f"cannot open {dev.name!r}: {e}") from e

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self._overflows += 1
        import numpy as np

        mono = indata[:, 0] if indata.ndim > 1 else indata
        if frames:
            self._last_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        if self._ring is not None:
            self._ring.write(mono)
        if self._taps:
            block = mono.copy()   # indata is reused by PortAudio; copy before handing off
            for q in self._taps:
                q.put(block)

    def level_dbfs(self) -> float:
        return rms_to_dbfs(self._last_rms)

    def read_last(self, seconds: float):
        if self._ring is None:
            raise AudioError("capture not started")
        return self._ring.read_last(int(seconds * self.samplerate))

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
