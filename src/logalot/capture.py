"""M1 — audio + CAT capture. STUB.

Two jobs, kept separate:

1. **CAT** via Hamlib's ``rigctld``. Run it externally against the FTDX10, then
   poll over its TCP socket for frequency (``f``) and mode (``m``). Frequency,
   mode and time come from here — never from the transcript.

       rigctld -m <model_id> -r /dev/tty.usbserial-XXXX -s 38400

2. **Audio**: pull the FTDX10 USB CODEC RX stream into a ring buffer for the ASR
   module. Use ``sounddevice`` (PortAudio) and pick the rig's input device by
   name.

Interfaces are defined now so ``asr`` and the daemon can be built against them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class RigState:
    freq_mhz: float | None
    mode: str | None


class CatClient(Protocol):
    def state(self) -> RigState: ...


class AudioSource(Protocol):
    def read(self, seconds: float) -> "object":  # numpy.ndarray at runtime
        ...


class RigctldClient:
    """TCP client for a running rigctld. TODO(M1)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4532) -> None:
        self.host = host
        self.port = port

    def state(self) -> RigState:
        raise NotImplementedError("M1: connect to rigctld, send 'f' and 'm'")


class SounddeviceSource:
    """RX audio from the rig's USB CODEC. TODO(M1)."""

    def __init__(self, device_name: str = "FTDX10", samplerate: int = 16000) -> None:
        self.device_name = device_name
        self.samplerate = samplerate

    def read(self, seconds: float):
        raise NotImplementedError("M1: capture from PortAudio device into buffer")
