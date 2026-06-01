"""M1 — audio + CAT capture.

Two jobs, kept separate:

1. **CAT** via Hamlib's ``rigctld`` (this module). Run ``rigctld`` externally
   against the FTDX10, then poll over its TCP socket for frequency and mode.
   Frequency, mode and time come from here — never from the transcript.

       rigctld -m <model_id> -r /dev/tty.usbserial-XXXX -s 38400

   The CAT client is **stdlib-only** (just ``socket``): it is the reliable,
   always-importable half of capture and is fully tested without a rig. It speaks
   rigctld's *extended response protocol* (commands prefixed with ``+``), in which
   every reply is a series of ``Name: value`` lines terminated by a ``RPRT <code>``
   status line — so we can read deterministically and surface rig errors instead
   of silently parsing a truncated reply.

2. **Audio**: pull the FTDX10 USB CODEC RX stream into a ring buffer for the ASR
   module. Needs ``sounddevice`` (PortAudio), so it lives behind the ``[capture]``
   extra and is imported lazily — importing this module must never require it.

Interfaces are defined so ``asr`` and the daemon can be built against them.
"""
from __future__ import annotations

import datetime as _dt
import socket
from dataclasses import dataclass
from typing import Protocol

from .models import band_for_freq

# Hamlib reports the rig's modulation token; ADIF wants its own MODE enum. Map
# the analog/CW cases authoritatively (CAT genuinely knows them). For the packet
# data modes Hamlib only knows "this is a data sub-channel" (PKTUSB/PKTLSB/PKTFM),
# not *which* protocol (FT8 vs RTTY vs ...) — that has to come from the operator
# or the parse layer — so we map them to ADIF "DATA" and keep the raw token in
# RigState.raw_mode rather than guessing. Unknown tokens are passed through
# upper-cased so nothing is ever silently dropped.
_HAMLIB_TO_ADIF_MODE: dict[str, str] = {
    "USB": "SSB", "LSB": "SSB",
    "CW": "CW", "CWR": "CW",
    "FM": "FM", "WFM": "FM", "FMN": "FM",
    "AM": "AM", "AMN": "AM",
    "RTTY": "RTTY", "RTTYR": "RTTY",
    "PKTUSB": "DATA", "PKTLSB": "DATA", "PKTFM": "DATA", "PKT": "DATA",
}


def adif_mode(hamlib_mode: str) -> str:
    """Translate a Hamlib mode token to its ADIF MODE. Unknown tokens pass
    through upper-cased so information is never lost."""
    m = hamlib_mode.strip().upper()
    return _HAMLIB_TO_ADIF_MODE.get(m, m)


def utc_now_adif() -> tuple[str, str]:
    """(qso_date, time_on) as ADIF strings from the system clock in UTC.

    Time is a capture-side concern, not a CAT one — the rig has no clock we trust
    — so the daemon stamps a QSO here. CLAUDE.md: system clock in UTC."""
    now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y%m%d"), now.strftime("%H%M%S")


@dataclass(slots=True)
class RigState:
    """A snapshot of what the rig is doing. ``mode`` is the ADIF MODE (what the
    QSO record wants); ``raw_mode`` keeps the original Hamlib token for context
    (e.g. distinguishing USB from LSB, or PKTUSB behind a "DATA")."""

    freq_mhz: float | None
    mode: str | None
    band: str | None = None
    raw_mode: str | None = None
    strength_dbs9: int | None = None   # S-meter, dB relative to S9 (Hamlib convention)
    ptt: bool | None = None            # True while *we* are transmitting


class CatClient(Protocol):
    def state(self) -> RigState: ...


class AudioSource(Protocol):
    def read(self, seconds: float) -> "object":  # numpy.ndarray at runtime
        ...


class CatError(RuntimeError):
    """rigctld is unreachable, dropped the connection, or returned an error.

    The capture daemon should treat this as "no fresh CAT data" and fall back to
    the last known state — never let it take down the manual logging path."""


class _Disconnected(CatError):
    """Transport-level failure (EOF, socket error) — worth one reconnect+retry.
    A rig-level ``RPRT`` error is a plain CatError and is *not* retried."""


class RigctldClient:
    """TCP client for a running ``rigctld``.

    Holds one persistent connection and reconnects transparently if it drops, so
    a brief ``rigctld`` restart doesn't wedge the daemon. Not thread-safe: poll it
    from a single thread (or guard it with your own lock).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4532,
        timeout: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._reader: object | None = None  # socket.makefile() text stream

    # --- connection management ------------------------------------------------

    def _connect(self) -> None:
        try:
            sock = socket.create_connection((self.host, self.port), self.timeout)
        except OSError as e:
            raise CatError(f"cannot reach rigctld at {self.host}:{self.port}: {e}") from e
        sock.settimeout(self.timeout)
        self._sock = sock
        # Line-buffered text view for reading replies; we still write via sendall.
        self._reader = sock.makefile("r", encoding="ascii", newline="\n")

    def close(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()  # type: ignore[attr-defined]
            except OSError:
                pass
            self._reader = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self) -> "RigctldClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- protocol -------------------------------------------------------------

    def _command(self, cmd: str) -> list[str]:
        """Run one extended-protocol command (without the leading ``+``) and
        return its payload lines. Raises CatError on any transport failure or a
        non-zero ``RPRT`` status. Retries once across a dropped connection so a
        rigctld blip is invisible to the caller.

        rigctld's extended replies are: an echo line (the canonical command name,
        e.g. ``get_freq:``), then one or more payload lines, then ``RPRT <code>``.
        Payload lines are usually ``Label: value`` but some (e.g. the S-meter from
        ``l STRENGTH``) are a bare value — so we return the raw lines and let
        :meth:`_value` extract, rather than assuming everything is labelled."""
        try:
            return self._command_once(cmd)
        except _Disconnected:
            # Stale socket (rigctld restarted, idle timeout): reconnect and retry
            # exactly once. A second failure is real and propagates. Rig-level
            # RPRT errors are *not* _Disconnected, so they don't reconnect.
            self.close()
            return self._command_once(cmd)

    def _command_once(self, cmd: str) -> list[str]:
        if self._sock is None:
            self._connect()
        assert self._sock is not None and self._reader is not None
        try:
            self._sock.sendall(f"+{cmd}\n".encode("ascii"))
            lines: list[str] = []
            while True:
                line = self._reader.readline()  # type: ignore[attr-defined]
                if line == "":  # EOF: peer closed the connection
                    raise _Disconnected("rigctld closed the connection")
                line = line.rstrip("\n")
                if line.startswith("RPRT "):
                    code = line[5:].strip()
                    if code != "0":
                        raise CatError(f"rigctld error on '{cmd}': RPRT {code}")
                    # Drop the leading echo line; return only the payload.
                    return lines[1:]
                lines.append(line)
        except (OSError, ValueError) as e:
            # OSError: socket-level failure. ValueError: read on a closed file.
            raise _Disconnected(f"CAT transport failure on '{cmd}': {e}") from e

    @staticmethod
    def _value(line: str) -> str:
        """Text after the first ``:`` (``Frequency: 14074000`` -> ``14074000``),
        or the whole line when there is no label (bare S-meter value)."""
        _, sep, rest = line.partition(":")
        return rest.strip() if sep else line.strip()

    def _one(self, cmd: str) -> str:
        lines = self._command(cmd)
        if not lines:
            raise CatError(f"empty rigctld reply to '{cmd}'")
        return self._value(lines[0])

    # --- queries --------------------------------------------------------------

    def freq_mhz(self) -> float:
        """Current dial frequency in MHz (rigctld reports Hz)."""
        raw = self._one("f")
        try:
            return int(raw) / 1_000_000
        except ValueError as e:
            raise CatError(f"unparseable frequency {raw!r}") from e

    def raw_mode(self) -> str:
        """Current Hamlib mode token (e.g. ``USB``, ``CW``, ``PKTUSB``)."""
        return self._one("m")

    def ptt(self) -> bool:
        """True while the rig is transmitting (``get_ptt``). Lets us label audio
        segments as ME (TX) vs the remote operator (RX) without tapping the mic."""
        return self._one("t") not in ("0", "")

    def strength_dbs9(self) -> int:
        """S-meter reading in dB relative to S9 (Hamlib ``STRENGTH`` level).
        0 dB == S9, -54 dB ~= S0, +10 dB == S9+10. UI converts to S-units."""
        raw = self._one("l STRENGTH")
        try:
            return int(raw)
        except ValueError as e:
            raise CatError(f"unparseable strength {raw!r}") from e

    def state(self) -> RigState:
        """Core snapshot: frequency, ADIF mode, band, raw mode. Two round-trips,
        no optional levels — cheap and always available on any rig."""
        freq = self.freq_mhz()
        raw = self.raw_mode()
        return RigState(
            freq_mhz=freq,
            mode=adif_mode(raw),
            band=band_for_freq(freq),
            raw_mode=raw,
        )

    def monitor(self) -> RigState:
        """Richer snapshot for the live dashboard: core state plus S-meter and
        PTT. S-meter/PTT are best-effort — a rig that doesn't support a level
        leaves it ``None`` rather than failing the whole read."""
        st = self.state()
        try:
            st.strength_dbs9 = self.strength_dbs9()
        except CatError:
            pass
        try:
            st.ptt = self.ptt()
        except CatError:
            pass
        return st


class SounddeviceSource:
    """RX audio from the rig's USB CODEC. TODO(M1 audio).

    ``sounddevice`` is imported lazily inside :meth:`read` so this module stays
    importable (and the CAT client usable) without the ``[capture]`` extra."""

    def __init__(self, device_name: str = "FTDX10", samplerate: int = 16000) -> None:
        self.device_name = device_name
        self.samplerate = samplerate

    def read(self, seconds: float):
        raise NotImplementedError("M1 audio: capture from PortAudio device into buffer")
