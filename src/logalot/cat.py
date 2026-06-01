"""Auto-detect the rig's CAT serial port and launch ``rigctld`` on it.

macOS renumbers USB-serial devices across reconnects (``/dev/cu.SLAB_USBtoUART``
one session, ``...UART5`` the next), so hard-coding the device is fragile. This
probes each candidate by actually starting ``rigctld`` against it and asking for
the dial frequency through :class:`logalot.capture.RigctldClient` — the device
that answers is the CAT port, and that very ``rigctld`` is left running.

Needs the Hamlib ``rigctld``/``rigctl`` binaries on PATH (``brew install hamlib``).
"""
from __future__ import annotations

import glob
import shutil
import socket
import subprocess
import time

from .capture import CatError, RigctldClient

# USB-serial device name patterns we might find the rig on (macOS /dev/cu.*).
_DEVICE_GLOBS = ("/dev/cu.SLAB_USBtoUART*", "/dev/cu.usbserial*", "/dev/cu.usbmodem*")


def candidate_devices() -> list[str]:
    """Serial devices that could be a rig, stable-sorted and de-duplicated."""
    found: set[str] = set()
    for pattern in _DEVICE_GLOBS:
        found.update(glob.glob(pattern))
    return sorted(found)


def rigctld_path() -> str | None:
    return shutil.which("rigctld")


def rigctld_command(device: str, model: str, baud: int, port: int) -> str:
    """The equivalent hand-typed rigctld line (for display)."""
    cmd = f"rigctld -m {model} -r {device} -s {baud}"
    if port != 4532:
        cmd += f" -t {port}"
    return cmd


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _spawn_rigctld(device: str, model: str, baud: int, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [rigctld_path() or "rigctld", "-m", str(model), "-r", device,
         "-s", str(baud), "-t", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def stop_rigctld(proc: subprocess.Popen) -> None:
    """Terminate a rigctld we launched, escalating to kill, so it never lingers
    holding the serial port."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def probe_device(device: str, model: str, baud: int, port: int,
                 settle: float = 1.3, on_spawn=None,
                 ) -> tuple[subprocess.Popen | None, float | None]:
    """Start rigctld on ``device`` and try to read the frequency.

    ``on_spawn(proc)`` is called immediately after launch so the caller can track
    (and reap) the child even if interrupted mid-probe. Returns ``(proc, freq)``
    with rigctld left running on success, or ``(None, None)`` after cleaning up.
    """
    proc = _spawn_rigctld(device, model, baud, port)
    if on_spawn is not None:
        on_spawn(proc)
    time.sleep(settle)
    if proc.poll() is not None:        # rigctld exited (couldn't open device/port)
        return None, None
    try:
        with RigctldClient(port=port) as client:
            freq = client.freq_mhz()
    except CatError:
        stop_rigctld(proc)
        return None, None
    return proc, freq
