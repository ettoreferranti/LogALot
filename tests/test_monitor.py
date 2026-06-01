"""Tests for the Step A rig-monitor.

Drives the snapshot/format logic directly against the in-process fake rigctld
(reused from the capture tests), so no HTTP client / httpx is needed.
"""
import pytest

pytest.importorskip("fastapi")

from logalot.capture import RigctldClient  # noqa: E402
from logalot.monitor import _PAGE, _snapshot, smeter_label, smeter_pct  # noqa: E402
from tests.test_capture import FakeRigctld  # reuse the fake rigctld  # noqa: E402


def test_smeter_label():
    assert smeter_label(None) == "—"
    assert smeter_label(0) == "S9"
    assert smeter_label(-54) == "S0"
    assert smeter_label(-24) == "S5"     # 9 + (-24/6) = 5
    assert smeter_label(10) == "S9+10dB"


def test_smeter_pct_monotonic_and_clamped():
    assert smeter_pct(None) == 0
    assert smeter_pct(-54) == 0
    assert smeter_pct(30) == 100
    assert smeter_pct(-100) == 0         # clamped
    assert smeter_pct(-24) > smeter_pct(-40)


def test_snapshot_ok():
    s = FakeRigctld({
        "f": "get_freq:\nFrequency: 14074000\nRPRT 0\n",
        "m": "get_mode:\nMode: USB\nPassband: 2400\nRPRT 0\n",
        "t": "get_ptt:\nPTT: 0\nRPRT 0\n",
        "l STRENGTH": "get_level: STRENGTH\n-24\nRPRT 0\n",
    })
    try:
        with RigctldClient(s.host, s.port) as c:
            d = _snapshot(c)
    finally:
        s.stop()
    assert d["ok"] is True
    assert d["band"] == "20m" and d["mode"] == "SSB"
    assert d["smeter_label"] == "S5"
    assert d["ptt"] is False
    assert d["freq_display"].startswith("14.074")
    assert "utc" in d


def test_snapshot_rig_offline():
    # Nothing listening: snapshot reports offline rather than raising.
    with RigctldClient("127.0.0.1", 1) as c:
        d = _snapshot(c)
    assert d["ok"] is False
    assert "error" in d and "utc" in d


def test_page_is_self_contained():
    assert "rig monitor" in _PAGE
    # Local-first: no external/CDN assets pulled at runtime.
    assert "http://" not in _PAGE and "https://" not in _PAGE
