"""Tests for the rigctld CAT client.

A tiny in-process TCP server speaks rigctld's extended response protocol so the
whole client is exercised end-to-end without a rig or Hamlib installed.
"""
import socket
import threading

import pytest

from logalot.capture import (
    CatError,
    RigctldClient,
    adif_mode,
    utc_now_adif,
)


# A scripted rigctld: maps an incoming command line to the reply bytes it sends.
# Replies follow the extended protocol: `Name: value` lines + a `RPRT <code>`.
class FakeRigctld:
    def __init__(self, responses: dict[str, str], *, drop_after: int | None = None):
        self.responses = responses
        self.drop_after = drop_after  # close the socket after N commands (once)
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.host, self.port = self.srv.getsockname()
        self.connections = 0
        self._stop = False
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            self.connections += 1
            self._handle(conn)

    def _handle(self, conn):
        conn.settimeout(2.0)
        served = 0
        with conn:
            f = conn.makefile("r", encoding="ascii", newline="\n")
            while True:
                try:
                    line = f.readline()
                except OSError:
                    return
                if line == "":
                    return
                if self.drop_after is not None and served >= self.drop_after:
                    self.drop_after = None  # only drop once, then behave
                    return  # close mid-stream to simulate a rigctld restart
                cmd = line.rstrip("\n").lstrip("+")
                reply = self.responses.get(cmd, "RPRT -1\n")
                conn.sendall(reply.encode("ascii"))
                served += 1

    def stop(self):
        self._stop = True
        try:
            self.srv.close()
        except OSError:
            pass


@pytest.fixture
def rig():
    servers: list[FakeRigctld] = []

    def make(responses, **kw):
        s = FakeRigctld(responses, **kw)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.stop()


# Replies mimic real rigctld extended protocol: a leading echo line (the
# canonical command name), payload line(s), then RPRT. The S-meter payload is a
# bare value with no label, which is exactly the case that used to be dropped.
FREQ_20M = "get_freq:\nFrequency: 14074000\nRPRT 0\n"
MODE_USB = "get_mode:\nMode: USB\nPassband: 2400\nRPRT 0\n"
PTT_RX = "get_ptt:\nPTT: 0\nRPRT 0\n"
STRENGTH = "get_level: STRENGTH\n-24\nRPRT 0\n"


def test_freq_mhz(rig):
    s = rig({"f": FREQ_20M})
    with RigctldClient(s.host, s.port) as c:
        assert c.freq_mhz() == pytest.approx(14.074)


def test_raw_mode(rig):
    s = rig({"m": MODE_USB})
    with RigctldClient(s.host, s.port) as c:
        assert c.raw_mode() == "USB"


def test_state_combines_freq_mode_band(rig):
    s = rig({"f": FREQ_20M, "m": MODE_USB})
    with RigctldClient(s.host, s.port) as c:
        st = c.state()
    assert st.freq_mhz == pytest.approx(14.074)
    assert st.mode == "SSB"        # ADIF mode
    assert st.raw_mode == "USB"    # original Hamlib token kept
    assert st.band == "20m"        # derived from freq via models.band_for_freq


def test_state_data_mode_maps_to_data_keeps_raw(rig):
    s = rig({"f": "get_freq:\nFrequency: 7074000\nRPRT 0\n",
             "m": "get_mode:\nMode: PKTUSB\nPassband: 3000\nRPRT 0\n"})
    with RigctldClient(s.host, s.port) as c:
        st = c.state()
    assert st.mode == "DATA"
    assert st.raw_mode == "PKTUSB"
    assert st.band == "40m"


def test_ptt_parsing(rig):
    s = rig({"t": PTT_RX})
    with RigctldClient(s.host, s.port) as c:
        assert c.ptt() is False
    s2 = rig({"t": "get_ptt:\nPTT: 1\nRPRT 0\n"})
    with RigctldClient(s2.host, s2.port) as c:
        assert c.ptt() is True


def test_strength_bare_value_not_dropped(rig):
    # Regression: the S-meter payload is a bare line with no `Label:`; it must
    # survive parsing rather than being silently discarded.
    s = rig({"l STRENGTH": STRENGTH})
    with RigctldClient(s.host, s.port) as c:
        assert c.strength_dbs9() == -24


def test_monitor_includes_smeter_and_ptt(rig):
    s = rig({"f": FREQ_20M, "m": MODE_USB, "t": PTT_RX, "l STRENGTH": STRENGTH})
    with RigctldClient(s.host, s.port) as c:
        st = c.monitor()
    assert st.band == "20m" and st.mode == "SSB"
    assert st.strength_dbs9 == -24
    assert st.ptt is False


def test_monitor_tolerates_missing_level(rig):
    # A rig that doesn't support STRENGTH/PTT (RPRT error) should still yield a
    # core snapshot with those fields left as None, not blow up.
    s = rig({"f": FREQ_20M, "m": MODE_USB, "t": "RPRT -1\n", "l STRENGTH": "RPRT -1\n"})
    with RigctldClient(s.host, s.port) as c:
        st = c.monitor()
    assert st.freq_mhz == pytest.approx(14.074)
    assert st.strength_dbs9 is None
    assert st.ptt is None


def test_rprt_error_raises(rig):
    s = rig({"f": "RPRT -5\n"})
    with RigctldClient(s.host, s.port) as c:
        with pytest.raises(CatError):
            c.freq_mhz()


def test_unreachable_raises():
    # Nothing listening on this port.
    c = RigctldClient("127.0.0.1", 1)
    with pytest.raises(CatError):
        c.freq_mhz()


def test_reconnects_after_dropped_connection(rig):
    # Server drops the connection on the very first command; the client should
    # transparently reconnect and retry, so the call still succeeds.
    s = rig({"f": FREQ_20M}, drop_after=0)
    with RigctldClient(s.host, s.port) as c:
        assert c.freq_mhz() == pytest.approx(14.074)
    assert s.connections == 2  # initial + reconnect


def test_unknown_mode_passes_through():
    assert adif_mode("USB") == "SSB"
    assert adif_mode("cw") == "CW"
    assert adif_mode("PKTLSB") == "DATA"
    assert adif_mode("C4FM") == "C4FM"  # unknown token preserved, upper-cased


def test_utc_now_adif_shapes():
    date, t = utc_now_adif()
    assert len(date) == 8 and date.isdigit()
    assert len(t) == 6 and t.isdigit()
