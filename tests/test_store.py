import json

from logalot.models import QSO, band_for_freq
from logalot.store import Store


def _qso(call="HB9XX"):
    return QSO(call=call, qso_date="20260601", time_on="123000", mode="SSB",
               band="20m", freq_mhz=14.250, rst_sent="59", rst_rcvd="57")


def test_append_and_read(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.add_qso(_qso("HB9XX"))
    s.add_qso(_qso("DL1ABC"))
    calls = [q.call for q in s.all_qso()]
    assert calls == ["HB9XX", "DL1ABC"]


def test_chain_verifies(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    for c in ["HB9XX", "DL1ABC", "W1AW"]:
        s.add_qso(_qso(c))
    assert s.verify_chain() is True


def test_chain_detects_tampering(tmp_path):
    path = str(tmp_path / "t.db")
    s = Store(path)
    s.add_qso(_qso("HB9XX"))
    s.add_qso(_qso("DL1ABC"))
    # Tamper with a payload directly, behind the append-only API.
    bad = json.dumps({"call": "EVIL"})
    s.conn.execute("UPDATE qso SET payload = ? WHERE id = 1", (bad,))
    s.conn.commit()
    assert s.verify_chain() is False


def test_candidate_confirm_flow(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    cid = s.add_candidate({"call": "HB9XX", "mode": "SSB"}, transcript="...",
                          confidence="ok")
    assert len(s.list_candidates()) == 1
    s.confirm_candidate(cid, _qso("HB9XX"))
    assert len(s.list_candidates()) == 0
    assert len(s.all_qso()) == 1
    assert s.verify_chain() is True


def test_band_for_freq():
    assert band_for_freq(14.250) == "20m"
    assert band_for_freq(7.100) == "40m"
    assert band_for_freq(999.0) is None
    assert band_for_freq(None) is None
