from logalot.export_adif import header, record, to_adif
from logalot.models import QSO


def _qso():
    return QSO(call="HB9XX", qso_date="20260601", time_on="123000", mode="FT8",
               band="20m", freq_mhz=14.074, rst_sent="-07", rst_rcvd="-12",
               my_call="HB9IKS")


def test_field_length_is_byte_accurate():
    rec = record(_qso())
    assert "<CALL:5>HB9XX" in rec
    assert "<STATION_CALLSIGN:6>HB9IKS" in rec
    assert rec.endswith("<EOR>")


def test_freq_formatting():
    rec = record(_qso())
    assert "<FREQ:6>14.074" in rec   # trailing zeros stripped, 6 chars


def test_optional_fields_omitted():
    q = QSO(call="W1AW", qso_date="20260601", time_on="000000", mode="CW")
    rec = record(q)
    assert "NAME" not in rec
    assert "QTH" not in rec
    assert "<MODE:2>CW" in rec


def test_header_and_eoh():
    h = header()
    assert "<ADIF_VER:5>3.1.4" in h
    assert h.rstrip().endswith("<EOH>")


def test_to_adif_roundtrip_shape():
    out = to_adif([_qso(), _qso()])
    assert out.count("<EOR>") == 2
    assert "<EOH>" in out
