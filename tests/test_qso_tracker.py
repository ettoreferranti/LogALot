"""Tests for the stateful QSO assembly (no MLX/audio)."""
from logalot.qso_tracker import OverRecord, QsoTracker


def _over(speaker, addressed=None, **kw):
    return OverRecord(speaker=speaker, addressed=addressed, band="20m",
                      freq_mhz=14.2, mode="SSB", utc="15:00:00", **kw)


def test_cq_then_answer_forms_one_qso():
    t = QsoTracker()
    t.ingest(_over("EA1ABC", "CQ", text="CQ CQ from EA1ABC"))      # calling CQ
    t.ingest(_over("G4XYZ", "EA1ABC", report="59", name="Tom"))    # answer
    assert len(t.qsos) == 1
    q = t.qsos[0]
    assert q.calls() == {"EA1ABC", "G4XYZ"}
    assert q.a.call == "EA1ABC" and q.a.heard          # CQ caller, heard
    assert q.b.call == "G4XYZ" and q.b.heard           # answered, heard
    assert q.b.name == "Tom"
    assert q.report_b_to_a == "59"                      # G4XYZ -> EA1ABC


def test_both_sides_heard_accumulate():
    t = QsoTracker()
    t.ingest(_over("HB9IKS", "DL1ABC", report="599", qth="Bern"))
    t.ingest(_over("DL1ABC", "HB9IKS", report="589", name="Hans", qth="Berlin"))
    assert len(t.qsos) == 1
    q = t.qsos[0]
    assert q.a.qth == "Bern" and q.b.name == "Hans" and q.b.qth == "Berlin"
    assert q.report_a_to_b == "599" and q.report_b_to_a == "589"
    assert q.a.heard and q.b.heard


def test_other_side_named_but_not_heard():
    t = QsoTracker()
    # We only ever copy HB9IKS; the station it works is named but never heard.
    t.ingest(_over("HB9IKS", "DL1ABC", report="59"))
    q = t.qsos[0]
    assert q.a.call == "HB9IKS" and q.a.heard
    assert q.b.call == "DL1ABC" and not q.b.heard


def test_separate_pairs_are_separate_qsos():
    t = QsoTracker()
    t.ingest(_over("EA1ABC", "G4XYZ"))
    t.ingest(_over("I2AAA", "F5BBB"))
    assert len(t.qsos) == 2


def test_new_pair_with_shared_station_is_new_qso():
    t = QsoTracker()
    t.ingest(_over("HB9IKS", "DL1ABC"))     # QSO 1
    t.ingest(_over("DL1ABC", "HB9IKS"))     # still QSO 1 (both heard)
    t.ingest(_over("HB9IKS", "G4XYZ"))      # HB9IKS now works someone else -> QSO 2
    assert len(t.qsos) == 2
    assert t.qsos[0].calls() == {"HB9IKS", "DL1ABC"}
    assert t.qsos[1].calls() == {"HB9IKS", "G4XYZ"}


def test_different_band_does_not_merge():
    t = QsoTracker()
    t.ingest(OverRecord(speaker="HB9IKS", addressed="DL1ABC", band="20m"))
    t.ingest(OverRecord(speaker="HB9IKS", addressed="DL1ABC", band="40m"))
    assert len(t.qsos) == 2


def test_unattributable_over_is_dropped():
    t = QsoTracker()
    assert t.ingest(_over(None, "CQ")) is None
    assert t.qsos == []


def test_to_dict_shape():
    t = QsoTracker()
    q = t.ingest(_over("EA1ABC", "G4XYZ", report="59"))
    d = q.to_dict()
    assert d["kind"] == "qso" and d["id"] == 1
    assert d["a"]["call"] == "EA1ABC" and d["b"]["call"] == "G4XYZ"
    assert d["band"] == "20m" and d["freq_mhz"] == 14.2
