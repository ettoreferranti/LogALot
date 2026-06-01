from logalot.validate import (
    confidence_flag,
    expand_phonetics,
    is_valid_call,
    looks_like_callsign,
    normalise_call,
)


def test_expand_phonetics_callsign():
    assert expand_phonetics("hotel bravo nine india kilo sierra") == "HB9IKS"


def test_expand_phonetics_mixed_punctuation():
    assert expand_phonetics("Whiskey-One, Alpha Whiskey.") == "W1AW"


def test_valid_calls():
    for c in ["HB9IKS", "W1AW", "4X4AA", "2E0ABC", "DL1ABC"]:
        assert is_valid_call(c), c


def test_invalid_calls():
    for c in ["HELLO", "CQ", "123", ""]:
        assert not is_valid_call(c), c


def test_normalise_strips_portable_affix():
    core, affixes = normalise_call("HB9IKS/P")
    assert core == "HB9IKS"
    assert affixes == ["P"]


def test_normalise_prefix_form():
    core, affixes = normalise_call("DL/HB9IKS")
    assert core == "HB9IKS"
    assert "DL" in affixes


def test_confidence_flag():
    assert confidence_flag("HB9IKS") == "ok"
    assert confidence_flag("not-a-call") == "review"


def test_looks_like_callsign_rejects_cq_and_qcodes():
    assert looks_like_callsign("HB9IKS")
    assert looks_like_callsign("w1aw")
    for junk in ["CQ", "CQDX", "QRZ", "DX", "QTH", "73"]:
        assert not looks_like_callsign(junk), junk
